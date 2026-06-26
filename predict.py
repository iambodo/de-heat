#!/usr/bin/env python3
"""
CHAP predict entry point for the Heat Mortality Germany model.

Usage:
    python predict.py <model_json> <historic_data_csv> <future_data_csv> <out_file_csv>

Input CSVs (from chap-core):
    time_period       — ISO week e.g. "2024W27" or "2024-W27"
    location          — DHIS2 Kreis (district) org unit UID
    mean_temperature  — weekly mean 2m temperature (°C)

Output CSV:
    time_period, location, sample_0 … sample_99
"""
import json, sys, os, pickle, zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

N_SAMPLES = 100
KERNEL_DAYS = 6

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Population (Destatis 2023) per Bundesland, used to weight per-Kreis baseline
_BL_POP = {
    'sTHbKLIUiJQ': 2953243,   # Schleswig-Holstein
    'fAJqvNnlzCz': 1910160,   # Hamburg
    'Oe1a2DZbDqa': 8101307,   # Niedersachsen
    'S6YiLzHwyRS':  684862,   # Bremen
    'yLvMb8w7bzj': 18137620,  # Nordrhein-Westfalen
    'xMmHrbkMQxr': 6400732,   # Hessen
    'zaf1vUB8JQr': 4100023,   # Rheinland-Pfalz
    'DqYmO9PfYbM': 11286556,  # Baden-Württemberg
    'FGE1fIzw6BE': 13369393,  # Bayern
    'twL5PpAyM0g':  994187,   # Saarland
    'S0gZ79eJSx8': 3755251,   # Berlin
    'hy1y1zKBxXo': 2573040,   # Brandenburg
    'tzcHKEmmNli': 1634987,   # Mecklenburg-Vorpommern
    'T7xo5gzevDD': 4055274,   # Sachsen
    'oazLtnGqAPR': 2166382,   # Sachsen-Anhalt
    'PrgdgbVZr2d': 2115485,   # Thüringen
}
_GERMANY_POP = sum(_BL_POP.values())


def load_kreis_list():
    """Load Kreis list from committed JSON, return (uid_to_idx, idx_to_uid, bl_uid_per_kreis, coords)."""
    path = Path(REPO_ROOT) / 'model_artifacts' / 'kreis_list.json'
    kreise = json.loads(path.read_text())
    uid_to_idx = {k['uid']: k['idx'] for k in kreise}
    idx_to_uid = {k['idx']: k['uid'] for k in kreise}
    bl_uid = {k['idx']: k['bundesland_uid'] for k in kreise}
    # coords[idx] = (lat, lon) or None if missing
    coords = {k['idx']: (k['lat'], k['lon']) for k in kreise if 'lat' in k and 'lon' in k}
    return uid_to_idx, idx_to_uid, bl_uid, coords


def interpolate_missing_kreise(temp_kreise, missing_idx, coords, n_periods):
    """
    Fill rows for Kreise with no temperature data using inverse-distance
    weighted average of the 5 nearest Kreise that do have data, per week.
    """
    all_idx = list(range(temp_kreise.shape[0]))
    has_data = [i for i in all_idx if i not in missing_idx and i in coords]
    if not has_data:
        return

    has_data_coords = np.array([(coords[i][0], coords[i][1]) for i in has_data])

    for k_idx in missing_idx:
        if k_idx not in coords:
            continue
        q = np.array(coords[k_idx])
        # Haversine-approximate: treat lat/lon as Euclidean (fine for neighbors in Germany)
        diffs = has_data_coords - q
        dists = np.sqrt((diffs[:, 0] ** 2) + (diffs[:, 1] ** 2))
        k_nn = min(5, len(has_data))
        nn_pos = np.argpartition(dists, k_nn - 1)[:k_nn]
        nn_idx = [has_data[p] for p in nn_pos]
        nn_dists = dists[nn_pos]

        if nn_dists.min() == 0:
            weights = np.where(nn_dists == 0, 1.0, 0.0)
        else:
            weights = 1.0 / (nn_dists ** 2)
        weights /= weights.sum()

        for w in range(n_periods):
            day_slice = slice(w * 7, (w + 1) * 7)
            neighbor_means = np.array([temp_kreise[n, day_slice].mean() for n in nn_idx])
            valid = neighbor_means != 0
            if not valid.any():
                continue
            w_valid = weights[valid] / weights[valid].sum()
            temp_kreise[k_idx, day_slice] = (neighbor_means[valid] * w_valid).sum()


# ── Checkpoint loader (pure numpy, no torch) ──────────────────────────────────

def load_weights(ckpt_path):
    dtype_map = {
        'DoubleStorage': np.float64,
        'FloatStorage':  np.float32,
        'LongStorage':   np.int64,
    }

    class Storage:
        def __init__(self, arr): self.arr = arr

    def make_rebuild(z):
        def rebuild(storage, offset, shape, stride, *args):
            n = int(np.prod(shape)) if shape else 1
            return storage.arr[offset:offset + n].reshape(shape) if shape else storage.arr[offset]
        return rebuild

    class TorchUnpickler(pickle.Unpickler):
        def __init__(self, f, z):
            super().__init__(f); self._z = z; self._rb = make_rebuild(z)
        def find_class(self, module, name):
            if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                return self._rb
            if module == 'torch' and name in dtype_map:
                dt = dtype_map[name]; z = self._z
                def make_s(key, device, numel, _dt=dt, _z=z):
                    with _z.open(f'trained_state/data/{key}') as sf:
                        return Storage(np.frombuffer(sf.read(), dtype=_dt))
                return make_s
            return super().find_class(module, name)
        def persistent_load(self, pid):
            _, cls, key, device, numel = pid
            return cls(key, device, numel)

    with zipfile.ZipFile(ckpt_path) as z:
        with z.open('trained_state/data.pkl') as f:
            ckpt = TorchUnpickler(f, z).load()
    return ckpt['state_dict']


# ── NumPy forward pass ────────────────────────────────────────────────────────

def conv1d(x, weight, bias):
    N, C_in, L = x.shape
    C_out, _, k = weight.shape
    L_out = L - k + 1
    out = np.zeros((N, C_out, L_out), dtype=x.dtype)
    for i in range(L_out):
        out[:, :, i] = np.einsum('nck,ock->no', x[:, :, i:i + k], weight)
    return out + bias[np.newaxis, :, np.newaxis]


def forward(temp_kreise, weights):
    """temp_kreise: [N_kreise, n_days]. Returns [n_days-5, N_kreise, 15, 2]."""
    w = weights
    x_bn = ((temp_kreise - w['norm.running_mean'][0])
            / np.sqrt(w['norm.running_var'][0] + 1e-5)
            * w['norm.weight'][0] + w['norm.bias'][0])
    x = x_bn[:, np.newaxis, :]
    x_mul     = conv1d(x, w['multiply.weight'], w['multiply.bias'])
    x_exp     = np.exp(x_mul)
    x_sum     = conv1d(x, w['conv1.weight'], w['conv1.bias'])
    x_sum_exp = np.exp(x_sum)
    x_exp_sum = conv1d(x_exp, w['conv2.weight'] ** 2, w['conv2.bias'])
    res = (x_exp_sum + x_sum_exp).transpose(2, 0, 1)
    out = res @ (w['fc.weight'] ** 2).T + w['fc.bias']
    return out.reshape(*out.shape[:2], 15, 2)


def apply_death_pred(outputs, weights, bdc):
    n_days = outputs.shape[0]
    wc   = np.concatenate([[1.0], weights['weekday_correction']])
    corr = np.tile(wc, n_days // 7 + 1)[:n_days][:, np.newaxis, np.newaxis, np.newaxis]
    factor = (0.99 + np.clip(outputs, -0.99, None)) \
           + 0.01 * np.exp(np.clip(outputs, None, -0.99) + 0.99)
    return factor * corr * bdc


def load_baseline(mort_dir, n_kreise, bl_uid_per_kreis):
    """Returns per-Kreis baseline [n_kreise, 15, 2]: daily deaths per age-sex group."""
    mort_dir = Path(mort_dir)
    men   = np.load(mort_dir / 'men_de_age_week.npy')
    women = np.load(mort_dir / 'women_de_age_week.npy')
    dm = np.where(men[0, 1:, :]   == -1, np.nan, men[0, 1:, :]).astype(np.float64)
    dw = np.where(women[0, 1:, :] == -1, np.nan, women[0, 1:, :]).astype(np.float64)
    last = 0
    for w in range(dm.shape[1] - 1, -1, -1):
        if not np.isnan(dm[:, w]).all():
            last = w; break
    win = slice(max(0, last - 51), last + 1)
    k = min(10, last + 1)
    base_m = np.sort(dm[:, win], axis=1)[:, :k].sum(1) / (k * 7)  # [15] daily per age
    base_w = np.sort(dw[:, win], axis=1)[:, :k].sum(1) / (k * 7)  # [15] daily per age
    germany_daily_age = np.stack([base_m, base_w], axis=-1)        # [15, 2]

    # Count Kreise per Bundesland to split BL share equally
    from collections import Counter
    bl_kreis_counts = Counter(bl_uid_per_kreis.values())

    bdc = np.zeros((n_kreise, 15, 2))
    for idx, bl_uid in bl_uid_per_kreis.items():
        bl_pop_share = _BL_POP.get(bl_uid, 0) / _GERMANY_POP
        n_bl_kreise  = bl_kreis_counts[bl_uid]
        bdc[idx] = germany_daily_age * bl_pop_share / n_bl_kreise

    return bdc


# ── Period helpers ────────────────────────────────────────────────────────────

def normalize_period(p):
    p = str(p)
    if '/' in p:
        start = p.split('/')[0]
        dt = datetime.strptime(start, '%Y-%m-%d')
        iso_year, iso_week, _ = dt.isocalendar()
        return f"{iso_year}W{iso_week:02d}"
    return p.replace('-W', 'W')


def period_sort_key(p):
    p = normalize_period(p)
    y, w = int(p[:4]), int(p[5:])
    return y * 100 + w


# ── Main ──────────────────────────────────────────────────────────────────────

def predict(model_path, historic_data_path, future_data_path, out_path):
    with open(model_path) as f:
        config = json.load(f)

    uid_to_idx, idx_to_uid, bl_uid_per_kreis, kreis_coords = load_kreis_list()
    n_kreise = len(uid_to_idx)
    print(f"Loaded {n_kreise} Kreise")

    print("Loading model weights …")
    weights  = load_weights(config['checkpoint'])
    baseline = load_baseline(config['mort_dir'], n_kreise, bl_uid_per_kreis)

    hist_df = pd.read_csv(historic_data_path)
    fut_df  = pd.read_csv(future_data_path)

    hist_df['time_period'] = hist_df['time_period'].apply(normalize_period)
    fut_df['time_period']  = fut_df['time_period'].apply(normalize_period)

    all_df      = pd.concat([hist_df, fut_df], ignore_index=True)
    all_periods = sorted(all_df['time_period'].unique(), key=period_sort_key)
    fut_set     = set(fut_df['time_period'].unique())

    # Build temperature matrix [n_kreise, n_days]
    n_days      = len(all_periods) * 7
    temp_kreise = np.zeros((n_kreise, n_days))

    unknown_locs = set()
    kreise_with_data = set()
    for t_idx, period in enumerate(all_periods):
        week_df   = all_df[all_df['time_period'] == period]
        day_slice = slice(t_idx * 7, (t_idx + 1) * 7)
        for _, row in week_df.iterrows():
            k_idx = uid_to_idx.get(str(row['location']))
            if k_idx is None:
                unknown_locs.add(str(row['location']))
                continue
            if 'mean_temperature' in row.index and pd.notna(row['mean_temperature']):
                temp_kreise[k_idx, day_slice] = float(row['mean_temperature'])
                kreise_with_data.add(k_idx)

    if unknown_locs:
        print(f"  Warning: {len(unknown_locs)} unknown location UIDs ignored: {list(unknown_locs)[:5]}")

    missing_kreise = set(range(n_kreise)) - kreise_with_data
    if missing_kreise:
        names = [idx_to_uid[i] for i in sorted(missing_kreise)]
        print(f"  Interpolating {len(missing_kreise)} Kreise with no temperature data from nearest neighbors")
        print(f"    UIDs: {names[:5]}{'...' if len(names) > 5 else ''}")
        interpolate_missing_kreise(temp_kreise, missing_kreise, kreis_coords, len(all_periods))

    # Forward pass
    temp_norm  = temp_kreise / 15.0
    raw_out    = forward(temp_norm, weights)                   # [n_days-5, n_kreise, 15, 2]
    n_pred     = raw_out.shape[0]
    bdc        = np.broadcast_to(baseline[np.newaxis], (n_pred, n_kreise, 15, 2)).copy()
    daily_full = apply_death_pred(raw_out, weights, bdc)
    daily_kreise = daily_full.sum(axis=(2, 3))                 # [n_pred, n_kreise]

    pred_start = KERNEL_DAYS - 1

    rows = []
    for t_idx, period in enumerate(all_periods):
        if period not in fut_set:
            continue
        day_start = t_idx * 7 - pred_start
        day_end   = day_start + 7
        if day_start < 0 or day_end > n_pred:
            print(f"  Skipping {period}: not enough context ({day_start}–{day_end} vs {n_pred})")
            continue

        week_kreise = daily_kreise[day_start:day_end].sum(axis=0)  # [n_kreise]
        chap_period = period[:4] + '-W' + period[5:]

        for k_idx in range(n_kreise):
            uid = idx_to_uid[k_idx]
            lam = max(float(week_kreise[k_idx]), 1.0)
            samples = np.random.poisson(lam, N_SAMPLES).astype(float)
            row = {'time_period': chap_period, 'location': uid}
            row.update({f'sample_{i}': s for i, s in enumerate(samples)})
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Predictions written: {len(out_df)} rows → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict.py <model_json> <historic_data> <future_data> <out_file>")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
