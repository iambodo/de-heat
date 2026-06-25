#!/usr/bin/env python3
"""
CHAP predict entry point for the Heat Mortality Germany model.

Usage:
    python predict.py <model_json> <historic_data_csv> <future_data_csv> <out_file_csv>

Input CSVs (from chap-core):
    time_period       — ISO week e.g. "2024W27" or "2024-W27"
    location          — DHIS2 Bundesland org unit UID
    mean_temperature  — weekly mean 2m temperature (°C)

Output CSV:
    time_period, location, sample_0 … sample_99
"""
import json, sys, os, pickle, zipfile
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

N_SAMPLES = 100
KERNEL_DAYS = 6

BL_UID_TO_IDX = {
    'sTHbKLIUiJQ': 0,   # Schleswig-Holstein
    'fAJqvNnlzCz': 1,   # Hamburg
    'Oe1a2DZbDqa': 2,   # Niedersachsen
    'S6YiLzHwyRS': 3,   # Bremen
    'yLvMb8w7bzj': 4,   # Nordrhein-Westfalen
    'xMmHrbkMQxr': 5,   # Hessen
    'zaf1vUB8JQr': 6,   # Rheinland-Pfalz
    'DqYmO9PfYbM': 7,   # Baden-Württemberg
    'FGE1fIzw6BE': 8,   # Bayern
    'twL5PpAyM0g': 9,   # Saarland
    'S0gZ79eJSx8': 10,  # Berlin
    'hy1y1zKBxXo': 11,  # Brandenburg
    'tzcHKEmmNli': 12,  # Mecklenburg-Vorpommern
    'T7xo5gzevDD': 13,  # Sachsen
    'oazLtnGqAPR': 14,  # Sachsen-Anhalt
    'PrgdgbVZr2d': 15,  # Thüringen
}
BL_IDX_TO_UID = {v: k for k, v in BL_UID_TO_IDX.items()}
BL_KREIS_COUNTS = [15, 1, 45, 2, 53, 26, 36, 44, 96, 6, 1, 18, 8, 13, 14, 22]


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
    return out.reshape(out.shape[0], 400, 15, 2)


def apply_death_pred(outputs, weights, bdc):
    n_days = outputs.shape[0]
    wc   = np.concatenate([[1.0], weights['weekday_correction']])
    corr = np.tile(wc, n_days // 7 + 1)[:n_days][:, np.newaxis, np.newaxis, np.newaxis]
    factor = (0.99 + np.clip(outputs, -0.99, None)) \
           + 0.01 * np.exp(np.clip(outputs, None, -0.99) + 0.99)
    return factor * corr * bdc


def load_baseline(mort_dir):
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
    base_m = np.sort(dm[:, win], axis=1)[:, :k].sum(1) / (k * 7)
    base_w = np.sort(dw[:, win], axis=1)[:, :k].sum(1) / (k * 7)
    germany_daily = base_m.sum() + base_w.sum()
    return germany_daily / 400 / 30


# ── Period helpers ────────────────────────────────────────────────────────────

def normalize_period(p):
    """Normalize to YYYYWnn. Handles '2024W27', '2024-W27', and DHIS2
    date-range format '2023-01-02/2023-01-08' (uses the start date)."""
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

    print("Loading model weights …")
    weights  = load_weights(config['checkpoint'])
    baseline = load_baseline(config['mort_dir'])

    hist_df = pd.read_csv(historic_data_path)
    fut_df  = pd.read_csv(future_data_path)

    hist_df['time_period'] = hist_df['time_period'].apply(normalize_period)
    fut_df['time_period']  = fut_df['time_period'].apply(normalize_period)

    all_df      = pd.concat([hist_df, fut_df], ignore_index=True)
    all_periods = sorted(all_df['time_period'].unique(), key=period_sort_key)
    fut_set     = set(fut_df['time_period'].unique())

    # Build temperature matrix [400, n_days]
    n_days      = len(all_periods) * 7
    temp_kreise = np.zeros((400, n_days))

    for t_idx, period in enumerate(all_periods):
        week_df   = all_df[all_df['time_period'] == period]
        day_slice = slice(t_idx * 7, (t_idx + 1) * 7)
        bl_temps  = np.full(16, np.nan)
        for _, row in week_df.iterrows():
            bl_idx = BL_UID_TO_IDX.get(str(row['location']))
            if bl_idx is not None and 'mean_temperature' in row.index:
                val = row['mean_temperature']
                if pd.notna(val):
                    bl_temps[bl_idx] = float(val)
        nan_mask = np.isnan(bl_temps)
        if nan_mask.any():
            fill = float(np.nanmean(bl_temps)) if not nan_mask.all() else 15.0
            bl_temps[nan_mask] = fill
        kreis_idx = 0
        for bl_i, count in enumerate(BL_KREIS_COUNTS):
            temp_kreise[kreis_idx:kreis_idx + count, day_slice] = bl_temps[bl_i]
            kreis_idx += count

    # Forward pass
    temp_norm   = temp_kreise / 15.0
    raw_out     = forward(temp_norm, weights)             # [n_days-5, 400, 15, 2]
    n_pred      = raw_out.shape[0]
    bdc         = np.full((n_pred, 400, 15, 2), baseline)
    daily_full  = apply_death_pred(raw_out, weights, bdc)
    daily_kreise = daily_full.sum(axis=(2, 3))            # [n_pred, 400]

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

        week_kreise = daily_kreise[day_start:day_end].sum(axis=0)   # [400]

        kreis_idx = 0
        for bl_i, count in enumerate(BL_KREIS_COUNTS):
            bl_uid = BL_IDX_TO_UID.get(bl_i)
            if bl_uid is None:
                kreis_idx += count; continue
            weekly_mean = float(week_kreise[kreis_idx:kreis_idx + count].sum())
            lam = max(weekly_mean, 1.0)
            samples = np.random.poisson(lam, N_SAMPLES).astype(float)
            row = {'time_period': period, 'location': bl_uid}
            row.update({f'sample_{i}': s for i, s in enumerate(samples)})
            rows.append(row)
            kreis_idx += count

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Predictions written: {len(out_df)} rows → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict.py <model_json> <historic_data> <future_data> <out_file>")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
