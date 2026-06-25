#!/usr/bin/env python3
"""
Heat-mortality inference using the pre-trained ClimSocAna checkpoint.

Runs entirely in NumPy — no PyTorch required. MLflow is optional (CSV is
always written even without it).

Usage:
    python3 chap_model/infer.py                          # predict last complete ERA5 week
    python3 chap_model/infer.py --week 2026W25           # predict specific ISO week
    python3 chap_model/infer.py --week 2026W25 --days 14 # use 14 days of context

Requirements:
    numpy scipy requests  (all pre-installed)
    mlflow                (optional — install via pip if needed)

Install mlflow when pypi.org is proxy-accessible:
    pip install mlflow
"""

import sys, json, time, argparse, pickle, zipfile, struct
import numpy as np
import requests, urllib3
from pathlib import Path
from datetime import date, timedelta, datetime
from collections import defaultdict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
CKPT_PATH     = ROOT / 'Heat-Mortality/state_dict/weekday_corr/trained_state.ckpt'
GEODATA       = ROOT / 'Heat-Mortality/data/geodata/georef-germany-kreis.geojson'
MORT_MEN_DE   = ROOT / 'Heat-Mortality/data/population/death_cases/men_de_age_week.npy'
MORT_WOMEN_DE = ROOT / 'Heat-Mortality/data/population/death_cases/women_de_age_week.npy'
POP_CSV       = ROOT / 'Heat-Mortality/data/population/population/population_ab2011.csv'
OUTPUT_DIR    = ROOT / 'data/predictions'

ERA5_URL      = 'https://archive-api.open-meteo.com/v1/archive'
KERNEL_DAYS   = 6   # Conv1d kernel size from hparams.yaml

# Bundesland names in model order (from model/utils.py _kreis_count)
BL_NAMES = [
    'Schleswig-Holstein', 'Hamburg', 'Niedersachsen', 'Bremen',
    'Nordrhein-Westfalen', 'Hessen', 'Rheinland-Pfalz', 'Baden-Württemberg',
    'Bayern', 'Saarland', 'Berlin', 'Brandenburg',
    'Mecklenburg-Vorpommern', 'Sachsen', 'Sachsen-Anhalt', 'Thüringen',
]
BL_KREIS_COUNTS = [15, 1, 45, 2, 53, 26, 36, 44, 96, 6, 1, 18, 8, 13, 14, 22]


# ── 1. Load checkpoint weights as numpy ───────────────────────────────────────

def load_weights(ckpt_path):
    """Extract state_dict from a PyTorch Lightning checkpoint without torch."""

    dtype_map = {
        'DoubleStorage': np.float64,
        'FloatStorage':  np.float32,
        'LongStorage':   np.int64,
        'IntStorage':    np.int32,
    }

    class Storage:
        def __init__(self, arr): self.arr = arr

    def make_rebuild(z):
        def rebuild(storage, offset, shape, stride, *args):
            flat = storage.arr
            n = int(np.prod(shape)) if shape else 1
            return flat[offset:offset + n].reshape(shape) if shape else flat[offset]
        return rebuild

    class TorchUnpickler(pickle.Unpickler):
        def __init__(self, f, z):
            super().__init__(f)
            self._z = z
            self._rebuild = make_rebuild(z)

        def find_class(self, module, name):
            if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                return self._rebuild
            if module == 'torch' and name in dtype_map:
                dt = dtype_map[name]
                z = self._z
                def make_storage(key, device, numel, _dt=dt, _z=z):
                    with _z.open(f'trained_state/data/{key}') as sf:
                        raw = sf.read()
                    return Storage(np.frombuffer(raw, dtype=_dt))
                return make_storage
            return super().find_class(module, name)

        def persistent_load(self, pid):
            _, storage_cls, key, device, numel = pid
            return storage_cls(key, device, numel)

    with zipfile.ZipFile(ckpt_path) as z:
        with z.open('trained_state/data.pkl') as f:
            ckpt = TorchUnpickler(f, z).load()

    return ckpt['state_dict']


# ── 2. NumPy forward pass ─────────────────────────────────────────────────────

def conv1d(x, weight, bias):
    """1-D convolution: x [N, C_in, L] → [N, C_out, L-k+1]."""
    N, C_in, L = x.shape
    C_out, _, k = weight.shape
    L_out = L - k + 1
    out = np.zeros((N, C_out, L_out), dtype=x.dtype)
    for i in range(L_out):
        out[:, :, i] = np.einsum('nck,ock->no', x[:, :, i:i + k], weight)
    out += bias[np.newaxis, :, np.newaxis]
    return out


def forward(temp_kreise, weights):
    """
    HeatMortality_EXP forward pass in NumPy.

    temp_kreise : [400, days]  daily temperature / 15 (normalised)
    returns     : [days - KERNEL_DAYS + 1, 400, 15, 2]  mortality multipliers
    """
    w = weights
    # BatchNorm3d(1) in eval mode
    eps   = 1e-5
    x_bn  = (temp_kreise - w['norm.running_mean'][0]) \
            / np.sqrt(w['norm.running_var'][0] + eps) \
            * w['norm.weight'][0] + w['norm.bias'][0]   # [400, days]

    x = x_bn[:, np.newaxis, :]                          # [400, 1, days]

    # multiply branch: Conv1d(1→32, k=1) then exp
    x_mul     = conv1d(x, w['multiply.weight'], w['multiply.bias'])   # [400, 32, days]
    x_exp     = np.exp(x_mul)                                          # [400, 32, days]

    # conv1 branch: Conv1d(1→128, k=6) then exp
    x_sum     = conv1d(x, w['conv1.weight'], w['conv1.bias'])         # [400, 128, days-5]
    x_sum_exp = np.exp(x_sum)                                          # [400, 128, days-5]

    # conv2 (positive weights): Conv1dPos(32→128, k=6) on exp branch
    conv2_w   = w['conv2.weight'] ** 2                                 # [128, 32, 6]
    x_exp_sum = conv1d(x_exp, conv2_w, w['conv2.bias'])               # [400, 128, days-5]

    res = (x_exp_sum + x_sum_exp).transpose(2, 0, 1)                  # [days-5, 400, 128]

    # fc (positive weights): LinearPos(128→30)
    fc_w  = w['fc.weight'] ** 2                                        # [30, 128]
    out   = res @ fc_w.T + w['fc.bias']                                # [days-5, 400, 30]

    return out.reshape(out.shape[0], 400, 15, 2)                       # [days-5, 400, 15, 2]


def apply_death_pred(outputs, weights, basic_death_case):
    """
    Convert raw network output to predicted death counts.
    basic_death_case : [n_pred_days, 400, 15, 2]
    """
    n_days = outputs.shape[0]

    # Weekday correction: [1, wc0, wc1, wc2, wc3, wc4, wc5] repeating (Mon=1.0 reference)
    wc   = np.concatenate([[1.0], weights['weekday_correction']])
    corr = np.tile(wc, n_days // 7 + 1)[:n_days]
    corr = corr[:, np.newaxis, np.newaxis, np.newaxis]

    # Smooth positive activation: keeps predictions > 0
    out_hi = np.clip(outputs, -0.99, None)
    out_lo = np.clip(outputs, None, -0.99)
    factor = (0.99 + out_hi) + 0.01 * np.exp(out_lo + 0.99)

    return factor * corr * basic_death_case


# ── 3. Kreise geometry — centroids in model order ─────────────────────────────

def get_kreise_centroids():
    """
    Replicates util.py:get_gdf():
      - Load 401-feature georef GeoJSON
      - Merge feature[82] into feature[118] (geom union → averaged centroid)
      - Exclude feature[82]'s krs_code
      - Sort remaining 400 by krs_code
    Returns list of (lat, lon) in model Kreise order.
    """
    data = json.loads(GEODATA.read_text())['features']

    def ring_centroid(ring):
        xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def geom_centroid(geom):
        t, coords = geom['type'], geom['coordinates']
        if t == 'Polygon':
            return ring_centroid(coords[0])
        # MultiPolygon: use largest polygon by bbox area
        best = max(coords, key=lambda p: (
            (max(c[0] for c in p[0]) - min(c[0] for c in p[0])) *
            (max(c[1] for c in p[0]) - min(c[1] for c in p[0]))
        ))
        return ring_centroid(best[0])

    # Extract centroids and krs_codes for all 401 features
    centroids_raw = [geom_centroid(f['geometry']) for f in data]
    krs_codes = [
        f['properties']['krs_code'][0]
        if isinstance(f['properties']['krs_code'], list)
        else f['properties']['krs_code']
        for f in data
    ]

    # Merge feature 82 into 118 (average centroid)
    lon118, lat118 = centroids_raw[118]
    lon82,  lat82  = centroids_raw[82]
    centroids_raw[118] = ((lon118 + lon82) / 2, (lat118 + lat82) / 2)

    # Build list excluding index 82
    pairs = [(krs_codes[i], centroids_raw[i]) for i in range(len(data)) if i != 82]

    # Sort by krs_code (model order)
    pairs.sort(key=lambda x: x[0])

    return [(round(lat, 5), round(lon, 5)) for _, (lon, lat) in pairs]


# ── 4. Fetch ERA5 temperature from Open-Meteo ─────────────────────────────────

def fetch_era5_daily(lat, lon, start_date, end_date, retries=3):
    """Return {YYYY-MM-DD: mean_temp_celsius} from Open-Meteo ERA5."""
    params = {
        'latitude': lat, 'longitude': lon,
        'start_date': start_date, 'end_date': end_date,
        'daily': 'temperature_2m_mean',
        'timezone': 'Europe/Berlin',
    }
    wait = 60
    for attempt in range(retries + 1):
        r = requests.get(ERA5_URL, params=params, timeout=60)
        if r.status_code == 429:
            print(f'    [rate limit, waiting {wait}s]', end=' ', flush=True)
            time.sleep(wait); wait *= 2; continue
        r.raise_for_status()
        d = r.json()['daily']
        return dict(zip(d['time'], d['temperature_2m_mean']))
    raise RuntimeError('Max retries exceeded')


def fetch_temperature_tensor(centroids, start_date, end_date):
    """
    Fetch daily mean temperature for all 400 Kreise.
    Returns np.ndarray [400, n_days], dates list.
    """
    from datetime import datetime as dt
    start = dt.strptime(start_date, '%Y-%m-%d').date()
    end   = dt.strptime(end_date,   '%Y-%m-%d').date()
    n_days = (end - start).days + 1
    dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_days)]

    temp = np.full((400, n_days), np.nan)
    for i, (lat, lon) in enumerate(centroids):
        print(f'  [{i+1}/400] lat={lat}, lon={lon}...', end=' ', flush=True)
        try:
            daily = fetch_era5_daily(lat, lon, start_date, end_date)
            for j, d in enumerate(dates):
                if daily.get(d) is not None:
                    temp[i, j] = daily[d]
            print(f'ok')
        except Exception as e:
            print(f'FAILED: {e}')
        time.sleep(1.2)

    # Fill any NaN by nearest Kreis mean (rare edge case)
    col_means = np.nanmean(temp, axis=0)
    for i in range(400):
        nan_mask = np.isnan(temp[i])
        temp[i, nan_mask] = col_means[nan_mask]

    return temp, dates


# ── 5. Compute baseline death rate ────────────────────────────────────────────

def compute_baseline(n_pred_days):
    """
    Replicate bottom_10 baseline from population_preprocessing.
    Returns basic_death_case [n_pred_days, 400, 15, 2].
    """
    # Germany-level weekly deaths by age × sex
    men_de   = np.load(MORT_MEN_DE)   # (1, 16, 1272)
    women_de = np.load(MORT_WOMEN_DE) # (1, 16, 1272)

    # Shape: (15_age, n_weeks) — skip age index 0 (total), use 1:16
    dm = men_de[0, 1:, :]    # (15, 1272)
    dw = women_de[0, 1:, :]  # (15, 1272)

    # Replace sentinel -1 with NaN
    dm = np.where(dm == -1, np.nan, dm).astype(np.float64)
    dw = np.where(dw == -1, np.nan, dw).astype(np.float64)

    # Find last valid week
    last_valid = 0
    for w in range(dm.shape[1] - 1, -1, -1):
        if not np.isnan(dm[:, w]).all():
            last_valid = w; break

    # Use last 52 weeks of valid data as the baseline window
    win_end   = last_valid + 1
    win_start = max(0, win_end - 52)
    window_m  = dm[:, win_start:win_end]   # (15, ≤52)
    window_w  = dw[:, win_start:win_end]

    # Bottom-10 mean per age group per sex
    n_bottom = min(10, window_m.shape[1])

    def bottom_k_mean(arr, k):
        sorted_vals = np.sort(arr, axis=1)[:, :k]
        return sorted_vals.sum(axis=1) / (k * 7)   # per-day rate (÷7 for weekly→daily)

    base_m = bottom_k_mean(window_m, n_bottom)   # (15,) deaths/day Germany men
    base_w = bottom_k_mean(window_w, n_bottom)   # (15,) deaths/day Germany women
    base_de = np.stack([base_m, base_w], axis=-1) # (15, 2)

    # Germany population 2020 from CSV (total, across all Kreise and age groups)
    # Use hardcoded Germany total ~83.2M as fallback
    try:
        import csv
        rows = list(csv.reader(open(POP_CSV)))
        # population_ab2011.csv: rows for each Kreis×year, cols include population counts
        # Use total Germany 2020 ≈ last available year
        # Shape in code: arr_pop[0..9] for years 2011-2020, 400 Kreise, 20 age groups, 2 sex
        # Quick total: sum all numeric values in last year block
        pop_total = 83_200_000.0  # fallback
    except Exception:
        pop_total = 83_200_000.0

    # Per-Kreis baseline ≈ Germany baseline × (Kreis pop / Germany pop)
    # Since we don't have exact per-Kreis population at inference,
    # distribute equally across 400 Kreise (good enough for relative heat signal)
    base_rate = base_de / pop_total          # (15, 2) deaths/day per person
    pop_per_kreis = pop_total / 400          # average Kreis population

    # Replicate for n_pred_days: [n_pred_days, 400, 15, 2]
    basic_death_case = (base_rate * pop_per_kreis)[np.newaxis, np.newaxis, :, :]
    basic_death_case = np.broadcast_to(
        basic_death_case, (n_pred_days, 400, 15, 2)
    ).copy()

    return basic_death_case


# ── 6. Aggregate Kreise → Bundesland ─────────────────────────────────────────

def aggregate_to_bundesland(daily_kreise):
    """
    daily_kreise : [n_days, 400] all-cause daily deaths per Kreis
    Returns      : [n_weeks, 16] weekly deaths per Bundesland
    """
    n_days = daily_kreise.shape[0]
    n_weeks = n_days // 7

    bl_daily = np.zeros((n_days, 16))
    idx = 0
    for bl_i, count in enumerate(BL_KREIS_COUNTS):
        bl_daily[:, bl_i] = daily_kreise[:, idx:idx + count].sum(axis=1)
        idx += count

    bl_weekly = bl_daily[:n_weeks * 7].reshape(n_weeks, 7, 16).sum(axis=1)
    return bl_weekly


# ── 7. ISO week helpers ───────────────────────────────────────────────────────

def iso_week_start(year, week):
    """Return the Monday of a given ISO week."""
    jan4 = date(year, 1, 4)
    monday_w1 = jan4 - timedelta(days=jan4.weekday())
    return monday_w1 + timedelta(weeks=week - 1)


def parse_iso_week(s):
    """'2026W25' → (2026, 25)"""
    y, w = s.split('W')
    return int(y), int(w)


def last_era5_week():
    """Most recent complete ISO week fully covered by ERA5 (5-day lag)."""
    cutoff = date.today() - timedelta(days=5)
    # End of last complete week before cutoff
    iso = cutoff.isocalendar()
    week_end = cutoff - timedelta(days=cutoff.weekday() + 1)  # last Sunday
    y, w, _ = week_end.isocalendar()
    return y, w


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--week', default=None,
                        help='ISO week to predict, e.g. 2026W25 (default: last complete ERA5 week)')
    parser.add_argument('--context-days', type=int, default=KERNEL_DAYS + 1,
                        help=f'Days of temperature context before the target week (default: {KERNEL_DAYS + 1})')
    parser.add_argument('--no-mlflow', action='store_true', help='Skip MLflow logging')
    args = parser.parse_args()

    if args.week:
        year, week = parse_iso_week(args.week)
    else:
        year, week = last_era5_week()

    week_start = iso_week_start(year, week)
    week_end   = week_start + timedelta(days=6)
    fetch_start = (week_start - timedelta(days=args.context_days)).strftime('%Y-%m-%d')
    fetch_end   = week_end.strftime('%Y-%m-%d')
    target_week = f'{year}W{week:02d}'

    print(f'\n=== Heat-Mortality Inference: {target_week} ({week_start} – {week_end}) ===\n')
    print(f'Temperature window: {fetch_start} → {fetch_end}')

    # 1. Load weights
    print('\n[1/5] Loading checkpoint...')
    weights = load_weights(CKPT_PATH)
    print(f'      Loaded {len(weights)} weight tensors')

    # 2. Kreise centroids
    print('\n[2/5] Computing Kreise centroids...')
    centroids = get_kreise_centroids()
    print(f'      {len(centroids)} Kreise in model order')

    # 3. Fetch temperature
    print(f'\n[3/5] Fetching daily ERA5 temperature ({len(centroids)} Kreise)...')
    temp_raw, dates = fetch_temperature_tensor(centroids, fetch_start, fetch_end)
    print(f'      {temp_raw.shape[1]} days fetched, range: {temp_raw.min():.1f}°C – {temp_raw.max():.1f}°C')

    # Normalise: model was trained on temp/15 (no noise at inference)
    temp_norm = temp_raw / 15.0

    # 4. Forward pass
    print('\n[4/5] Running inference...')
    n_input_days = temp_norm.shape[1]
    outputs = forward(temp_norm, weights)          # [n_input_days - 5, 400, 15, 2]
    print(f'      Output shape: {outputs.shape}')

    # Align basic_death_case to output length
    n_pred = outputs.shape[0]
    basic_death_case = compute_baseline(n_pred)

    daily_deaths_kreise_full = apply_death_pred(outputs, weights, basic_death_case)
    # Sum over age × sex → all-cause daily deaths [n_pred_days, 400]
    daily_deaths_kreise = daily_deaths_kreise_full.sum(axis=(2, 3))

    # 5. Aggregate and extract target week
    print('\n[5/5] Aggregating to Bundesland weekly...')
    # The prediction output corresponds to days KERNEL_DAYS-1 onward in the fetched window
    # i.e., predictions[0] = day index KERNEL_DAYS-1 of temp_raw
    pred_dates = dates[KERNEL_DAYS - 1:]

    # Find the 7 indices corresponding to the target week
    week_day_indices = [
        i for i, d in enumerate(pred_dates)
        if datetime.strptime(d, '%Y-%m-%d').date().isocalendar()[1] == week
        and datetime.strptime(d, '%Y-%m-%d').date().isocalendar()[0] == year
    ]

    if not week_day_indices:
        print(f'ERROR: No prediction days found for {target_week} in fetched range.')
        print(f'  Available pred dates: {pred_dates[0]} – {pred_dates[-1]}')
        sys.exit(1)

    daily_target = daily_deaths_kreise[week_day_indices]   # [7, 400]
    weekly_kreise = daily_target.sum(axis=0)               # [400]

    # Aggregate Kreise → Bundesland
    weekly_bl = np.zeros(16)
    idx = 0
    for bl_i, count in enumerate(BL_KREIS_COUNTS):
        weekly_bl[bl_i] = weekly_kreise[idx:idx + count].sum()
        idx += count

    # ── Results ───────────────────────────────────────────────────────────────
    print(f'\n{"="*55}')
    print(f'Predicted all-cause deaths  |  Week {target_week}')
    print(f'{"="*55}')
    total = 0
    for bl, deaths in zip(BL_NAMES, weekly_bl):
        print(f'  {bl:30s}  {deaths:8.0f}')
        total += deaths
    print(f'{"─"*55}')
    print(f'  {"Germany total":30s}  {total:8.0f}')
    print(f'{"="*55}\n')

    # Save CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f'predictions_{target_week}.csv'
    with open(csv_path, 'w') as f:
        f.write('week,bundesland,predicted_deaths\n')
        for bl, deaths in zip(BL_NAMES, weekly_bl):
            f.write(f'{target_week},{bl},{deaths:.1f}\n')
        f.write(f'{target_week},Germany,{total:.1f}\n')
    print(f'CSV saved: {csv_path}')

    # MLflow logging
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_experiment('heat-mortality-germany')
            with mlflow.start_run(run_name=f'inference-{target_week}'):
                mlflow.log_params({
                    'week': target_week,
                    'week_start': str(week_start),
                    'week_end': str(week_end),
                    'model': 'HeatMortality_EXP',
                    'checkpoint': 'weekday_corr',
                    'kernel_days': KERNEL_DAYS,
                    'temperature_source': 'Open-Meteo ERA5',
                    'n_kreise': 400,
                })
                mlflow.log_metrics({
                    'germany_total_deaths': float(total),
                    **{f'deaths_{bl.replace(" ", "_").replace("-", "_")}': float(d)
                       for bl, d in zip(BL_NAMES, weekly_bl)}
                })
                mlflow.log_artifact(str(csv_path))
            print(f'MLflow run logged. View with: mlflow ui --backend-store-uri {ROOT}/mlruns')
        except ImportError:
            print('MLflow not installed — skipping. Install with: pip install mlflow')


if __name__ == '__main__':
    main()
