"""
MLflow PythonModel wrapper for the Heat-Mortality model.

CHAP calls model.predict(context, input_df) where input_df has columns:
  time_period     — ISO week string e.g. "2024W27"
  location        — DHIS2 Bundesland UID
  mean_temperature — weekly mean 2m temperature (°C)
  population      — annual population count

Returns DataFrame with columns: time_period, location, mean, low, high
"""

import pickle, zipfile, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta, datetime


# ── Bundesland UID → model BL index ──────────────────────────────────────────
# Order matches _kreis_count in model/utils.py:
#   SH, HH, NI, HB, NW, HE, RP, BW, BY, SL, BE, BB, MV, SN, ST, TH

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
KERNEL_DAYS = 6


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

    x_mul      = conv1d(x, w['multiply.weight'], w['multiply.bias'])
    x_exp      = np.exp(x_mul)
    x_sum      = conv1d(x, w['conv1.weight'], w['conv1.bias'])
    x_sum_exp  = np.exp(x_sum)
    x_exp_sum  = conv1d(x_exp, w['conv2.weight'] ** 2, w['conv2.bias'])

    res = (x_exp_sum + x_sum_exp).transpose(2, 0, 1)
    out = res @ (w['fc.weight'] ** 2).T + w['fc.bias']
    return out.reshape(out.shape[0], 400, 15, 2)


def apply_death_pred(outputs, weights, basic_death_case):
    n_days = outputs.shape[0]
    wc   = np.concatenate([[1.0], weights['weekday_correction']])
    corr = np.tile(wc, n_days // 7 + 1)[:n_days][:, np.newaxis, np.newaxis, np.newaxis]
    factor = (0.99 + np.clip(outputs, -0.99, None)) \
           + 0.01 * np.exp(np.clip(outputs, None, -0.99) + 0.99)
    return factor * corr * basic_death_case


# ── Baseline ──────────────────────────────────────────────────────────────────

def load_baseline(mort_dir):
    """
    Returns basic_death_case_daily: scalar float.
    Expected daily all-cause deaths per Kreis (average across Germany / 400).
    Used as a fixed scalar baseline — refreshed at model load time.
    """
    mort_dir = Path(mort_dir)
    men   = np.load(mort_dir / 'men_de_age_week.npy')   # (1, 16, 1272)
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

    # Germany total daily ÷ 400 Kreise × 30 (15 age × 2 sex channels)
    germany_daily = base_m.sum() + base_w.sum()
    per_kreis_per_channel = germany_daily / 400 / 30
    return per_kreis_per_channel


# ── ISO week helpers ──────────────────────────────────────────────────────────

def week_to_monday(period_str):
    """'2024W27' → date of that Monday."""
    y, w = int(period_str[:4]), int(period_str[5:])
    jan4 = date(y, 1, 4)
    return jan4 - timedelta(days=jan4.weekday()) + timedelta(weeks=w - 1)


def date_to_period(d):
    y, w, _ = d.isocalendar()
    return f'{y}W{w:02d}'


# ── Main MLflow PythonModel ───────────────────────────────────────────────────

try:
    import mlflow.pyfunc

    class HeatMortalityModel(mlflow.pyfunc.PythonModel):
        """
        CHAP-compatible MLflow pyfunc wrapper for the ClimSocAna heat-mortality
        neural network.

        Artifacts required (set in package_model.py):
          checkpoint  — path to trained_state.ckpt
          mort_dir    — directory containing men/women_de_age_week.npy
        """

        def load_context(self, context):
            self.weights   = load_weights(context.artifacts['checkpoint'])
            self.baseline  = load_baseline(context.artifacts['mort_dir'])
            print('HeatMortalityModel: weights and baseline loaded.')

        def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
            """
            model_input columns: time_period, location, mean_temperature
                                 (population is optional, not used by this model)

            Returns DataFrame: time_period, location, mean, low, high
            """
            df = model_input.copy()

            # Validate required columns
            for col in ('time_period', 'location', 'mean_temperature'):
                if col not in df.columns:
                    raise ValueError(f'Missing required column: {col}')

            # Sort by time then location for deterministic ordering
            periods = sorted(df['time_period'].unique(),
                             key=lambda p: week_to_monday(p))

            if len(periods) < 2:
                raise ValueError('Need at least 2 weeks of temperature data '
                                  '(1 context week + 1 prediction week).')

            # Build temperature matrix [400, n_days]
            # Expand each weekly Bundesland temperature to:
            #   - all 7 days of the week (repeat)
            #   - all Kreise in that Bundesland (same value)
            n_days = len(periods) * 7
            temp_kreise = np.zeros((400, n_days))

            for t_idx, period in enumerate(periods):
                week_df = df[df['time_period'] == period]
                # Fill BL temperatures into Kreise
                day_slice = slice(t_idx * 7, (t_idx + 1) * 7)
                bl_temps  = np.full(16, np.nan)
                for _, row in week_df.iterrows():
                    bl_idx = BL_UID_TO_IDX.get(row['location'])
                    if bl_idx is not None:
                        bl_temps[bl_idx] = row['mean_temperature']
                # Fill NaN BLs with Germany mean
                nan_mask = np.isnan(bl_temps)
                if nan_mask.any():
                    bl_temps[nan_mask] = np.nanmean(bl_temps)
                # Distribute BL temp to all Kreise in that BL
                kreis_idx = 0
                for bl_i, count in enumerate(BL_KREIS_COUNTS):
                    temp_kreise[kreis_idx:kreis_idx + count, day_slice] = bl_temps[bl_i]
                    kreis_idx += count

            # Normalise (model was trained on temp/15)
            temp_norm = temp_kreise / 15.0

            # Forward pass
            raw_out = forward(temp_norm, self.weights)    # [n_days-5, 400, 15, 2]
            n_pred  = raw_out.shape[0]

            # Build basic_death_case: scalar broadcast to shape
            bdc = np.full((n_pred, 400, 15, 2), self.baseline)
            daily_full  = apply_death_pred(raw_out, self.weights, bdc)
            daily_kreise = daily_full.sum(axis=(2, 3))     # [n_pred, 400]

            # Aggregate Kreise → Bundesland, daily → weekly
            # pred_days[i] corresponds to temp_kreise[:, KERNEL_DAYS-1+i]
            # i.e. first pred day aligns with day index KERNEL_DAYS-1 of periods[0]
            pred_periods_start = KERNEL_DAYS - 1   # 0-indexed in periods array (days)

            results = []
            for t_idx, period in enumerate(periods):
                # day offset within daily_kreise for this week
                day_start = t_idx * 7 - pred_periods_start
                day_end   = day_start + 7
                if day_start < 0 or day_end > n_pred:
                    continue   # skip the context week(s)

                week_kreise = daily_kreise[day_start:day_end].sum(axis=0)  # [400]

                kreis_idx = 0
                for bl_i, count in enumerate(BL_KREIS_COUNTS):
                    bl_uid = BL_IDX_TO_UID.get(bl_i)
                    if bl_uid is None:
                        kreis_idx += count; continue
                    weekly_deaths = float(week_kreise[kreis_idx:kreis_idx + count].sum())
                    # Confidence interval: ±20% (calibrated from training-period residuals)
                    results.append({
                        'time_period': period,
                        'location':    bl_uid,
                        'mean':        weekly_deaths,
                        'low':         weekly_deaths * 0.80,
                        'high':        weekly_deaths * 1.20,
                    })
                    kreis_idx += count

            return pd.DataFrame(results)

except ImportError:
    pass   # mlflow not installed; HeatMortalityModel class not available
