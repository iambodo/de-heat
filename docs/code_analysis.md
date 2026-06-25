# Code Analysis: ClimSocAna/Heat-Mortality (Publication branch)

## Correction to paper synopsis

The paper does **not** use an R DLNM / quasi-Poisson GLM. It uses a **shallow neural network implemented in PyTorch / PyTorch Lightning**. The DLNM framing described in some summaries of the paper is not the actual code. What follows is derived directly from the repository.

---

## Repository structure

```
Heat-Mortality/
  train_lightning.py          — training entrypoint
  util.py                     — GeoDataFrame helper for Kreise shapes
  model/
    heat_mortality_model.py   — MortModel: Conv1d + FC network (HeatMortalityBase)
    temp_attention.py         — TempModel: attention-based station→Kreise interpolation
    loss_function.py          — multi-scale Poisson NLL loss
    metric.py                 — daily + weekly MSE
    utils.py                  — population/mortality data loading, land_grouping
    dataset.py                — DummyDataset (data is held as tensors, not a real Dataset)
    config.py                 — file paths
  data/
    population/
      population_ab2011.csv   — annual population by Kreise × 20 age groups × sex, 2011–2020
      death_cases/*.npy       — mortality arrays (daily Germany, weekly by age/sex/Bundesland)
    district_T/
      interpolated_DWD/t_pred_recent.pt  — district temperature 2021–2023 (TempModel output)
    geodata/georef-germany-kreis.geojson — Kreise boundaries
  state_dict/weekday_corr/    — trained model checkpoint (.ckpt)
  result.ipynb                — inference notebook
```

---

## Two models

### 1. TempModel (`model/temp_attention.py`)

An **attention-based spatial interpolator** that maps DWD weather station point observations to district-level temperature.

- Input: station coordinates + daily temperature measurements (shape: `[stations, days]`)
- Learned parameters: district centroids `geo_kreis` (trained to optimal positions), attention scale
- Mechanism: inverse-distance attention — `exp(-scale × distance(kreis, station))`, normalized to 1
- Output: daily temperature per district (`[400, days]`)
- Loss: MSE against Helmholtz Munich 1km gridded data (training target)
- Used to produce `data/district_T/interpolated_DWD/t_pred_recent.pt` for 2021–2023

This model is pre-run. Its output `.pt` file is the temperature input to MortModel for recent years.

---

### 2. MortModel (`model/heat_mortality_model.py`)

The main mortality model. Takes the full daily temperature time series for all 400 Kreise and predicts daily deaths per Kreis × age group × sex.

#### Architecture

Two variants exist; `HeatMortality_EXP` is the published one (`useExp=True`):

```
Input: temperature tensor [days × 400_kreise]  (normalized: /15, + random noise during training)
  │
  ├─ Conv1d(in=1, out=32, kernel=kernel_days×points_per_day, stride=points_per_day)  [multiply branch: exp transform]
  ├─ Conv1dPos(in=32, out=128, kernel=kernel_days, stride=1)                          [positive-weight conv]
  │
  └─ Conv1d(in=1, out=128, kernel=kernel_days, stride=1)                              [sum branch: exp]
       └─ exp()
  
  result = x_exp_sum + x_sum_exp
  LinearPos(128 → 30)   [positive weights → non-negative output]

Output: relative mortality deviation [days × 400_kreise × 15_age_groups × 2_sex]
```

**Key design choices:**

- `kernel_days=6` (default): the Conv1d kernel window spans 6 days — this implements the **lag effect** without an explicit lag model. The network learns which days in the preceding 6 contribute to today's mortality.
- **Positive-weight layers** (`Conv1dPos`, `LinearPos`): weights are stored as `w²`, ensuring non-negative contributions. This encodes the prior that temperature only increases mortality (no protective extreme heat effect in the network weights).
- **Exponential activation**: the `HeatMortality_EXP` variant uses `exp()` branches, giving the network capacity to model the exponential rise in mortality at extreme temperatures.
- **Baseline death rate**: the model predicts a *multiplier* on the expected baseline, not absolute deaths:

```python
prediction = (0.99 + outputs.clip(-0.99) + 0.01 * (outputs.clip(max=-0.99) + 0.99).exp()) \
             * weekday_correction \
             * basic_death_case
```

This ensures predictions are always positive (minimum ≈ 1% of baseline) and interpretable as excess mortality above baseline.

#### Baseline death rate (`baseline_mode='bottom_10'`)

Rather than an explicit seasonality model (spline or Fourier terms), the baseline is computed as a **rolling 10-lowest-week mean** over a 52-week window:

```python
# For each week t, take the 10 lowest weekly death counts in the past year:
base = deaths[t:t+52].topk(10, largest=False).values.sum() / 70  # 70 = 10 weeks × 7 days
```

This approximates the "cold-week" baseline mortality, removing seasonal confounding without model parameters.

#### Weekday correction

Day-of-week effects in mortality registration (fewer recorded on weekends) are handled by 6 learnable scalar parameters (one per day relative to Monday). Applied as a multiplier on every prediction.

---

## Data pipeline

### Temperature

| Period | Source | Format |
|--------|--------|--------|
| 2000–2020 | Helmholtz Munich 1km gridded daily (non-open) | CSV per Kreise |
| 2011–2020 | CERRA reanalysis (backup/fallback) | CSV per Kreise, 4 points/day |
| 2021–2023 | TempModel output from DWD stations | `.pt` tensor |

Loading (`weather_preprocessing` in `model/utils.py`):
1. Read Helmholtz Munich CSV → `[days × 400_kreise]`
2. Fill NaN with CERRA mean ± learned bias
3. Returns tensor in °C

### Mortality

Pre-processed `.npy` arrays in `data/population/death_cases/`:

| File | Shape | Contents |
|------|-------|----------|
| `de_daily.npy` | `[1, days, ...]` | Daily deaths, Germany total, all causes |
| `men_de_age_week.npy` | `[1, weeks, 15_age_groups]` | Weekly deaths, Germany, men |
| `women_de_age_week.npy` | same | Weekly deaths, Germany, women |
| `men_land_age_week.npy` | `[16, weeks, 15_age_groups]` | Weekly deaths, per Bundesland, men |
| `women_land_age_week.npy` | same | per Bundesland, women |

Training window: 2011–2020 (indices `11*366 : 21*366` for daily).

### Population

CSV: `population_ab2011.csv` — annual Kreise × 20 age groups × sex, 2011–2020.
Interpolated daily during training.

---

## Loss function (multi-scale supervision)

Because Kreise-level death labels are **not available** (only Bundesland and Germany totals), the model is supervised at coarser spatial scales:

```
Poisson NLL on daily deaths summed to Bundesland level
+
Poisson NLL on weekly deaths summed to Germany (by age × sex)
+
Poisson NLL on weekly deaths summed to Bundesland (by broad age group × sex)
```

The model predicts at Kreise level but is constrained by Bundesland/Germany observations. This is a form of **spatial disaggregation** — the network must learn a plausible district-level distribution consistent with observed state totals.

The 400→16 Bundesland aggregation uses hardcoded Kreise counts per Land in `_kreis_count`:
```python
_kreis_count = [0, 15, 1, 45, 2, 53, 26, 36, 44, 96, 6, 1, 18, 8, 13, 14, 22]
# Bundesländer order: SH, HH, NI, HB, NW, HE, RP, BW, BY, SL, BE, BB, MV, SN, ST, TH
```

---

## Training

- Framework: PyTorch Lightning, up to 20,000 epochs, early stopping (patience=50)
- Optimizer: Adam lr=0.001, ReduceLROnPlateau (patience=20)
- Precision: float64
- Hardware: 20 GB GPU (DGX)
- The dataset is a `DummyDataset` (n=10) — all data sits in GPU memory as pre-loaded tensors
- Noise augmentation: Gaussian noise ∼U(-0.5, 0.5) added to temperature during training

Trained checkpoint is available at `state_dict/weekday_corr/trained_state.ckpt`.

---

## CHAP compatibility analysis

### The fundamental mismatch

CHAP expects a model that takes **tabular rows** (one row = one location × one time period) and returns predictions for those rows. The Heat-Mortality model takes the **entire time series** as a single tensor and returns the entire predicted time series. These are architecturally different.

| Dimension | CHAP expectation | Heat-Mortality model |
|-----------|-----------------|---------------------|
| Input unit | One row: (location, period, temperature, population) | Full tensor: (all_days × 400_kreise) |
| Output unit | One row: predicted deaths | Full tensor: (days × 400_kreise × 15_age × 2_sex) |
| Spatial scope | Any subset of org units | Fixed 400 Kreise (hardcoded order) |
| Temporal scope | Arbitrary future period | Indexed offset into training tensor |

### What would be needed for CHAP

**Option A — Sliding window adapter (recommended)**

Restructure inference so that for a given (location, week) prediction, the model runs a forward pass on the last N days of temperature for that location. The Conv1d kernel naturally supports this as a sliding window.

Steps:
1. At inference, construct a temperature window `[kernel_days × 400_kreise]` from DHIS2 temperature values
2. Run a single Conv1d forward pass → get mortality multiplier for that day
3. Multiply by current population × baseline death rate → predicted deaths
4. Wrap in `mlflow.pyfunc.PythonModel` with CHAP's expected input schema

**Option B — Retrain as a tabular model**

Use the paper's insight (lag effect, baseline, weekday correction) but reimplement as a scikit-learn or statsmodels model:
- Features: temperature at week t, t-1, t-2 (3-week lag); week-of-year; population; Bundesland
- Target: weekly deaths
- Model: Poisson GLM or gradient boosting with Poisson loss
- Trains on our DHIS2 data (Bundesland × weekly, 2000–2026)
- Much simpler CHAP integration

**Option B is more practical** given:
- We have Bundesland mortality (not Kreise)
- We have weekly data (not daily)
- We cannot replicate the 20GB GPU training pipeline in CHAP
- The tabular model can be trained entirely on DHIS2-resident data

### Suggested CHAP model inputs/outputs

```
Inputs:
  time_period    : ISO week string (e.g. "2024W27")
  location       : DHIS2 org unit UID (Bundesland)
  mean_temperature: weekly mean 2m temperature (°C) from ERA5
  population     : annual population count

Outputs:
  mean           : predicted all-cause weekly deaths
  low            : 95% CI lower bound
  high           : 95% CI upper bound
```

### Heat-attributable mortality

Neither the raw model output nor a Poisson GLM directly gives "heat-attributable" deaths. To compute this:
1. Predict deaths at observed temperature: `deaths_obs`
2. Predict deaths at baseline temperature (MMT or population-specific threshold): `deaths_baseline`
3. Heat-attributable = `deaths_obs - deaths_baseline`

The threshold can be estimated from the fitted temperature-mortality curve (the temperature at minimum predicted mortality).

---

## Summary: what to reuse from this repo

| Component | Reusable? | How |
|-----------|-----------|-----|
| Trained checkpoint (`trained_state.ckpt`) | Partial | Can run inference for 2021–2023 with DWD data; not adaptable to new input format without surgery |
| `bottom_10` baseline concept | Yes | Replicate in Python: rolling window of 10 lowest weeks |
| Weekday correction | Yes | 6 learnable scalars; can include in statsmodels GLM as dummy variables |
| Lag structure (Conv1d kernel) | Yes | Replicate with lagged features (week t, t-1, t-2) in tabular model |
| `land_grouping` aggregation | Informative | Documents which Kreise belong to which Bundesland (hardcoded counts) |
| TempModel (station→district interpolation) | No | We use ERA5 directly; no station interpolation needed |
| `.npy` mortality arrays | No | We have DHIS2 data; these are non-open Destatis preprocessed files |
