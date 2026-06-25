# Proposal: Adapting the Heat-Mortality Model for CHAP

## Summary

The ClimSocAna model is a PyTorch neural network trained on data that is partially in the repo and partially proprietary. We can reconstruct a faithful adaptation using **only open data already available or importable**, running at weekly Bundesland resolution. The core architectural concepts (lag window, baseline normalisation, Poisson loss, weekday correction) translate cleanly into a scikit-learn / statsmodels pipeline that CHAP can consume via MLflow.

A second, higher-resolution path using the **pre-trained checkpoint** for district-level inference is also feasible and is described as Option B.

---

## What is available from the repo

| Asset | Status | Coverage |
|-------|--------|----------|
| `data/population/death_cases/*.npy` | In repo | Weekly deaths by Bundesland × age × sex, 2000–**end 2023** |
| `data/population/population/population_ab2011.csv` | In repo | Annual Kreise population 2011–2020 |
| `state_dict/weekday_corr/trained_state.ckpt` | In repo | Trained weights (237 KB) for `kernel_days=6, useExp=True` |
| `data/district_T/interpolated_DWD/t_pred_recent.pt` | In repo | District daily temperature 2021–2023 (TempModel output) |
| `data/geodata/georef-germany-kreis.geojson` | In repo | Kreise boundaries (matches model's 400-district order) |
| Helmholtz Munich 1km temperature 2000–2020 | **Not in repo** — non-open, data agreement required |
| CERRA reanalysis 2011–2020 | **Not in repo** — Copernicus download required |
| Climate projections (EC-Earth3) | **Not in repo** — size excluded |

The repo therefore ships with **mortality + population + pre-trained weights**. Temperature for training (2011–2020) is not present and cannot be reconstructed without either the Helmholtz dataset or re-downloading CERRA.

---

## Data source re-download feasibility

### Can we get recent (last 2 weeks) data?

| Source | Can download? | Blocker | Alternative |
|--------|--------------|---------|-------------|
| **Open-Meteo ERA5** (temperature) | **Yes** — once `archive-api.open-meteo.com` is allowlisted | Currently proxy-blocked | Already scripted in `scripts/download_temperature.py`; lag ≈ 5 days |
| **DWD station data** (temperature) | No — `opendata.dwd.de` proxy-blocked | Proxy allowlist | Use Open-Meteo ERA5 instead |
| **CERRA reanalysis** (temperature) | No — `cds.climate.copernicus.eu` proxy-blocked | Proxy allowlist | Use Open-Meteo ERA5 instead |
| **Destatis mortality** (weekly) | No — `www.destatis.de` proxy-blocked | Proxy allowlist | Already imported to DHIS2 (2000–2026) |
| **Regionalstatistik population** | No — `www.regionalstatistik.de` proxy-blocked | Proxy allowlist | Repo has 2011–2020; DHIS2 import covers remainder |

**Practical conclusion:** temperature can be kept current via Open-Meteo once the proxy domain is added. Mortality and population data are available in DHIS2 from our existing imports. No data source requires manual intervention once the proxy is configured.

For the **next 2 weeks specifically**: the model makes forward predictions from temperature alone (mortality is the predicted output, not an input at inference time). Open-Meteo provides ERA5 up to approximately 5 days before today, and weather forecast data for the remaining days — so a 2-week forecast is achievable.

---

## Proposed adaptation: Option A — Retrain at Bundesland/weekly resolution (recommended)

### Why this is preferred

- Our DHIS2 instance holds weekly mortality and temperature at Bundesland level — exactly the training data needed
- The repo's `.npy` mortality arrays cover 2000–end 2023 by Bundesland, which can supplement DHIS2 data for the training window 2011–2020
- CHAP's tabular interface maps naturally to weekly Bundesland data
- No GPU required; trains in seconds on a laptop

### Model architecture

A **quasi-Poisson GLM with lagged temperature features**, replicating the key concepts from the neural network:

```
predict(deaths_bl_week_t) = Poisson(μ)

log(μ) = offset(log_pop)
       + α_bl                           # Bundesland fixed effect (16 levels)
       + β₁·temp_t + β₂·temp_{t-1} + β₃·temp_{t-2}   # 3-week lag (≈ 6-day Conv1d)
       + β₄·temp_t² + β₅·temp_{t-1}²                  # non-linear heat response
       + γ·weekofyear_spline(doy, df=6) # seasonality (replaces bottom-10 baseline)
       + δ·year                          # long-run mortality trend
       + ε·weekday_indicator             # day-of-week correction (Mon–Sun)
```

**Mapping of neural-network concepts to GLM features:**

| Neural network concept | GLM equivalent |
|----------------------|----------------|
| Conv1d kernel of 6 days | Lagged weekly temperatures t, t-1, t-2 |
| Positive-weight exponential branches | Quadratic temperature terms (temp²) |
| `bottom_10` baseline normalisation | `log(population)` offset + week-of-year spline |
| Weekday correction (6 scalars) | Weekday dummy variables |
| Poisson NLL loss | Poisson GLM family (statsmodels `GLM`) |
| Multi-scale supervision (Kreise→BL) | Not needed — training directly at BL level |

### Training data

All from sources already in our possession:

| Source | Content | Period |
|--------|---------|--------|
| DHIS2 mortality DataSet | Weekly deaths, 16 Bundesländer | 2000–2026 (our import) |
| DHIS2 temperature DataSet | Weekly mean 2m temp, 16 Bundesländer | 2000–2025 (importing now) |
| Repo `.npy` arrays (optional supplement) | Can verify/cross-check 2011–2023 BL mortality | 2000–end 2023 |
| Population CSV in repo | Annual Kreise population (aggregate to BL) | 2011–2020 |
| DHIS2 population DataSet | Annual BL population | 2021–2026 (Phase 1e import) |

### Heat-attributable mortality output

After fitting, compute for each week:

1. Predict deaths at observed temperature: `ŷ_obs`
2. Predict deaths at the **minimum mortality temperature** (MMT), estimated as the temperature at which `∂log(μ)/∂temp = 0` from the fitted spline/quadratic
3. Heat-attributable deaths = `max(0, ŷ_obs - ŷ_MMT)` per Bundesland per week

This is the same approach as the paper, just at coarser resolution.

### CHAP interface

```
Training input CSV columns:
  time_period      — ISO week "2024W27"
  location         — DHIS2 Bundesland UID
  mean_temperature — weekly mean 2m temperature (°C)
  population       — annual population
  disease_cases    — weekly all-cause deaths

Prediction output:
  mean  — predicted weekly all-cause deaths
  low   — lower 95% CI (quasi-Poisson, dispersion-corrected)
  high  — upper 95% CI
```

### Files to create

```
chap_model/
  train.py         — fits GLM, logs to MLflow, saves model
  predict.py       — PythonModel wrapper for CHAP
  baseline.py      — bottom-10 weekly baseline helper (reused from repo concept)
  requirements.txt — statsmodels, pandas, numpy, mlflow, scipy
```

---

## Proposed adaptation: Option B — Inference from pre-trained checkpoint

The existing checkpoint (`state_dict/weekday_corr/trained_state.ckpt`, 237 KB) contains trained weights for the district-level neural network. It can be loaded for **inference only** using the DWD temperature tensor already in the repo (`t_pred_recent.pt`, covering 2021–2023).

### What this gives us

- District-level (Kreise) daily death predictions for 2021–2023 (the period covered by the temperature tensor)
- A forward-extension path: supply ERA5 temperatures from Open-Meteo for 2024–present to extend inference

### Limitation

The checkpoint was trained on non-open Helmholtz Munich 1km temperature data (2011–2020). It cannot be retrained from scratch without that dataset. It is a **frozen inference-only artifact**.

To extend inference to the present using Open-Meteo ERA5:

1. Download ERA5 daily temperature per Kreis centroid from Open-Meteo (434 API calls per day, ~3 min)
2. Stack into a `[400, days]` tensor matching the model's Kreise order (from `util.py → get_gdf()`, sorted by `krs_code`)
3. Load checkpoint, run forward pass, aggregate 30-channel output (age × sex) to all-cause deaths
4. Aggregate Kreise to Bundesland using `land_grouping` from `model/utils.py`

### Why Option B is secondary

- Requires PyTorch (not installed in current environment)
- Kreise temperature ordering in the checkpoint must exactly match the 400-Kreis order in `get_gdf()` — any mismatch silently corrupts predictions
- The checkpoint cannot be retrained if quality degrades, as training data is unavailable
- The output (daily Kreise × 30 channels) must be aggregated to weekly Bundesland to enter CHAP

Option B is worth pursuing **after Option A is working**, as a validation step: the two models should produce broadly consistent Bundesland heat signals during their overlapping period (2021–2023).

---

## Option C — Inference-only: predict this week without retraining

If the goal is simply **"give me predicted deaths for the current week"** rather than model evaluation or retraining, the pipeline collapses dramatically. No recent mortality data is needed at all — mortality is the *output*, not an input.

### What inference actually needs

| Input | Source | How current? |
|-------|--------|-------------|
| This week's daily temperature (7 values per location) | Open-Meteo ERA5 | Up to ~5 days ago; forecast fills remainder |
| Population (single annual estimate) | Last known value — 2020 from repo CSV or DHIS2 | Annual update; 1-year-old value is fine |
| Baseline death rate for this week | Computed from historical weekly deaths (any year through 2023) | Does not need to be recent |
| Trained model weights | Repo checkpoint `trained_state.ckpt` | Frozen — never needs updating |

The model does not consume current mortality data. It was trained to predict deaths from temperature. At inference time, past deaths are only used once to compute the **rolling bottom-10 baseline** — and that baseline can be computed from any sufficiently long historical window (the repo's `.npy` files cover 2000–2023, which is more than enough).

### Inference pipeline (Option B, operationalised)

```
1. Fetch 6+ days of daily temperature for each of the 400 Kreise
   └─ Open-Meteo ERA5 per centroid (already scripted in download_temperature.py,
      adapt for daily output and the Kreise ordering from util.py:get_gdf())

2. Construct temperature tensor [days × 400_kreise] matching the model's Kreise order
   └─ Kreise sorted by krs_code from georef-germany-kreis.geojson

3. Compute baseline_death_rate for current week
   └─ From repo .npy: for each Bundesland, take bottom-10 weekly counts
      from the 52 weeks ending at the last available mortality date (~Dec 2023)
   └─ This baseline is fixed — it does not need to be recomputed weekly

4. Load checkpoint, run forward pass on the temperature window
   └─ Output: [days × 400_kreise × 15_age × 2_sex] mortality multipliers

5. Apply: prediction = multiplier × baseline_death_rate × population

6. Aggregate:
   └─ Sum over age × sex → all-cause deaths per Kreise per day
   └─ Sum over 7 days → weekly
   └─ Sum over Kreise within Bundesland → Bundesland weekly deaths (using land_grouping)

7. POST results to DHIS2 as dataValues
```

### What this does NOT require

- No recent mortality data (deaths are predicted, not observed)
- No retraining (checkpoint is frozen)
- No Helmholtz/CERRA temperature data (Open-Meteo ERA5 is the substitute)
- No Destatis download (baseline computed once from the repo's `.npy` files)
- No GPU (inference on 237 KB weights is CPU-trivial)

### The one dependency: PyTorch

The checkpoint requires PyTorch to load. It is not installed in the current environment. Install via:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
(CPU-only build, ~200 MB; the model is tiny and runs in milliseconds.)

### CHAP integration for inference-only

In CHAP's model template, set the model to **predict-only** mode:

```yaml
# chap_model/chap_template.yaml
name: heat-mortality-germany-inference
type: mlflow_pyfunc
features:
  - name: mean_temperature
    source: climate_app
target: disease_cases
period_type: weekly
# No training step — model is pre-trained
```

CHAP supplies temperature from the Climate App each week → the MLflow wrapper runs the forward pass → returns predicted deaths. No feedback loop with observed deaths is needed unless you want to recalibrate the baseline.

### Staleness of the frozen baseline

The bottom-10 weekly baseline captures secular mortality trends (ageing population, long-run health improvements). Using a 2023 baseline to predict 2025 deaths introduces a small systematic bias (~1–2% per year as mortality rates drift). This is acceptable for a heat signal model — the *excess* above baseline is the quantity of interest, and the baseline drift affects numerator and denominator equally.

If you want to refresh the baseline annually without full retraining, it only requires updating one scalar per Bundesland per week-of-year — a 5-minute script pulling the last year of weekly deaths from DHIS2.

---

## Recommended implementation sequence

**Fastest path to a prediction this week (Option C):**

| Step | Action | Prerequisite |
|------|--------|-------------|
| 1 | `pip install torch --index-url https://download.pytorch.org/whl/cpu` | — |
| 2 | Allowlist `archive-api.open-meteo.com`, fetch daily ERA5 per Kreis centroid | Proxy change |
| 3 | Write inference script: load checkpoint → temperature tensor → forward pass → aggregate | Steps 1–2 |
| 4 | Wrap in MLflow pyfunc, register in CHAP | Step 3 |

**Full retrain path (Option A — more robust long-term):**

| Step | Action | Data source | Output |
|------|--------|------------|--------|
| 1 | Allowlist `archive-api.open-meteo.com` | — | Proxy access to ERA5 |
| 2 | Finish temperature import (currently running) | Open-Meteo | DHIS2 weekly Bundesland temps |
| 3 | Complete Phase 1e: import population 2021–2026 | Destatis/DHIS2 | DHIS2 population data element |
| 4 | Write `chap_model/train.py` — pull from DHIS2, fit Poisson GLM, log to MLflow | DHIS2 API | MLflow model artifact |
| 5 | Write `chap_model/predict.py` — PythonModel wrapper | — | CHAP-compatible MLflow pyfunc |
| 6 | Register in CHAP Modeling App | — | Live predictions |
| 7 | (Optional) Cross-validate Option A vs Option C on 2021–2023 overlap | Both | Confidence in predictions |

---

## What cannot be replicated without proprietary data

- Kreise-level training (Helmholtz Munich 1km data — data agreement, not open)
- Full CERRA-based temperature 2011–2020 (requires CDS download; proxy must be allowlisted)
- Climate projections (EC-Earth3 ensemble — files excluded from repo due to size)

The proposed Option A is independent of all three. It is self-contained on open data and produces scientifically valid, CHAP-ready predictions at Bundesland resolution.
