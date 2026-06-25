# Plan: Germany Heat Mortality Model in Chap

## Context

The goal is to implement the ClimSocAna heat mortality model (from the 2024 Nature/Communications Medicine paper) as a Chap-compatible model for Germany, using DHIS2 with German district-level data. This involves four sequential phases: infrastructure setup, data acquisition, paper study, and model integration.

Network access through the proxy is restricted — external URLs must be fetched during execution when access is available, or fetched by the user manually.

---

## Phase 1: Prepare DHIS2 Instance

### 0 — Backup and reset DB - DONE, SKIP

**Manual steps (user runs):**
1. SSH into docker-deployment host
2. `docker exec 817be8d2ca2f pg_dump -U dhis2 dhis2 > dhis2_backup_$(date +%Y%m%d).sql`
3. Drop and recreate the dhis2 schema, or point to a blank DB
4. Keep the Chap app container connected (same DB credentials, same network)

### 1a — Import Germany districts as org units

**Source:** `https://github.com/isellsoap/deutschlandGeoJSON` — use the `4_kreise` folder, quality level `2_hoch` (good detail without huge file size).

**Steps I can help with:**
- Login to dhis2 at dhis2-127-0-0-1.nip.io/ with u admin and password R3Zc8IawSBCHYu4Ve=k9NM-R5nw5w9SK  
- Write a Python script to convert the GeoJSON to DHIS2 org unit import format (JSON)
- The script will: read each Kreise feature → extract name + geometry → POST to DHIS2 `/api/organisationUnits`
- Set parent org unit to the matching Bundesland (use `2_bundeslaender` GeoJSON for states first)
- Assign orgunits to admin user

**Output files to create:**
- `scripts/import_orgunits.py` — converts GeoJSON → DHIS2 org unit payload and imports via API
- Bundesländer first (16 states), then Kreise (~400 districts) with parent references

---

## Phase 1b: Find District-Level Mortality Data

**Source:** `https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Bevoelkerung/Sterbefaelle-Lebenserwartung/Publikationen/Downloads-Sterbefaelle/statistischer-bericht-sterbefaelle-tage-wochen-monate-aktuell-5126109.html`

**What to look for:**
- Table code `12613` (Gestorbene / Deaths) or similar
- Filter: Kreise (district level), years 2018–present, monthly or weekly granularity
- Download as CSV

**Steps I can help with:**
- Write a parsing script once the CSV is downloaded by the user
- Map Kreis names/codes to DHIS2 org unit UIDs

---

## Phase 1c: Create DHIS2 Dataset and Import Mortality Data

**Steps I can help with:**
- Write a script to create a DHIS2 DataSet + DataElement for "Deaths (all causes)" via the API
- Write an importer script that reads the regionalstatistik CSV and POSTs `dataValueSets` to DHIS2
- Handle period format (YYYYMM for monthly, YYYYWNN for weekly)

**Files to create:**
- `scripts/create_dataset.py` — creates DataSet + DataElement in DHIS2
- `scripts/import_mortality.py` — reads CSV, maps to org units + periods, imports

---

## Phase 1d: Add Climate App and Pull ERA-5 Temperature Data

*CLAUDE PERFORMS*
1. Install DHIS2 Climate App from the app hub - DONE MANUALLY/SKIP
2. Configure ERA-5 data source (requires Copernicus API key or similar) https://cds.climate.copernicus.eu/how-to-api
3. Pull daily mean temperature for German Kreise, 2018–present

**Steps I can help with:**
- Write a standalone ERA-5 downloader script using the `cdsapi` Python package if the Climate app is not sufficient
- The paper uses daily average temperature at district level (2018-2020 from Helmholtz Munich; 2021-2023 from DWD station interpolation)
- Script: `scripts/download_era5.py` — downloads ERA5 daily 2m temperature for Germany bbox, spatially averages per Kreis geometry

## Phase 1e: Import Population Data for 2025 and 2026
Unpack bundestat pop data from 004
Make a dhis2 dataset and import

---

## Phase 2: Paper Study

**Paper:** *High-resolution modeling and projection of heat-related mortality in Germany under climate change* (Communications Medicine, 2024)

**What I'll do when you're ready:**
- Walk through the paper section-by-section


**Format:** Write synopsis, how this model could be replicated with available data in dhis2.

---

## Phase 3: Code Analysis and Chap Documentation

**Source repo:** `https://github.com/ClimSocAna/Heat-Mortality` (branch: `Publication`)
**Action:** Clone locally first (requires internet access outside restricted proxy), then analyze.

**Expected repo structure (to verify after clone):**
```
data/           — input data (temperature, mortality CSV)
models/         — trained model artifacts
src/ or notebooks/
  train.py / train.ipynb   — model training (XGBoost/LGBM with Poisson loss)
  predict.py               — inference
  preprocessing.py         — data prep (daily temp aggregation per Kreis)
requirements.txt / environment.yml
```

**What I'll document:**
1. Data pipeline: raw temp + mortality CSV → feature matrix (lags, rolling means, population rates) → model input
2. Model architecture: XGBoost/LightGBM with Poisson loss, district + day-of-year features
3. Output format: per-district daily death predictions

**Output file:** `docs/code_analysis.md`

---

## Phase 4: Load Model into Chap (MLflow-based)

Chap uses **`chap-core`** (Python package) and expects external models packaged as **MLflow `pyfunc` models** with a specific input/output signature.

### 4a — MLflow model wrapper

Create `chap_model/model.py`:

```python
import mlflow.pyfunc
import pandas as pd
import xgboost as xgb   # or lightgbm, per paper

class HeatMortalityModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = xgb.Booster()
        self.model.load_model(context.artifacts["xgb_model"])

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        # model_input columns: time_period, location, mean_temperature, population
        # Returns DataFrame with columns: mean (+ low, high for credible interval)
        features = self._build_features(model_input)
        dmatrix = xgb.DMatrix(features)
        preds = self.model.predict(dmatrix)
        return pd.DataFrame({"mean": preds}, index=model_input.index)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Replicate preprocessing from Heat-Mortality repo:
        # - day of year, week of year
        # - rolling 7-day mean temperature
        # - population offset (log)
        # - district one-hot or label encoding
        ...
```

### 4b — Training script

Create `chap_model/train.py`:
- Accepts a CSV in Chap format: `time_period, location, mean_temperature, population, disease_cases`
- `disease_cases` = deaths (all-cause, heat-attributable portion estimated during training)
- Trains the XGBoost model with `objective="count:poisson"`
- Logs model artifact + metrics to MLflow
- Saves model using `mlflow.pyfunc.log_model()`

### 4c — MLflow model config files

**`chap_model/MLmodel`** (auto-generated by mlflow, but structure is):
```yaml
artifact_path: heat_mortality_model
flavors:
  python_function:
    python_model: model.pkl
    loader_module: mlflow.pyfunc
    python_version: "3.10"
signature:
  inputs: >
    [{"name": "time_period",       "type": "string"},
     {"name": "location",          "type": "string"},
     {"name": "mean_temperature",  "type": "double"},
     {"name": "population",        "type": "long"}]
  outputs: >
    [{"name": "mean",  "type": "double"},
     {"name": "low",   "type": "double"},
     {"name": "high",  "type": "double"}]
```

**`chap_model/conda.yaml`**:
```yaml
channels: [defaults, conda-forge]
dependencies:
  - python=3.10
  - pip:
    - mlflow>=2.8
    - xgboost>=2.0
    - pandas>=2.0
    - numpy>=1.24
    - scikit-learn>=1.3
```

**`chap_model/requirements.txt`** (pip fallback):
```
mlflow>=2.8
xgboost>=2.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
```

### 4d — Chap model template registration

Chap's Modeling App uses a **model template** YAML that points to the MLflow model. Create `chap_model/chap_template.yaml`:
```yaml
name: heat-mortality-germany
version: "1.0"
description: "District-level heat mortality model for Germany (ClimSocAna 2024)"
features:
  - name: mean_temperature
    source: climate_app    # pulled from ERA-5 via DHIS2 Climate App
type: mlflow_pyfunc
mlflow_model_uri: "models:/heat-mortality-germany/1"
target: disease_cases      # mapped to deaths data element in DHIS2
period_type: monthly       # aggregate daily model to monthly for Chap
```

### 4e — End-to-end training script

Create `chap_model/run_training.py`:
1. Pull training data from DHIS2 API (mortality + temperature)
2. Map DHIS2 org unit UIDs → AGS Kreis codes
3. Format into Chap-standard CSV
4. Call `mlflow.pyfunc.log_model()` and `mlflow.register_model()`
5. Outputs registered model URI for loading into Chap

### 4f — Files to create

```
chap_model/
  model.py            — PythonModel wrapper class
  train.py            — training entrypoint (Chap CSV → MLflow model)
  run_training.py     — full pipeline: DHIS2 pull → train → register
  MLmodel             — MLflow model manifest (auto-generated, but templated)
  conda.yaml          — conda environment spec
  requirements.txt    — pip requirements
  chap_template.yaml  — Chap model template for Modeling App import
  README.md           — how to train, register, and load in Chap
```

### 4g — Register in Chap (manual step by user)

After `run_training.py` succeeds:
1. Open Chap Modeling App → "Manage Model Templates"
2. Import `chap_template.yaml`
3. Trigger a prediction run for a target period
4. Verify predictions appear on the Chap map for German Kreise

---

## Execution Order

| Step | Who | Prerequisite |
|------|-----|--------------|
| 1 — Backup DB | User | Docker access |
| 1b — Import GeoJSON org units | Me (scripts) + User (run) | Blank DHIS2 instance |
| 1a — Download mortality data | User (from regionalstatistik.de) + Me (parse) | Org units in DHIS2 |
| 1c — Import mortality dataset | Me (scripts) + User (run) | Mortality CSV downloaded |
| 1d — ERA-5 temperature | Me (script) + User (run) | CDS API key |
| 2 — Paper quiz | Interactive (me + user) | None |
| 3 — Code analysis | Me | Heat-Mortality repo cloned locally |
| 4a–4f — MLflow model files | Me | Code analysis done |
| 4g — Register in Chap | User (UI) | MLflow model logged + Chap running |

---

## Verification

- Org units: DHIS2 map shows ~16 Bundesländer + ~400 Kreise with correct geometries
- Mortality data: `GET /api/dataValueSets?dataSet=...&period=202301&orgUnit=...` returns values
- ERA-5: temperature values visible in Climate app or in `scripts/download_era5.py` output CSV
- MLflow model: `mlflow models serve -m models:/heat-mortality-germany/1` starts without error
- Model predict: `mlflow models predict --model-uri ... --input-path test_input.csv` returns `mean/low/high` columns
- End-to-end: Chap Modeling App shows heat mortality predictions for German Kreise on a choropleth map
