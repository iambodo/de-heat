# CHAP Integration Setup

## What this does

CHAP's Modeling App fetches temperature and population from DHIS2, passes them to the
heat-mortality MLflow model, stores the predicted deaths back into DHIS2, and displays
them on the map.

## Architecture

```
DHIS2 (temperature + population data)
    ↕  fetched by chap-core
chap-core (Python backend)
    ↕  loads MLflow model
MLflow pyfunc model (chap_model/chap_pyfunc.py)
    ↕  runs NumPy forward pass
Pre-trained checkpoint (Heat-Mortality/state_dict/weekday_corr/trained_state.ckpt)
    ↓  writes predictions
DHIS2 (predicted deaths data element)
    ↓  displayed by
CHAP Modeling App (frontend already installed at /api/apps/dhis2-chapmodeling-app)
```

## Step 1 — Install Python dependencies

Requires `pypi.org` and `files.pythonhosted.org` on the proxy allowlist.

```bash
pip install mlflow pandas numpy requests
pip install chap-core          # DHIS2 CHAP backend
```

## Step 2 — Package and register the model

```bash
cd /home/brodo512/de-heat
python3 chap_model/package_model.py
```

This:
1. Logs the model to `./mlruns` (local MLflow tracking server)
2. Registers it under the name `heat-mortality-germany` version 1
3. Writes the CHAP model template to DHIS2 dataStore at `modeling/heat-mortality-germany-v1`
4. Runs a smoke test predicting Germany weekly deaths (expected: 15,000–22,000)

## Step 3 — Start chap-core

chap-core is a FastAPI backend that bridges CHAP's Modeling App to MLflow models.

```bash
chap-core serve \
  --dhis2-url https://dhis2-127-0-0-1.nip.io \
  --dhis2-username admin \
  --dhis2-password "R3Zc8IawSBCHYu4Ve=k9NM-R5nw5w9SK" \
  --mlflow-tracking-uri file:///home/brodo512/de-heat/mlruns \
  --port 8000
```

Or, if chap-core uses a config file:
```bash
chap-core serve --config chap_model/chap_core_config.yaml
```

## Step 4 — Configure the Modeling App

Open the CHAP Modeling App in DHIS2:
`https://dhis2-127-0-0-1.nip.io/api/apps/dhis2-chapmodeling-app/`

1. Go to **Settings** → set the chap-core backend URL to `http://localhost:8000`
2. Go to **Model Templates** → the `Heat Mortality Germany` template should appear
   (it was registered in Step 2 via the dataStore)
3. If it doesn't appear, use **Import Template** → upload `chap_model/chap_template.yaml`

## Step 5 — Run a prediction

In the Modeling App:
1. Select **Heat Mortality Germany (ClimSocAna)**
2. Choose org units: select Germany (all Bundesländer will be included)
3. Choose period: select the target week (e.g., current week)
4. Click **Generate Predictions**

CHAP will:
- Fetch ERA5 temperature from DHIS2 (data element `ERA5_TEMP_2M_MEAN`)
- Fetch population from DHIS2
- Pass both to the MLflow model
- Store predicted deaths as a new data value
- Display on the choropleth map

## Notes

- **Temperature data**: must be imported first (currently running via `scripts/download_temperature.py`).
  Temperature DataSet UID is `ERA5 Temperature (Kreise Weekly)`.
  However, the model uses Bundesland-level temperature. Ensure the temperature data element
  `Fnf55anfV8Z` has weekly values for all 16 Bundesländer.

- **Population**: the model does not strictly require population (it uses an internal baseline).
  If CHAP passes population, it is available in `model_input` but currently ignored.

- **Confidence interval**: ±20% is a placeholder. To recalibrate, compare predictions
  against observed deaths from 2021–2023 (available in DHIS2) and compute RMSE-based intervals.

- **Prediction frequency**: run weekly, shortly after the ERA5 data for the previous week
  becomes available (~5-day lag from Open-Meteo).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `chap-core` command not found | Not installed | `pip install chap-core` |
| Template not visible in UI | dataStore not populated | Re-run `package_model.py` or import `chap_template.yaml` manually |
| `mean_temperature` not found | Temperature not imported to DHIS2 | Wait for `download_temperature.py` to finish |
| MLflow model not found | `mlruns` path wrong | Use absolute path in `tracking_uri` |
| Predictions all zero | BL UIDs not matching | Check `BL_UID_TO_IDX` in `chap_pyfunc.py` vs actual DHIS2 UIDs |
