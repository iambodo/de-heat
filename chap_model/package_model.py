#!/usr/bin/env python3
"""
Package the heat-mortality model as an MLflow pyfunc model and register
the CHAP model template in DHIS2's dataStore.

Run once after installing mlflow:
    pip install mlflow pandas
    python3 chap_model/package_model.py

Then start the MLflow UI to see the logged run:
    mlflow ui --backend-store-uri ./mlruns
"""

import sys, json, os, requests, urllib3
os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE', 'true')
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from chap_model.chap_pyfunc import HeatMortalityModel, BL_UID_TO_IDX, BL_IDX_TO_UID

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT     = Path(__file__).parent.parent
CKPT     = ROOT / 'Heat-Mortality/state_dict/weekday_corr/trained_state.ckpt'
MORT_DIR = ROOT / 'Heat-Mortality/data/population/death_cases'

DHIS2_BASE = 'https://dhis2-127-0-0-1.nip.io'
AUTH       = ('admin', 'R3Zc8IawSBCHYu4Ve=k9NM-R5nw5w9SK')

TEMP_DE_UID     = 'Fnf55anfV8Z'   # Mean Temperature 2m  (ERA5_TEMP_2M_MEAN)
MORTALITY_DE_UID = 'm2rnAHcpz6U'  # Deaths (all causes)

MLFLOW_EXPERIMENT = 'heat-mortality-germany'
MODEL_NAME        = 'heat-mortality-germany'


def package_and_log():
    mlflow.set_tracking_uri(f'file://{ROOT}/mlruns')
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print('Packaging model with MLflow...')
    with mlflow.start_run(run_name='package-v1') as run:
        run_id = run.info.run_id

        mlflow.log_params({
            'kernel_days':       6,
            'model_arch':        'HeatMortality_EXP',
            'checkpoint':        'weekday_corr',
            'training_period':   '2011-2020',
            'training_data':     'Helmholtz-Munich 1km + CERRA reanalysis',
            'spatial_resolution': 'Kreise (400 districts)',
            'temporal_resolution': 'daily → weekly aggregated',
        })

        artifacts = {
            'checkpoint': str(CKPT),
            'mort_dir':   str(MORT_DIR),
        }

        mlflow.pyfunc.log_model(
            artifact_path='model',
            python_model=HeatMortalityModel(),
            artifacts=artifacts,
            pip_requirements=['numpy', 'pandas', 'requests'],
            input_example=pd.DataFrame([{
                'time_period':    '2024W25',
                'location':       'sTHbKLIUiJQ',
                'mean_temperature': 18.4,
                'population':     2922005,
            }]),
        )

        model_uri = f'runs:/{run_id}/model'
        mlflow.register_model(model_uri, MODEL_NAME)

        print(f'Model logged. Run ID: {run_id}')
        print(f'Model URI: {model_uri}')

    return run_id, model_uri


def register_chap_template(run_id, model_uri):
    """Write CHAP model template to DHIS2 dataStore."""
    print('\nRegistering CHAP model template in DHIS2 dataStore...')

    # Get Bundesland org unit UIDs → names mapping
    session = requests.Session()
    session.auth = AUTH
    session.verify = False

    r = session.get(f'{DHIS2_BASE}/api/organisationUnits',
                    params={'paging': 'false', 'fields': 'id,name', 'filter': 'level:eq:2'})
    r.raise_for_status()
    ou_map = {ou['id']: ou['name'] for ou in r.json()['organisationUnits']}

    # Check if dataStore namespace exists
    ns_r = session.get(f'{DHIS2_BASE}/api/dataStore/modeling')
    if ns_r.status_code == 404:
        create_r = session.post(f'{DHIS2_BASE}/api/dataStore/modeling/templates',
                                json=[])
        if not create_r.ok:
            # Try creating key directly
            session.post(f'{DHIS2_BASE}/api/dataStore/modeling/templates', json=[])

    template = {
        'id':          'heat-mortality-germany-v1',
        'name':        'Heat Mortality Germany (ClimSocAna)',
        'description': 'District-level heat-attributable mortality model for Germany. '
                       'Pre-trained shallow neural network (ClimSocAna 2024). '
                       'Bundesland weekly predictions from ERA5 temperature.',
        'version':     '1.0.0',
        'type':        'mlflow_pyfunc',
        'mlflow': {
            'tracking_uri': f'file://{ROOT}/mlruns',
            'model_uri':    model_uri,
            'run_id':       run_id,
        },
        'features': [
            {
                'name':           'mean_temperature',
                'dhis2_id':       TEMP_DE_UID,
                'description':    'Weekly mean 2m temperature from ERA5 (°C)',
                'period_type':    'Weekly',
            }
        ],
        'target': {
            'dhis2_id':    MORTALITY_DE_UID,
            'description': 'All-cause weekly deaths',
            'period_type': 'Weekly',
        },
        'org_unit_level': 2,
        'period_type':    'Weekly',
        'context_weeks':  2,
        'notes': (
            'Requires at least 2 weeks of temperature data as context window. '
            'Confidence interval is ±20% (placeholder; recalibrate with local data). '
            'Model was trained on 2011-2020 data; baseline uses 2000-2023 mortality. '
            'Temperature source: ERA5 via Open-Meteo (archive-api.open-meteo.com).'
        ),
    }

    # PUT to dataStore
    key_url = f'{DHIS2_BASE}/api/dataStore/modeling/heat-mortality-germany-v1'
    check = session.get(key_url)
    if check.ok:
        resp = session.put(key_url, json=template)
    else:
        resp = session.post(key_url, json=template)

    if resp.ok:
        print(f'  Template registered at dataStore key: modeling/heat-mortality-germany-v1')
    else:
        print(f'  ERROR: {resp.status_code} {resp.text[:200]}')

    # Also write as local YAML for manual import
    try:
        import yaml
        yaml_path = ROOT / 'chap_model/chap_template_registered.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False)
        print(f'  Template also saved to: {yaml_path}')
    except ImportError:
        json_path = ROOT / 'chap_model/chap_template_registered.json'
        json_path.write_text(json.dumps(template, indent=2))
        print(f'  Template also saved to: {json_path}')


def smoke_test(model_uri):
    """Quick end-to-end test with synthetic data."""
    print('\nRunning smoke test...')
    model = mlflow.pyfunc.load_model(model_uri)

    test_input = pd.DataFrame([
        {'time_period': '2024W24', 'location': uid, 'mean_temperature': 17.0 + i, 'population': 2000000}
        for i, uid in enumerate(list(BL_UID_TO_IDX.keys()))
    ] + [
        {'time_period': '2024W25', 'location': uid, 'mean_temperature': 19.0 + i, 'population': 2000000}
        for i, uid in enumerate(list(BL_UID_TO_IDX.keys()))
    ])

    preds = model.predict(test_input)
    print(f'  Smoke test passed. Predictions shape: {preds.shape}')
    germany_total = preds[preds['time_period'] == '2024W25']['mean'].sum()
    print(f'  Germany total (2024W25): {germany_total:,.0f} deaths/week')
    print(f'  Columns: {list(preds.columns)}')
    return preds


def main():
    for path in [CKPT, MORT_DIR]:
        if not Path(path).exists():
            print(f'ERROR: required path not found: {path}')
            print('Ensure Heat-Mortality repo is cloned at Heat-Mortality/')
            sys.exit(1)

    run_id, model_uri = package_and_log()
    register_chap_template(run_id, model_uri)
    smoke_test(model_uri)

    print(f'\n{"="*55}')
    print('Setup complete.')
    print(f'  MLflow UI:   mlflow ui --backend-store-uri {ROOT}/mlruns')
    print(f'  Model URI:   {model_uri}')
    print(f'  DHIS2 key:   dataStore/modeling/heat-mortality-germany-v1')
    print('='*55)


if __name__ == '__main__':
    main()
