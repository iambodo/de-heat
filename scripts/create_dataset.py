#!/usr/bin/env python3
"""
Create a DHIS2 DataSet + DataElement for district-level mortality data.

Usage:
    python3 scripts/create_dataset.py

Creates:
    DataElement: "Deaths (all causes)" — monthly, integer
    DataSet: "Germany District Mortality" — monthly, assigned to all Kreise
    CategoryOptionCombo: uses default
"""

import json
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DHIS2_BASE = "https://dhis2-127-0-0-1.nip.io"
AUTH = ("admin", "R3Zc8IawSBCHYu4Ve=k9NM-R5nw5w9SK")

session = requests.Session()
session.auth = AUTH
session.verify = False
session.headers.update({"Content-Type": "application/json"})


def api(method, path, **kwargs):
    url = f"{DHIS2_BASE}/api{path}"
    resp = session.request(method, url, **kwargs)
    if not resp.ok:
        print(f"  ERROR {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
    return resp.json()


def get_or_create(resource, filter_field, filter_value, payload, name_for_log):
    result = api("GET", f"/{resource}?filter={filter_field}:eq:{requests.utils.quote(str(filter_value))}&fields=id,{filter_field}")
    items = result.get(resource, [])
    if items:
        uid = items[0]["id"]
        print(f"  Already exists: {name_for_log} ({uid})")
        return uid
    result = api("POST", f"/{resource}", json=payload)
    uid = result["response"]["uid"]
    print(f"  Created: {name_for_log} ({uid})")
    return uid


def get_default_coc():
    result = api("GET", "/categoryOptionCombos?filter=name:eq:default&fields=id,name")
    items = result.get("categoryOptionCombos", [])
    if not items:
        raise RuntimeError("Could not find default CategoryOptionCombo")
    return items[0]["id"]


def get_all_kreise_uids():
    result = api("GET", "/organisationUnits?paging=false&fields=id&filter=level:eq:3")
    uids = [ou["id"] for ou in result.get("organisationUnits", [])]
    print(f"  Found {len(uids)} Kreise (level 3 org units)")
    return uids


def main():
    print("=== Create Germany District Mortality Dataset ===")

    # 1. DataElement: deaths (all causes)
    de_uid = get_or_create(
        "dataElements",
        "code",
        "DE_DEATHS_ALL",
        {
            "name": "Deaths (all causes)",
            "shortName": "Deaths",
            "code": "DE_DEATHS_ALL",
            "domainType": "AGGREGATE",
            "valueType": "INTEGER",
            "aggregationType": "SUM",
            "zeroIsSignificant": False,
        },
        "DataElement: Deaths (all causes)",
    )

    # 2. DataSet: monthly mortality
    kreise_uids = get_all_kreise_uids()
    if not kreise_uids:
        print("  WARNING: No Kreise found. Run import_orgunits.py first.")

    ds_uid = get_or_create(
        "dataSets",
        "code",
        "DS_DE_MORTALITY",
        {
            "name": "Germany District Mortality",
            "shortName": "DE Mortality",
            "code": "DS_DE_MORTALITY",
            "periodType": "Monthly",
            "dataSetElements": [{"dataElement": {"id": de_uid}}],
            "organisationUnits": [{"id": uid} for uid in kreise_uids],
            "openFuturePeriods": 1,
        },
        "DataSet: Germany District Mortality",
    )

    print(f"\n=== Done ===")
    print(f"DataElement UID : {de_uid}")
    print(f"DataSet UID     : {ds_uid}")
    print(f"\nSave these UIDs — needed by import_mortality.py")


if __name__ == "__main__":
    main()
