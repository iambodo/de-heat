#!/usr/bin/env python3
"""
Import Germany Bundesländer and Kreise as DHIS2 org units.

GeoJSON source: https://github.com/isellsoap/deutschlandGeoJSON
  - 2_bundeslaender/2_hoch.geo.json  →  properties: {id, name, type}
  - 4_kreise/2_hoch.geo.json         →  properties: {NAME_1 (Bundesland), NAME_3 (Kreis), TYPE_3}

Usage:
    cd de-heat
    python3 scripts/import_orgunits.py
"""

import json
import sys
import time
import requests
import urllib3
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DHIS2_BASE = "https://dhis2-127-0-0-1.nip.io"
AUTH = ("admin", "R3Zc8IawSBCHYu4Ve=k9NM-R5nw5w9SK")
OPENING_DATE = "2018-01-01"

BUNDESLAENDER_FILE = Path("data/geojson/bundeslaender.geo.json")
KREISE_FILE = Path("data/geojson/kreise.geo.json")

session = requests.Session()
session.auth = AUTH
session.verify = False
session.headers.update({"Content-Type": "application/json"})


def api(method, path, **kwargs):
    url = f"{DHIS2_BASE}/api{path}"
    resp = session.request(method, url, **kwargs)
    if not resp.ok:
        print(f"  ERROR {resp.status_code}: {resp.text[:400]}")
        resp.raise_for_status()
    return resp.json()


def short_name(name, max_len=50):
    return name[:max_len]


def find_or_create_root():
    result = api("GET", "/organisationUnits?filter=name:eq:Germany&fields=id,name")
    units = result.get("organisationUnits", [])
    if units:
        uid = units[0]["id"]
        print(f"  Found existing root: Germany ({uid})")
        return uid
    result = api("POST", "/organisationUnits", json={
        "name": "Germany",
        "shortName": "Germany",
        "openingDate": OPENING_DATE,
        "code": "DE",
    })
    uid = result["response"]["uid"]
    print(f"  Created root: Germany ({uid})")
    return uid


def import_bundeslaender(root_uid):
    """Import 16 Bundesländer. Returns {bl_name: uid}."""
    print("\n--- Importing Bundesländer ---")
    features = json.loads(BUNDESLAENDER_FILE.read_text())["features"]
    print(f"  {len(features)} features found")

    name_to_uid = {}
    for feat in features:
        props = feat["properties"]
        name = props["name"]           # e.g. "Baden-Württemberg"
        code = props.get("id", "")     # e.g. "DE-BW"
        geom = feat.get("geometry")

        existing = api("GET", f"/organisationUnits?filter=name:eq:{requests.utils.quote(name)}&fields=id,name")
        if existing.get("organisationUnits"):
            uid = existing["organisationUnits"][0]["id"]
            print(f"  Exists: {name} ({uid})")
            name_to_uid[name] = uid
            continue

        payload = {
            "name": name,
            "shortName": short_name(name),
            "openingDate": OPENING_DATE,
            "parent": {"id": root_uid},
            "code": code,
        }
        if geom:
            payload["geometry"] = geom

        result = api("POST", "/organisationUnits", json=payload)
        uid = result["response"]["uid"]
        print(f"  Created: {name} (code={code}, uid={uid})")
        name_to_uid[name] = uid
        time.sleep(0.05)

    print(f"  {len(name_to_uid)} Bundesländer mapped")
    return name_to_uid


def import_kreise(bl_name_to_uid):
    """Import ~434 Kreise under their Bundesland (matched by NAME_1)."""
    print("\n--- Importing Kreise ---")
    features = json.loads(KREISE_FILE.read_text())["features"]
    print(f"  {len(features)} features found")

    created = skipped = errors = 0

    for feat in features:
        props = feat["properties"]
        name = props["NAME_3"]          # e.g. "Oldenburg"
        bl_name = props["NAME_1"]       # e.g. "Niedersachsen"
        kreis_type = props.get("TYPE_3", "")
        geom = feat.get("geometry")

        parent_uid = bl_name_to_uid.get(bl_name)
        if not parent_uid:
            print(f"  WARNING: No parent for '{name}' (Bundesland='{bl_name}') — skipping")
            errors += 1
            continue

        existing = api("GET", f"/organisationUnits?filter=name:eq:{requests.utils.quote(name)}&fields=id,name")
        if existing.get("organisationUnits"):
            skipped += 1
            continue

        payload = {
            "name": name,
            "shortName": short_name(name),
            "openingDate": OPENING_DATE,
            "parent": {"id": parent_uid},
        }
        if kreis_type:
            payload["description"] = kreis_type
        if geom:
            payload["geometry"] = geom

        try:
            result = api("POST", "/organisationUnits", json=payload)
            uid = result["response"]["uid"]
            print(f"  Created: {name} ({bl_name}, uid={uid})")
            created += 1
        except Exception as e:
            print(f"  FAILED: {name} — {e}")
            errors += 1
        time.sleep(0.05)

    print(f"\n  Done: {created} created, {skipped} already existed, {errors} errors")


def assign_orgunits_to_admin(root_uid):
    print("\n--- Assigning org units to admin user ---")
    users = api("GET", "/users?filter=username:eq:admin&fields=id,username,organisationUnits")
    admin = users["users"][0]
    admin_uid = admin["id"]

    current_ids = {ou["id"] for ou in admin.get("organisationUnits", [])}
    if root_uid in current_ids:
        print("  Admin already has Germany assigned")
        return

    api("PATCH", f"/users/{admin_uid}", json={"organisationUnits": [{"id": root_uid}]})
    api("PATCH", f"/users/{admin_uid}", json={"dataViewOrganisationUnits": [{"id": root_uid}]})
    print(f"  Assigned Germany ({root_uid}) to admin")


def main():
    print("=== DHIS2 Germany Org Unit Import ===")

    for f in [BUNDESLAENDER_FILE, KREISE_FILE]:
        if not f.exists():
            print(f"ERROR: Missing {f}")
            sys.exit(1)

    info = api("GET", "/system/info")
    print(f"Connected to DHIS2 {info.get('version')}")

    root_uid = find_or_create_root()
    bl_map = import_bundeslaender(root_uid)
    import_kreise(bl_map)
    assign_orgunits_to_admin(root_uid)

    print("\n=== Summary ===")
    for level, label in [(1, "Root"), (2, "Bundesländer"), (3, "Kreise")]:
        r = api("GET", f"/organisationUnits?paging=false&fields=id&filter=level:eq:{level}")
        print(f"  Level {level} ({label}): {len(r['organisationUnits'])}")


if __name__ == "__main__":
    main()
