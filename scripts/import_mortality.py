#!/usr/bin/env python3
"""
Parse Destatis/Regionalstatistik mortality data (xlsx or csv) and import into DHIS2.

Supports two data granularities:
  --level bundesland   Monthly deaths by Bundesland (from the Destatis xlsx already downloaded)
  --level kreis        Annual/monthly deaths by Kreis (from Regionalstatistik.de GENESIS)

Usage — Bundesland level (test pipeline with what we have):
    python3 scripts/import_mortality.py \\
        data/geojson/statistischer-bericht-sterbefaelle-tage-wochen-monate-aktuell-5126109.xlsx \\
        --data-element m2rnAHcpz6U --dataset Eo18mMLmTlH --level bundesland

Usage — Kreis level (after downloading from regionalstatistik.de):
    python3 scripts/import_mortality.py data/mortality/kreise_deaths.xlsx \\
        --data-element m2rnAHcpz6U --dataset Eo18mMLmTlH --level kreis

Getting Kreis-level data:
    1. Go to https://www.regionalstatistik.de/genesis/online/
    2. Search for table "12613" → "Gestorbene"
    3. Filter: Kreise-Ebene, years 2018+, monthly if available
    4. Download as xlsx or csv
"""

import sys
import argparse
import requests
import urllib3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_xlsx import load_workbook

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DHIS2_BASE = "https://dhis2-127-0-0-1.nip.io"
AUTH = ("admin", "R3Zc8IawSBCHYu4Ve=k9NM-R5nw5w9SK")
BATCH_SIZE = 500

# Bundesland name → DHIS2 org unit UID (built at runtime)
BL_NAME_MAP = {}
KREIS_NAME_MAP = {}
KREIS_CODE_MAP = {}

session = requests.Session()
session.auth = AUTH
session.verify = False
session.headers.update({"Content-Type": "application/json"})

MONTH_DE = {
    "Januar": "01", "Februar": "02", "März": "03", "April": "04",
    "Mai": "05", "Juni": "06", "Juli": "07", "August": "08",
    "September": "09", "Oktober": "10", "November": "11", "Dezember": "12",
}


def api(method, path, **kwargs):
    url = f"{DHIS2_BASE}/api{path}"
    resp = session.request(method, url, **kwargs)
    if not resp.ok:
        print(f"  ERROR {resp.status_code}: {resp.text[:400]}")
        resp.raise_for_status()
    return resp.json()


def build_orgunit_maps():
    print("Loading org unit maps from DHIS2...")
    r2 = api("GET", "/organisationUnits?paging=false&fields=id,name&filter=level:eq:2")
    for ou in r2["organisationUnits"]:
        BL_NAME_MAP[ou["name"].strip()] = ou["id"]

    r3 = api("GET", "/organisationUnits?paging=false&fields=id,name,code&filter=level:eq:3")
    for ou in r3["organisationUnits"]:
        KREIS_NAME_MAP[ou["name"].strip().lower()] = ou["id"]
        if ou.get("code"):
            KREIS_CODE_MAP[ou["code"].strip()] = ou["id"]

    print(f"  {len(BL_NAME_MAP)} Bundesländer, {len(KREIS_NAME_MAP)} Kreise")


def get_default_coc_uid():
    r = api("GET", "/categoryOptionCombos?filter=name:eq:default&fields=id")
    return r["categoryOptionCombos"][0]["id"]


def parse_bundesland_monthly(sheets):
    """
    Extract monthly deaths from sheets csv-12613-11 to csv-12613-13.
    These have: Gebiet=Bundesland, Geschlecht, Jahr, Alter, Monat, Sterbefaelle
    We use Geschlecht=Insgesamt, Alter=Insgesamt to get all-cause total.
    """
    print("Parsing Bundesland monthly data (sheets csv-12613-11/12/13)...")
    records = []
    for sheet_name in ["csv-12613-11", "csv-12613-12", "csv-12613-13"]:
        if sheet_name not in sheets:
            print(f"  Sheet not found: {sheet_name}")
            continue
        rows = sheets[sheet_name]
        if not rows:
            continue
        header = rows[0]
        # Expected: Statistik, Gebiet, Geschlecht, Jahr, Alter, Monat, Sterbefaelle, Qualitaetskennzeichen
        print(f"  {sheet_name}: {len(rows)-1} data rows, header={header}")
        for row in rows[1:]:
            if len(row) < 7:
                continue
            gebiet = row[1].strip()
            geschlecht = row[2].strip()
            jahr = row[3].strip()
            alter = row[4].strip()
            monat = row[5].strip()
            sterbefaelle = row[6].strip()

            if geschlecht != "Insgesamt" or alter != "Insgesamt":
                continue
            if monat == "Insgesamt" or not monat:
                continue

            month_num = MONTH_DE.get(monat)
            if not month_num:
                continue
            if not sterbefaelle or sterbefaelle in ("-", ""):
                continue
            try:
                deaths = int(sterbefaelle.replace(".", "").replace(",", ""))
            except ValueError:
                continue

            period = f"{jahr}{month_num}"
            records.append((gebiet, period, deaths))

    print(f"  Parsed {len(records)} Bundesland-monthly records")
    return records


def parse_kreis_data(sheets):
    """
    Parse Kreis-level data. Format depends on the downloaded GENESIS table.
    Tries to auto-detect columns.
    Expected columns: Kreis name or AGS code, Jahr, Monat (or just Jahr), deaths count.
    """
    print("Parsing Kreis-level data...")
    # Try common sheet names from GENESIS downloads
    for sheet_name in sheets:
        rows = sheets[sheet_name]
        if not rows or len(rows) < 5:
            continue
        header = [str(h).lower() for h in rows[0]]
        if any("kreis" in h or "ags" in h or "rs" in h for h in header):
            print(f"  Using sheet: {sheet_name} ({len(rows)} rows)")
            print(f"  Header: {rows[0]}")
            # Return raw rows for manual column mapping
            return rows
    print("  No Kreis-level sheet auto-detected. Printing all sheet names:")
    for name, rows in sheets.items():
        print(f"    {name}: {len(rows)} rows")
    return []


def post_batch(values, dataset_uid, de_uid, default_coc_uid):
    payload = {
        "dataValues": [
            {
                "dataElement": de_uid,
                "period": v["period"],
                "orgUnit": v["orgUnit"],
                "value": str(v["value"]),
            }
            for v in values
        ],
    }
    # force=true bypasses dataset input-period and org-unit assignment checks
    result = api("POST", "/dataValueSets?force=true&importStrategy=CREATE_AND_UPDATE", json=payload)
    ic = result.get("response", result).get("importCount", {})
    return ic.get("imported", 0), ic.get("updated", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx_file", help="Path to mortality xlsx file")
    parser.add_argument("--data-element", required=True, help="DataElement UID")
    parser.add_argument("--dataset", required=True, help="DataSet UID")
    parser.add_argument("--level", choices=["bundesland", "kreis"], default="bundesland",
                        help="Data granularity to import")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, do not import")
    args = parser.parse_args()

    path = Path(args.xlsx_file)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    print(f"Loading {path.name}...")
    sheets = load_workbook(path)
    print(f"Loaded {len(sheets)} sheets: {list(sheets.keys())[:6]}...")

    build_orgunit_maps()
    default_coc_uid = get_default_coc_uid()

    if args.level == "bundesland":
        records = parse_bundesland_monthly(sheets)
        values = []
        unmatched = set()
        for gebiet, period, deaths in records:
            ou_uid = BL_NAME_MAP.get(gebiet)
            if not ou_uid:
                unmatched.add(gebiet)
                continue
            values.append({"period": period, "orgUnit": ou_uid, "value": deaths})

        print(f"\nMatched: {len(values)}, Unmatched: {len(unmatched)}")
        if unmatched:
            print(f"Unmatched Gebiete: {sorted(unmatched)}")

    elif args.level == "kreis":
        rows = parse_kreis_data(sheets)
        if not rows:
            print("No Kreis data found. Download the correct table from regionalstatistik.de.")
            sys.exit(1)
        # Auto-detection failed — print headers for user to specify columns manually
        print("\nKreis data found but column mapping needs manual review.")
        print("Re-run with --level kreis after verifying the sheet structure.")
        sys.exit(0)

    if args.dry_run:
        print(f"\nDry run complete. Would import {len(values)} values.")
        if values:
            print(f"Sample: {values[:3]}")
        return

    print(f"\nImporting {len(values)} values in batches of {BATCH_SIZE}...")
    total_imported = total_updated = 0
    for i in range(0, len(values), BATCH_SIZE):
        batch = values[i:i + BATCH_SIZE]
        imp, upd = post_batch(batch, args.dataset, args.data_element, default_coc_uid)
        total_imported += imp
        total_updated += upd
        print(f"  Batch {i//BATCH_SIZE + 1}: {imp} imported, {upd} updated")

    print(f"\n=== Done: {total_imported} imported, {total_updated} updated ===")


if __name__ == "__main__":
    main()
