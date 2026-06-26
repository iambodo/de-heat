#!/usr/bin/env python3
"""
Download weekly mean 2m temperature for German Kreise from Open-Meteo
and import into DHIS2.

Uses ERA5 reanalysis (archive API) for historical data, then fills the
current and next ISO week with the Open-Meteo forecast API so recent
and upcoming temperatures are always present.

Checkpoints progress to data/era5_checkpoint.json so interruptions
can be resumed without re-fetching completed Kreise.

Usage:
    python3 scripts/download_temperature.py
    python3 scripts/download_temperature.py --start 2018-01-01
    python3 scripts/download_temperature.py --dry-run
    python3 scripts/download_temperature.py --reset   # clear checkpoint and restart
"""

import sys
import json
import time
import argparse
import requests
import urllib3
from pathlib import Path
from datetime import date, timedelta, datetime
from collections import defaultdict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DHIS2_BASE = "https://dhis2-127-0-0-1.nip.io"
AUTH = ("admin", "R3Zc8IawSBCHYu4Ve=k9NM-R5nw5w9SK")
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

KREISE_FILE = Path("data/geojson/kreise.geo.json")
CHECKPOINT_FILE = Path("data/era5_checkpoint.json")

DE_NAME = "Mean Temperature 2m"
DE_SHORT = "MeanTemp2m"
DE_CODE = "ERA5_TEMP_2M_MEAN"
DATASET_NAME = "ERA5 Temperature (Kreise Weekly)"

BATCH_SIZE = 1000
REQUEST_DELAY = 1.5   # seconds between Open-Meteo requests
RETRY_DELAYS = [60, 120, 300]  # backoff on 429: 1min, 2min, 5min

dhis2 = requests.Session()
dhis2.auth = AUTH
dhis2.verify = False
dhis2.headers.update({"Content-Type": "application/json"})

climate = requests.Session()


# ── DHIS2 helpers ─────────────────────────────────────────────────────────────

def api(method, path, **kwargs):
    url = f"{DHIS2_BASE}/api{path}"
    resp = dhis2.request(method, url, **kwargs)
    if not resp.ok:
        print(f"  DHIS2 ERROR {resp.status_code}: {resp.text[:400]}")
        resp.raise_for_status()
    return resp.json()


def ensure_data_element():
    r = api("GET", f"/dataElements?filter=code:eq:{DE_CODE}&fields=id,name")
    if r["dataElements"]:
        uid = r["dataElements"][0]["id"]
        print(f"  DataElement exists: {DE_NAME} ({uid})")
        return uid
    result = api("POST", "/dataElements", json={
        "name": DE_NAME,
        "shortName": DE_SHORT,
        "code": DE_CODE,
        "domainType": "AGGREGATE",
        "valueType": "NUMBER",
        "aggregationType": "AVERAGE",
        "zeroIsSignificant": True,
    })
    uid = result["response"]["uid"]
    print(f"  Created DataElement: {DE_NAME} ({uid})")
    return uid


def ensure_dataset(de_uid):
    r = api("GET", f"/dataSets?filter=name:eq:{requests.utils.quote(DATASET_NAME)}&fields=id")
    if r["dataSets"]:
        uid = r["dataSets"][0]["id"]
        print(f"  DataSet exists: {DATASET_NAME} ({uid})")
        return uid
    kreise = api("GET", "/organisationUnits?paging=false&fields=id&filter=level:eq:3")
    ou_ids = [{"id": ou["id"]} for ou in kreise["organisationUnits"]]
    result = api("POST", "/dataSets", json={
        "name": DATASET_NAME,
        "shortName": "ERA5 Temp Weekly",
        "periodType": "Weekly",
        "dataSetElements": [{"dataElement": {"id": de_uid}}],
        "organisationUnits": ou_ids,
        "openFuturePeriods": 2,
    })
    uid = result["response"]["uid"]
    print(f"  Created DataSet: {DATASET_NAME} ({uid})")
    return uid


def get_kreis_uid_map():
    """Returns {(kreis_name, bundesland_name): uid} for all level-3 org units."""
    kreise = api("GET", "/organisationUnits?paging=false&fields=id,name,parent[name]&filter=level:eq:3")
    mapping = {}
    for ou in kreise["organisationUnits"]:
        parent_name = ou.get("parent", {}).get("name", "")
        mapping[(ou["name"].strip(), parent_name.strip())] = ou["id"]
    return mapping


def post_batch(values, de_uid):
    payload = {
        "dataValues": [
            {"dataElement": de_uid, "period": v["period"],
             "orgUnit": v["orgUnit"], "value": str(v["value"])}
            for v in values
        ]
    }
    result = api("POST", "/dataValueSets?force=true&importStrategy=CREATE_AND_UPDATE", json=payload)
    ic = result.get("response", result).get("importCount", {})
    return ic.get("imported", 0), ic.get("updated", 0)


# ── Geometry helpers ───────────────────────────────────────────────────────────

def centroid(geometry):
    def ring_centroid(ring):
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    gtype = geometry["type"]
    coords = geometry["coordinates"]
    if gtype == "Polygon":
        return ring_centroid(coords[0])
    elif gtype == "MultiPolygon":
        best = max(coords, key=lambda poly: (
            (max(p[0] for p in poly[0]) - min(p[0] for p in poly[0])) *
            (max(p[1] for p in poly[0]) - min(p[1] for p in poly[0]))
        ))
        return ring_centroid(best[0])
    raise ValueError(f"Unsupported geometry: {gtype}")


# ── Open-Meteo fetch ───────────────────────────────────────────────────────────

def _fetch_daily_raw(lat, lon, start_date, end_date, use_forecast=False):
    """Fetch daily mean 2m temperature. Returns {date_str: celsius_or_None}."""
    url = FORECAST_URL if use_forecast else ARCHIVE_URL
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Berlin",
    }
    for attempt, wait in enumerate([0] + RETRY_DELAYS):
        if wait:
            print(f" [rate limited, waiting {wait}s]", end=" ", flush=True)
            time.sleep(wait)
        resp = climate.get(url, params=params, timeout=90)
        if resp.status_code == 429:
            if attempt < len(RETRY_DELAYS):
                continue
            resp.raise_for_status()
        resp.raise_for_status()
        break
    daily = resp.json().get("daily", {})
    return dict(zip(daily.get("time", []), daily.get("temperature_2m_mean", [])))


def fetch_weekly_temperature(lat, lon, start_date, end_date):
    """
    Fetch daily ERA5 archive + forecast and aggregate to ISO weeks.

    The archive API lags ~5-7 days. If end_date is within that lag window or
    in the future, the forecast API back-fills the gap so the current and
    next ISO week always have data. Returns {YYYYWnn: mean_celsius}.
    """
    archive_cutoff = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    archive_end = min(end_date, archive_cutoff)

    daily = _fetch_daily_raw(lat, lon, start_date, archive_end, use_forecast=False)

    # Extend into current / next week via forecast if requested end is recent
    if end_date > archive_cutoff:
        try:
            forecast = _fetch_daily_raw(lat, lon, archive_cutoff, end_date, use_forecast=True)
            for d, t in forecast.items():
                if d not in daily and t is not None:
                    daily[d] = t
        except Exception as e:
            print(f" [forecast fetch failed: {e}]", end=" ", flush=True)

    weekly = defaultdict(list)
    for d, t in daily.items():
        if t is None:
            continue
        dt = datetime.strptime(d, "%Y-%m-%d")
        iso_year, iso_week, _ = dt.isocalendar()
        period = f"{iso_year}W{iso_week:02d}"
        weekly[period].append(t)

    return {p: round(sum(vs) / len(vs), 2) for p, vs in weekly.items()}


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"completed": [], "failed": []}


def save_checkpoint(cp):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(cp, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2000-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch only, do not import")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint and restart")
    args = parser.parse_args()

    if args.end is None:
        # End of next ISO week (Sunday) so forecast covers current + next week
        today = date.today()
        days_until_next_sunday = (6 - today.weekday() + 7) % 7 + 7
        args.end = (today + timedelta(days=days_until_next_sunday)).strftime("%Y-%m-%d")

    print(f"=== ERA5 Weekly Temperature: {args.start} → {args.end} ===")
    print(f"(Archive up to ~7 days ago, forecast fills current + next ISO week)\n")

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint cleared.\n")

    if not KREISE_FILE.exists():
        print(f"ERROR: {KREISE_FILE} not found")
        sys.exit(1)

    print("Setting up DHIS2 metadata...")
    de_uid = ensure_data_element()
    if not args.dry_run:
        ensure_dataset(de_uid)

    print("Loading Kreise org unit map from DHIS2...")
    kreis_map = get_kreis_uid_map()
    print(f"  {len(kreis_map)} Kreise in DHIS2\n")

    features = json.loads(KREISE_FILE.read_text())["features"]
    print(f"Loading {len(features)} Kreise from GeoJSON...")

    cp = load_checkpoint()
    completed_set = set(cp["completed"])

    unmatched = []
    to_process = []
    for feat in features:
        props = feat["properties"]
        name = props["NAME_3"].strip()
        bl = props["NAME_1"].strip()
        key = (name, bl)
        ou_uid = kreis_map.get(key)
        if not ou_uid:
            unmatched.append(f"{name} ({bl})")
            continue
        to_process.append({
            "name": name,
            "bl": bl,
            "ou_uid": ou_uid,
            "lat": round(centroid(feat["geometry"])[1], 4),
            "lon": round(centroid(feat["geometry"])[0], 4),
        })

    if unmatched:
        print(f"  WARNING: {len(unmatched)} Kreise not matched in DHIS2:")
        for u in unmatched[:10]:
            print(f"    {u}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")

    remaining = [k for k in to_process if k["ou_uid"] not in completed_set]
    print(f"\n{len(to_process)} matched, {len(completed_set)} already done, {len(remaining)} to fetch\n")

    all_values = []
    total_imp = total_upd = 0

    for i, kreis in enumerate(remaining):
        label = f"[{i+1}/{len(remaining)}] {kreis['name']} ({kreis['bl']})"
        print(f"  {label}...", end=" ", flush=True)

        try:
            weekly = fetch_weekly_temperature(kreis["lat"], kreis["lon"], args.start, args.end)
            print(f"{len(weekly)} weeks")
        except Exception as e:
            print(f"FAILED: {e}")
            cp["failed"].append(kreis["ou_uid"])
            save_checkpoint(cp)
            continue

        for period, mean_temp in weekly.items():
            all_values.append({"period": period, "orgUnit": kreis["ou_uid"], "value": mean_temp})

        cp["completed"].append(kreis["ou_uid"])

        # Flush batch every 50 Kreise (or at end)
        if not args.dry_run and (len(all_values) >= BATCH_SIZE or i == len(remaining) - 1):
            imp = upd = 0
            for j in range(0, len(all_values), BATCH_SIZE):
                b_imp, b_upd = post_batch(all_values[j:j + BATCH_SIZE], de_uid)
                imp += b_imp
                upd += b_upd
            total_imp += imp
            total_upd += upd
            print(f"    → imported {imp}, updated {upd} (running total: {total_imp}+{total_upd})")
            all_values = []
            save_checkpoint(cp)

        time.sleep(REQUEST_DELAY)

    if args.dry_run and all_values:
        print(f"\nDry run — {len(all_values)} values would be imported.")
        print(f"Sample: {all_values[:3]}")
        return

    print(f"\n=== Done: {total_imp} imported, {total_upd} updated ===")
    if cp["failed"]:
        print(f"Failed Kreise: {len(cp['failed'])} — re-run to retry")


if __name__ == "__main__":
    main()
