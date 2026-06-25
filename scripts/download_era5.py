#!/usr/bin/env python3
"""
Download ERA5 daily 2m temperature for Germany and compute per-Kreis spatial averages.

Prerequisites:
    pip install cdsapi xarray netCDF4 geopandas rasterstats shapely requests

CDS API key setup (https://cds.climate.copernicus.eu/how-to-api):
    Create ~/.cdsapirc with:
        url: https://cds.climate.copernicus.eu/api
        key: <your-key>

Usage:
    python3 scripts/download_era5.py --year 2018 --kreise-geojson data/geojson/kreise.geo.json
    python3 scripts/download_era5.py --year 2018 --year 2019 --year 2020 ...

Output:
    data/era5/temperature_<year>.csv   (columns: rs, kreis_name, date, mean_temp_c)
"""

import argparse
import json
from pathlib import Path
import cdsapi
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterstats import zonal_stats


ERA5_VARIABLE = "2m_temperature"
GERMANY_BBOX = [47.2, 5.8, 55.1, 15.1]   # [S, W, N, E]
DATA_DIR = Path("data/era5")


def download_era5_year(year: int) -> Path:
    """Download ERA5 daily mean 2m temp for Germany for a full year."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"era5_t2m_{year}.nc"

    if out_path.exists():
        print(f"  Already downloaded: {out_path}")
        return out_path

    print(f"  Downloading ERA5 {year} from CDS...")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": ERA5_VARIABLE,
            "year": str(year),
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
            "area": GERMANY_BBOX,
            "format": "netcdf",
        },
        str(out_path),
    )
    print(f"  Saved: {out_path}")
    return out_path


def compute_daily_means(nc_path: Path) -> xr.Dataset:
    """Resample 3-hourly ERA5 to daily mean."""
    print(f"  Computing daily means from {nc_path.name}...")
    ds = xr.open_dataset(nc_path)
    # ERA5 land uses 't2m' variable name
    var_name = "t2m" if "t2m" in ds else list(ds.data_vars)[0]
    daily = ds[var_name].resample(time="1D").mean()
    # Convert Kelvin to Celsius
    daily = daily - 273.15
    daily.attrs["units"] = "°C"
    return daily


def spatial_average_per_kreis(daily_da: xr.DataArray, kreise_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute area-weighted mean temperature per Kreis per day."""
    print(f"  Computing spatial averages for {len(kreise_gdf)} Kreise...")
    records = []

    # ERA5 uses longitude 0-360 or -180-180 depending on version
    lons = daily_da.coords["longitude"].values if "longitude" in daily_da.coords else daily_da.coords["lon"].values
    lats = daily_da.coords["latitude"].values if "latitude" in daily_da.coords else daily_da.coords["lat"].values

    # Build affine transform for rasterio (assumes regular grid)
    from affine import Affine
    lon_res = abs(lons[1] - lons[0])
    lat_res = abs(lats[1] - lats[0])
    transform = Affine(lon_res, 0, lons.min() - lon_res/2,
                       0, -lat_res, lats.max() + lat_res/2)

    kreise_proj = kreise_gdf.to_crs("EPSG:4326")

    for t_idx, time_val in enumerate(daily_da.coords["time"].values):
        date_str = str(time_val)[:10]
        arr = daily_da.isel(time=t_idx).values
        # zonal_stats expects (row=lat, col=lon) with lat descending
        if lats[0] < lats[-1]:
            arr = arr[::-1, :]

        stats = zonal_stats(
            kreise_proj,
            arr,
            affine=transform,
            stats=["mean"],
            nodata=np.nan,
        )
        for i, (_, row) in enumerate(kreise_proj.iterrows()):
            mean_temp = stats[i].get("mean")
            if mean_temp is not None:
                records.append({
                    "rs": row.get("RS") or row.get("AGS") or row.get("rs"),
                    "kreis_name": row.get("GEN") or row.get("name") or row.get("NAME_2"),
                    "date": date_str,
                    "mean_temp_c": round(mean_temp, 2),
                })

        if (t_idx + 1) % 30 == 0:
            print(f"    Processed {t_idx + 1} days...")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, action="append", required=True, help="Year(s) to download")
    parser.add_argument("--kreise-geojson", default="data/geojson/kreise.geo.json",
                        help="Path to Kreise GeoJSON file")
    args = parser.parse_args()

    kreise_gdf = gpd.read_file(args.kreise_geojson)
    print(f"Loaded {len(kreise_gdf)} Kreise from {args.kreise_geojson}")

    for year in args.year:
        print(f"\n=== Processing {year} ===")
        nc_path = download_era5_year(year)
        daily_da = compute_daily_means(nc_path)
        df = spatial_average_per_kreis(daily_da, kreise_gdf)

        out_csv = DATA_DIR / f"temperature_{year}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Saved {len(df)} rows to {out_csv}")

    print("\n=== ERA5 download complete ===")
    print(f"Output CSVs in {DATA_DIR}/")
    print("Next: import temperature data into DHIS2 Climate App or use directly in model training.")


if __name__ == "__main__":
    main()
