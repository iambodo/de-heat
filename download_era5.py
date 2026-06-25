# download_era5.py — run in host/WSL2 user env
import cdsapi

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels-monthly-means",
    {
        "product_type": "monthly_averaged_reanalysis",
        "variable": "2m_temperature",
        "year": ["2023"],                      # adjust
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "area": [55.1, 5.8, 47.2, 15.1],       # Germany: N, W, S, E
        "format": "netcdf",
    },
    "era5_2m_temp_germany_2023.nc",
)
print("done")