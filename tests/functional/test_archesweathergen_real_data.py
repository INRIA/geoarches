from pathlib import Path

import numpy as np
import xarray as xr

from geoarches.dataloaders import era5

WEATHERBENCH_ERA5_PATH = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
)
FORECAST_START = np.datetime64("2019-12-31T00:00:00")
FORECAST_TARGET = np.datetime64("2020-01-01T00:00:00")


def download_archesweathergen_era5_data(data_dir: Path) -> Path:
    """Download the smallest ERA5 slice needed to predict 2020-01-01 from 2019-12-31."""
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "era5_240_2019_12_31_to_2020_01_01.nc"

    if output_path.exists():
        return output_path

    variables = era5.surface_variables + era5.level_variables
    ds = xr.open_zarr(WEATHERBENCH_ERA5_PATH)
    ds = ds[variables].sel(time=slice(FORECAST_START, FORECAST_TARGET))
    ds = ds.sel(level=era5.pressure_levels)
    ds = ds.chunk({"time": -1, "level": -1, "latitude": 121, "longitude": 240})
    ds.to_netcdf(output_path)

    return output_path


def test_download_archesweathergen_era5_data(tmp_path):
    data_path = download_archesweathergen_era5_data(tmp_path / "era5_240" / "full")

    with xr.open_dataset(data_path) as ds:
        assert ds.time.to_numpy()[0].astype("datetime64[s]") == FORECAST_START
        assert ds.time.to_numpy()[-1].astype("datetime64[s]") == FORECAST_TARGET
        assert len(ds.time) == 5
        assert set(era5.surface_variables + era5.level_variables).issubset(ds.data_vars)
        assert ds.sizes["latitude"] == 121
        assert ds.sizes["longitude"] == 240
        assert list(ds.level.to_numpy()) == era5.pressure_levels
