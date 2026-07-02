from pathlib import Path

import numpy as np
import torch
import xarray as xr

from geoarches.dataloaders import era5
from geoarches.lightning_modules import load_module

WEATHERBENCH_ERA5_PATH = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
)
DATA_START = np.datetime64("2019-12-30T00:00:00")
FORECAST_START = np.datetime64("2019-12-31T00:00:00")
FORECAST_TARGET = np.datetime64("2020-01-01T00:00:00")


def download_archesweathergen_era5_data(data_dir: Path) -> Path:
    """Download the smallest ERA5 slice needed by ArchesWeatherGen for one forecast."""
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "era5_240_2019_12_30_to_2020_01_01.nc"

    if output_path.exists():
        return output_path

    variables = era5.surface_variables + era5.level_variables
    ds = xr.open_zarr(WEATHERBENCH_ERA5_PATH)
    ds = ds[variables].sel(time=slice(DATA_START, FORECAST_TARGET))
    ds = ds.sel(level=era5.pressure_levels)
    ds = ds.chunk({"time": -1, "level": -1, "latitude": 121, "longitude": 240})
    ds.to_netcdf(output_path)

    return output_path


def load_archesweathergen_model(device: str = "cpu"):
    gen_model, gen_config = load_module("archesweathergen", device=device)
    return gen_model.to(device), gen_config


def get_archesweathergen_real_data_batch(data_dir: Path, device: str = "cpu"):
    ds = era5.Era5Forecast(
        path=str(data_dir),
        domain="all",
        lead_time_hours=24,
        load_prev=True,
        norm_scheme="pangu",
    )
    return {k: v[None].to(device) for k, v in ds[0].items()}


def test_download_archesweathergen_era5_data(tmp_path):
    data_path = download_archesweathergen_era5_data(tmp_path / "era5_240" / "full")

    with xr.open_dataset(data_path) as ds:
        assert ds.time.to_numpy()[0].astype("datetime64[s]") == DATA_START
        assert ds.time.to_numpy()[-1].astype("datetime64[s]") == FORECAST_TARGET
        assert len(ds.time) == 9
        assert set(era5.surface_variables + era5.level_variables).issubset(ds.data_vars)
        assert ds.sizes["latitude"] == 121
        assert ds.sizes["longitude"] == 240
        assert list(ds.level.to_numpy()) == era5.pressure_levels


def test_load_archesweathergen_model_with_real_data_batch(tmp_path):
    device = "cpu"
    data_dir = tmp_path / "era5_240" / "full"
    download_archesweathergen_era5_data(data_dir)

    batch = get_archesweathergen_real_data_batch(data_dir, device=device)
    gen_model, gen_config = load_archesweathergen_model(device=device)

    assert gen_config.module.module.name == "archesweathergen-s-ft"
    assert gen_model.training is False
    assert next(gen_model.parameters()).device == torch.device(device)
    assert {"state", "next_state", "prev_state", "timestamp", "lead_time_hours"} <= set(batch)
    assert batch["timestamp"].item() == np.datetime64(FORECAST_START, "s").astype(int)
