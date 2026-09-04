from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from geoarches.dataloaders import era5
from geoarches.lightning_modules import load_module

WEATHERBENCH_ERA5_PATH = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
)
DATA_START = np.datetime64("2019-12-31T00:00:00")
FORECAST_START = np.datetime64("2020-01-01T00:00:00")
FORECAST_TARGET = np.datetime64("2020-01-03T00:00:00")
TARGET_PREDICTION_PATH = Path(__file__).with_name("aw_m4_target_prediction.pt")


def download_archesweather_era5_data(data_dir: Path) -> Path:
    """Download the smallest ERA5 slice needed by ArchesWeather M4 2-step forecast."""
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "era5_240_2019_12_31_to_2020_01_03.nc"

    if output_path.exists():
        return output_path

    variables = era5.surface_variables + era5.level_variables
    ds = xr.open_zarr(WEATHERBENCH_ERA5_PATH)
    ds = ds[variables].sel(time=slice(DATA_START, FORECAST_TARGET))
    ds = ds.sel(level=era5.pressure_levels)
    ds = ds.chunk({"time": -1, "level": -1, "latitude": 121, "longitude": 240})
    ds.to_netcdf(output_path)

    return output_path


@pytest.fixture(scope="module")
def archesweather_m4_data_path(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("archesweather_m4") / "era5_240" / "full"
    return download_archesweather_era5_data(data_dir)


@pytest.fixture(scope="module")
def archesweather_m4_batch_and_model(archesweather_m4_data_path):
    model, config = load_module(
        "archesweather-m-seed0",
        device="cpu",
        avg_with_modules=[
            "archesweather-m-seed1",
            "archesweather-m-skip-seed0",
            "archesweather-m-skip-seed1",
        ],
    )
    ds = era5.Era5Forecast(
        path=str(archesweather_m4_data_path.parent),
        domain="test",
        lead_time_hours=24,
        load_prev=True,
        multistep=2,
    )
    batch = {k: v[None].to("cpu") for k, v in ds[0].items()}
    return batch, model.to("cpu"), config


def test_download_archesweather_m4_era5_data(archesweather_m4_data_path):
    with xr.open_dataset(archesweather_m4_data_path) as ds:
        assert ds.time.to_numpy()[0].astype("datetime64[s]") == DATA_START
        assert ds.time.to_numpy()[-1].astype("datetime64[s]") == FORECAST_TARGET
        variables = era5.surface_variables + era5.level_variables
        assert set(variables).issubset(ds.data_vars)
        assert ds.sizes["latitude"] == 121
        assert ds.sizes["longitude"] == 240
        assert list(ds.level.to_numpy()) == era5.pressure_levels


def test_load_archesweather_m4_model_with_real_data_batch(archesweather_m4_batch_and_model):
    batch, model, config = archesweather_m4_batch_and_model

    assert model.training is False
    assert next(model.parameters()).device == torch.device("cpu")
    assert {"state", "next_state", "prev_state", "timestamp", "lead_time_hours"} <= set(batch)
    assert batch["timestamp"].item() == np.datetime64(FORECAST_START, "s").astype(int)


def test_archesweather_m4_multistep_prediction_against_real_outputs(
    archesweather_m4_batch_and_model,
):
    batch, model, _ = archesweather_m4_batch_and_model
    expected = torch.load(TARGET_PREDICTION_PATH, map_location="cpu", weights_only=False)

    with torch.no_grad():
        pred = model.forward_multistep(batch, iters=2)

    assert set(pred.keys()) == set(expected.keys())
    for key in expected.keys():
        assert pred[key].shape == expected[key].shape
        assert not torch.isnan(pred[key]).any()

    for key in expected.keys():
        torch.testing.assert_close(pred[key], expected[key], rtol=1e-4, atol=1e-4)
