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
FORECAST_TARGET = np.datetime64("2020-01-02T00:00:00")
TARGET_PREDICTION_PATH = Path(__file__).with_name("target_prediction.pt")
TARGET_PREDICTION_SEED = 0
TARGET_PREDICTION_NUM_STEPS = 25
TARGET_PREDICTION_SCALE_INPUT_NOISE = 1.05


def download_archesweathergen_era5_data(data_dir: Path) -> Path:
    """Download the smallest ERA5 slice needed by ArchesWeatherGen for one forecast."""
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "era5_240_2019_12_31_to_2020_01_02.nc"

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
def archesweathergen_data_path(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("archesweathergen") / "era5_240" / "full"
    return download_archesweathergen_era5_data(data_dir)


@pytest.fixture(scope="module")
def archesweathergen_batch_and_model(archesweathergen_data_path):
    ds = era5.Era5Forecast(
        path=str(archesweathergen_data_path.parent),
        domain="test",
        lead_time_hours=24,
        load_prev=True,
        norm_scheme="pangu",
    )
    batch = {k: v[None].to("cpu") for k, v in ds[0].items()}
    gen_model, gen_config = load_module("archesweathergen", device="cpu")
    return batch, gen_model.to("cpu"), gen_config


def test_download_archesweathergen_era5_data(archesweathergen_data_path):
    with xr.open_dataset(archesweathergen_data_path) as ds:
        assert ds.time.to_numpy()[0].astype("datetime64[s]") == DATA_START
        assert ds.time.to_numpy()[-1].astype("datetime64[s]") == FORECAST_TARGET
        assert len(ds.time) == 9
        assert set(era5.surface_variables + era5.level_variables).issubset(ds.data_vars)
        assert ds.sizes["latitude"] == 121
        assert ds.sizes["longitude"] == 240
        assert list(ds.level.to_numpy()) == era5.pressure_levels


def test_load_archesweathergen_model_with_real_data_batch(archesweathergen_batch_and_model):
    batch, gen_model, gen_config = archesweathergen_batch_and_model

    assert gen_config.module.module.name == "archesweathergen-s-ft"
    assert gen_model.training is False
    assert next(gen_model.parameters()).device == torch.device("cpu")
    assert {"state", "next_state", "prev_state", "timestamp", "lead_time_hours"} <= set(batch)
    assert batch["timestamp"].item() == np.datetime64(FORECAST_START, "s").astype(int)


def test_archesweathergen_prediction_matches_target(archesweathergen_batch_and_model):
    batch, gen_model, _ = archesweathergen_batch_and_model
    expected = torch.load(TARGET_PREDICTION_PATH, map_location="cpu", weights_only=False)

    sample = gen_model.sample(
        batch,
        seed=TARGET_PREDICTION_SEED,
        num_steps=TARGET_PREDICTION_NUM_STEPS,
        scale_input_noise=TARGET_PREDICTION_SCALE_INPUT_NOISE,
        disable_tqdm=True,
    ).cpu()

    assert set(sample.keys()) == set(expected.keys())
    for key in expected.keys():
        torch.testing.assert_close(sample[key], expected[key], rtol=0, atol=5e-5)
