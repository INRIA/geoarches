import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensordict
import torch
import xarray as xr

from geoarches.dataloaders import era5
from geoarches.dataloaders.netcdf import optionally_rename_dimensions
from geoarches.lightning_modules import load_module

# Compatibility alias in case serialized tensordicts reference geoarches.utils.tensordict
if "geoarches.utils.tensordict" not in sys.modules:
    sys.modules["geoarches.utils.tensordict"] = tensordict

GCS_ERA5_DAILY_AVG_PATH = "gs://aimip-era5/era5_1x1_daily_averaged/era5_1978_00h.zarr"
GCS_MONTHLY_FORCING_PATH = "gs://aimip-era5/ERA5-0.25deg-monthly-mean-forcing-1978-2024_regridded_conservative_2025_11_03.zarr"

AIMIP_SURFACE_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "sea_ice_cover",
]
AIMIP_LEVEL_VARIABLES = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "vertical_velocity",
]
AIMIP_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
AIMIP_FORCING_VARIABLES = ["sea_surface_temperature", "sea_ice_cover"]

ROLLOUT_START_TIME = "1978-10-01T00:00:00"
ROLLOUT_END_TIME = "2025-01-01T00:00:00"

DATA_START = np.datetime64("1978-09-29T00:00:00")
FORECAST_START = np.datetime64("1978-09-30T00:00:00")
FORECAST_TARGET = np.datetime64("1978-10-01T00:00:00")
TARGET_PREDICTION_PATH = Path(__file__).with_name("aimip_aw_target_prediction_denormalized.pt")


def set_rollout_timestamp_bounds(
    ds: era5.Era5Forecast,
    start_time: str = ROLLOUT_START_TIME,
    end_time: str = ROLLOUT_END_TIME,
) -> None:
    """Sets the timestamp bounds for the Era5Forecast dataset matching Kauldron RolloutEvaluator."""
    start_ts = pd.Timestamp(start_time)
    start_ts = start_ts - pd.Timedelta(hours=ds.lead_time_hours)
    if ds.load_prev:
        start_ts = start_ts - pd.Timedelta(hours=ds.lead_time_hours)

    end_ts = pd.Timestamp(end_time)
    end_ts = end_ts + ds.multistep * pd.Timedelta(hours=ds.lead_time_hours)

    ds.set_timestamp_bounds(start_ts, end_ts)
    ds.filter_timestamps_by_hour()

    if ds.timestamps[0][-1] != start_ts:
        raise ValueError(
            f"Tried to set timestamp bounds: [{start_ts}, {end_ts}). "
            f"First timestamp is actually: {ds.timestamps[0][-1]}"
        )


def download_aimip_era5_data(data_dir: Path) -> Path:
    """Download the smallest ERA5 slice needed by AIMIP ArchesWeather for one forecast."""
    output_path = data_dir / "daily_averaged" / "era5_1978_00h.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return output_path

    variables = AIMIP_SURFACE_VARIABLES + AIMIP_LEVEL_VARIABLES
    ds = xr.open_zarr(GCS_ERA5_DAILY_AVG_PATH)
    ds = optionally_rename_dimensions(ds)
    ds = ds[variables].sel(time=slice(DATA_START, FORECAST_TARGET))
    ds = ds.sel(level=AIMIP_PRESSURE_LEVELS)
    ds = ds.chunk({"time": -1, "level": -1, "latitude": 181, "longitude": 360})
    ds.to_netcdf(output_path)

    return output_path


def download_aimip_forcings_data(data_dir: Path) -> Path:
    """Download the smallest monthly forcing slice needed for one forecast."""
    output_path = (
        data_dir
        / "forcings"
        / "ERA5-0.25deg-monthly-mean-forcing-1978-2024_regridded_conservative_2025_11_03.nc"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return output_path

    ds = xr.open_zarr(GCS_MONTHLY_FORCING_PATH)
    ds = optionally_rename_dimensions(ds)
    ds = ds[AIMIP_FORCING_VARIABLES].sel(
        time=slice(np.datetime64("1978-09-01"), np.datetime64("1978-11-01"))
    )
    ds.to_netcdf(output_path)

    return output_path


@pytest.fixture(scope="module")
def aimip_archesweather_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("aimip_archesweather")
    download_aimip_era5_data(data_dir)
    download_aimip_forcings_data(data_dir)
    return data_dir


@pytest.fixture(scope="module")
def aimip_archesweather_dataset(aimip_archesweather_data_dir):
    _, config = load_module("aimip-archesweather-m-seed0", device="cpu")

    era5_file = aimip_archesweather_data_dir / "daily_averaged" / "era5_1978_00h.nc"
    forcings_file = (
        aimip_archesweather_data_dir
        / "forcings"
        / "ERA5-0.25deg-monthly-mean-forcing-1978-2024_regridded_conservative_2025_11_03.nc"
    )
    forcings_stats_file = (
        "ERA5-0.25deg-monthly-mean-forcing-1978-2013_regridded_conservative_norm_stats.nc"
    )

    ds = era5.Era5Forecast(
        stats_cfg=config.stats,
        path=str(era5_file.parent),
        domain="aimip_rollout_z00",
        lead_time_hours=24,
        timedelta_hours=24,
        load_prev=True,
        multistep=1,
        set_timestamp_bounds=False,
        variables=config.stats.module.variables,
        dimension_indexers=dict(level=config.stats.module.levels),
        warning_on_nan=False,
        interpolate_input="zero_after_norm",
        forcings_path=str(forcings_file),
        forcings_stats_path=forcings_stats_file,
        forcing_vars=AIMIP_FORCING_VARIABLES,
    )
    set_rollout_timestamp_bounds(ds, start_time=ROLLOUT_START_TIME, end_time=ROLLOUT_END_TIME)
    return ds


def postprocess_pred_state(pred, ds: era5.Era5Forecast):
    """Clips sea_ice_cover in normalized space matching Kauldron WriteEvaluator."""
    pred = pred.clone()
    if "surface" in ds.variables and "sea_ice_cover" in ds.variables["surface"]:
        sic_idx = ds.variables["surface"].index("sea_ice_cover")
        sic_mean = ds.data_mean["surface"][sic_idx, 0, 0, 0].item()
        sic_std = ds.data_std["surface"][sic_idx, 0, 0, 0].item()
        clip_min = -sic_mean / sic_std
        clip_max = (1.0 - sic_mean) / sic_std
        pred["surface"][:, sic_idx] = torch.clamp(
            pred["surface"][:, sic_idx], min=clip_min, max=clip_max
        )
    return pred


@pytest.fixture(scope="module")
def aimip_archesweather_batch_and_model(aimip_archesweather_dataset):
    model, config = load_module("aimip-archesweather-m-seed0", device="cpu")
    batch = {k: v[None].to("cpu") for k, v in aimip_archesweather_dataset[0].items()}
    return batch, model.to("cpu"), config


def test_download_aimip_era5_data(aimip_archesweather_data_dir):
    era5_path = aimip_archesweather_data_dir / "daily_averaged" / "era5_1978_00h.nc"
    assert era5_path.exists()
    with xr.open_dataset(era5_path) as ds:
        assert ds.time.to_numpy()[0].astype("datetime64[s]") == DATA_START
        assert ds.time.to_numpy()[-1].astype("datetime64[s]") == FORECAST_TARGET
        assert len(ds.time) == 3
        variables = AIMIP_SURFACE_VARIABLES + AIMIP_LEVEL_VARIABLES
        assert set(variables).issubset(ds.data_vars)
        assert ds.sizes["latitude"] == 181
        assert ds.sizes["longitude"] == 360
        assert list(ds.level.to_numpy()) == AIMIP_PRESSURE_LEVELS


def test_download_aimip_forcings_data(aimip_archesweather_data_dir):
    forcings_path = (
        aimip_archesweather_data_dir
        / "forcings"
        / "ERA5-0.25deg-monthly-mean-forcing-1978-2024_regridded_conservative_2025_11_03.nc"
    )
    assert forcings_path.exists()
    with xr.open_dataset(forcings_path) as ds:
        assert "sea_surface_temperature" in ds.data_vars
        assert "sea_ice_cover" in ds.data_vars


def test_load_aimip_archesweather_model_with_real_data_batch(
    aimip_archesweather_batch_and_model,
):
    batch, model, config = aimip_archesweather_batch_and_model

    assert model.training is False
    assert next(model.parameters()).device == torch.device("cpu")
    assert {
        "state",
        "next_state",
        "prev_state",
        "timestamp",
        "lead_time_hours",
        "forcings",
    } <= set(batch)
    assert batch["timestamp"].item() == np.datetime64(FORECAST_START, "s").astype(int)


def test_aimip_archesweather_prediction_against_real_outputs(
    aimip_archesweather_batch_and_model,
    aimip_archesweather_dataset,
):
    batch, model, _ = aimip_archesweather_batch_and_model
    ds = aimip_archesweather_dataset
    expected = torch.load(TARGET_PREDICTION_PATH, map_location="cpu", weights_only=False)

    with torch.no_grad():
        pred = model(batch)

    assert set(pred.keys()) == set(expected.keys())
    for key in expected.keys():
        assert pred[key].shape == expected[key].shape
        assert not torch.isnan(pred[key]).any()

    loss = model.loss(pred, batch["next_state"])
    assert not torch.isnan(loss)

    # Postprocess: clip sea_ice_cover in normalized space, then denormalize
    pred = postprocess_pred_state(pred, ds)
    pred = ds.denormalize(pred)

    for key in expected.keys():
        valid_mask = ~batch["nan_mask"][key]
        stds = ds.data_std[key]
        # Since we are comparing denormalized values, normalize the difference
        # by variable std.
        normalized_diff = torch.abs(pred[key] - expected[key]) / stds

        # 99.9% of points must be under 0.03 sigma.
        tolerance = 3e-2  # maximum allowed error is 0.03 std
        mismatched_elements = (normalized_diff[valid_mask] >= tolerance).sum().item()
        total_elements = valid_mask.sum().item()
        p999 = torch.quantile(normalized_diff[valid_mask], 0.999).item()
        assert p999 < tolerance, (
            f"99.9th percentile error too high for key `{key}`: {p999:.4f} sigma\n"
            f"Mismatched elements for {key}: {mismatched_elements} / {total_elements} ({mismatched_elements * 100 / total_elements:.2f}%)\n"
            f"Max normalized difference for {key} is {normalized_diff[valid_mask].max().item():.4f} sigma"
        )

        # No extreme localized divergence (> 0.25 sigma)
        max_sigma_diff = normalized_diff[valid_mask].max().item()
        assert max_sigma_diff < 0.25, (
            f"Max normalized difference for {key} is: {max_sigma_diff:.4f} sigma"
        )

        # torch.testing.assert_close(
        #     pred[key][valid_mask],
        #     expected[key][valid_mask],
        #     # Relax values because we have denormalized values.
        #     rtol=1e-2,
        #     atol=1e-1,
        # )
