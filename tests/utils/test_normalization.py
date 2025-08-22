import numpy as np
import pytest
import torch
import xarray as xr
from tensordict.tensordict import TensorDict

from geoarches.dataloaders.era5_constants import (
    arches_default_level_variables,
    arches_default_pressure_levels,
    arches_default_surface_variables,
)
from geoarches.utils import normalization

LAT = 121
NUM_LEVELS = len(arches_default_pressure_levels)


def create_fake_stats_ds(variables, levels, stats_path):
    ds = xr.Dataset()
    for var_type, var_list in variables.items():
        for var in var_list:
            if var_type == "surface":
                data = np.random.rand(2)
                ds[var] = xr.DataArray(
                    data, dims=["statistic"], coords={"statistic": ["mean", "std"]}
                )
            else:
                data = np.random.rand(2, len(levels))
                ds[var] = xr.DataArray(
                    data,
                    dims=["statistic", "level"],
                    coords={"statistic": ["mean", "std"], "level": levels},
                )
    ds.to_netcdf(stats_path)


def test_init_defaults():
    norm_stats = normalization.NormalizationStatistics()
    assert norm_stats.norm_scheme == "pangu"
    assert norm_stats.variables["surface"] == arches_default_surface_variables
    assert norm_stats.variables["level"] == arches_default_level_variables
    assert norm_stats.levels == arches_default_pressure_levels
    assert norm_stats.loss_weight_per_variable == normalization.default_var_weights


def test_init_graphcast():
    variables = {
        "surface": ["T2m", "U10m"],
        "level": ["Z", "U"],
    }
    levels = [500, 850]
    norm_stats = normalization.NormalizationStatistics(
        variables=variables, levels=levels, norm_scheme="graphcast"
    )
    assert norm_stats.norm_scheme == "graphcast"
    assert norm_stats.variables == variables
    assert norm_stats.levels == levels


def test_init_invalid_scheme():
    with pytest.raises(ValueError):
        normalization.NormalizationStatistics(norm_scheme="invalid")


def test_init_pangu_invalid_vars():
    variables = {
        "surface": ["T2m", "U10m"],
        "level": ["Z", "U"],
    }
    with pytest.raises(AssertionError):
        normalization.NormalizationStatistics(variables=variables, norm_scheme="pangu")


@pytest.mark.parametrize(
    "norm_scheme, variables, levels, stats_path_arg",
    [
        (
            "pangu",
            {
                "surface": arches_default_surface_variables,
                "level": arches_default_level_variables,
            },
            arches_default_pressure_levels,
            None,
        ),
        (
            "pangu",
            {
                "surface": arches_default_surface_variables,
                "level": arches_default_level_variables,
            },
            arches_default_pressure_levels,
            "fake_pangu.nc",
        ),
        (
            "graphcast",
            {
                "surface": ["2m_temperature", "10m_u_component_of_wind"],
                "level": ["geopotential", "u_component_of_wind"],
            },
            [500, 850],
            None,
        ),
        (
            "graphcast",
            {
                "surface": ["2m_temperature", "10m_u_component_of_wind"],
                "level": ["geopotential", "u_component_of_wind"],
            },
            [500, 850],
            "fake_graphcast.nc",
        ),
    ],
    ids=[
        "pangu_default_path",
        "pangu_custom_path",
        "graphcast_default_path",
        "graphcast_custom_path",
    ],
)
def test_load_normalization_stats(
    tmp_path,
    norm_scheme,
    variables,
    levels,
    stats_path_arg,
):
    stats_path = None
    if stats_path_arg is not None:
        stats_path = tmp_path / stats_path_arg
        create_fake_stats_ds(variables, levels, stats_path)
    num_surface_vars = len(variables["surface"])
    num_level_vars = len(variables["level"])
    num_levels = len(levels)

    norm_stats = normalization.NormalizationStatistics(
        norm_scheme=norm_scheme, variables=variables, levels=levels, stats_path=stats_path
    )

    mean, std = norm_stats.load_normalization_stats()
    assert isinstance(mean, TensorDict)
    assert isinstance(std, TensorDict)
    assert "surface" in mean
    assert "level" in mean
    assert "surface" in std
    assert "level" in std
    assert mean["surface"].shape == (num_surface_vars, 1, 1, 1)
    assert mean["level"].shape == (num_level_vars, num_levels, 1, 1)
    assert std["surface"].shape == (num_surface_vars, 1, 1, 1)
    assert std["level"].shape == (num_level_vars, num_levels, 1, 1)


def test_load_graphcast_timedelta_stats():
    variables = {
        "surface": ["2m_temperature", "10m_u_component_of_wind"],
        "level": ["geopotential", "u_component_of_wind"],
    }
    levels = [500, 850]
    norm_stats = normalization.NormalizationStatistics(
        variables=variables, levels=levels, norm_scheme="graphcast"
    )
    surface_stds, level_stds = norm_stats.load_timedelta_stats()
    assert surface_stds.shape == (2, 1, 1, 1)
    assert level_stds.shape == (2, 2, 1, 1)


def test_load_pangu_timedelta_stats():
    """Tests that the pangu timedelta stats are loaded correctly."""
    norm_stats = normalization.NormalizationStatistics(norm_scheme="pangu")
    surface_stds, level_stds = norm_stats.load_timedelta_stats()

    expected_surface_stds = torch.tensor([3.8920, 4.5422, 2.0727, 584.0980]).reshape(-1, 1, 1, 1)
    expected_level_stds = (
        torch.tensor([5.9786e02, 7.4878e00, 8.9492e00, 2.7132e00, 9.5222e-04, 0.3])
        .reshape(-1, 1, 1, 1)
        .expand(-1, NUM_LEVELS, -1, -1)
    )

    assert surface_stds.shape == (4, 1, 1, 1)
    torch.testing.assert_close(surface_stds, expected_surface_stds)
    assert level_stds.shape == (6, NUM_LEVELS, 1, 1)
    torch.testing.assert_close(level_stds, expected_level_stds)


def test_compute_loss_coeffs_shape_pangu():
    norm_stats = normalization.NormalizationStatistics(norm_scheme="pangu")
    loss_coeffs = norm_stats.compute_loss_coeffs(latitude=LAT)
    assert isinstance(loss_coeffs, TensorDict)
    assert "surface" in loss_coeffs
    assert "level" in loss_coeffs
    assert loss_coeffs["surface"].shape == (4, 1, LAT, 1)
    assert loss_coeffs["level"].shape == (6, NUM_LEVELS, LAT, 1)


def test_compute_loss_coeffs_shape_graphcast():
    levels = [500, 850]
    norm_stats = normalization.NormalizationStatistics(levels=levels, norm_scheme="graphcast")
    loss_coeffs = norm_stats.compute_loss_coeffs(latitude=LAT)
    assert isinstance(loss_coeffs, TensorDict)
    assert "surface" in loss_coeffs
    assert "level" in loss_coeffs
    assert loss_coeffs["surface"].shape == (4, 1, LAT, 1)
    assert loss_coeffs["level"].shape == (6, 2, LAT, 1)


def test_compute_loss_coeffs_pangu_no_delta():
    norm_stats = normalization.NormalizationStatistics(norm_scheme="pangu")
    loss_coeffs = norm_stats.compute_loss_coeffs(latitude=LAT, loss_delta_normalization=False)
    assert isinstance(loss_coeffs, TensorDict)
    assert "surface" in loss_coeffs
    assert "level" in loss_coeffs
    assert loss_coeffs["surface"].shape == (4, 1, LAT, 1)
    assert loss_coeffs["level"].shape == (6, NUM_LEVELS, LAT, 1)
