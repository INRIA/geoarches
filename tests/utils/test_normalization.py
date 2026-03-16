import pytest
from tensordict import TensorDict

from geoarches.dataloaders.era5 import (
    arches_default_level_variables,
    arches_default_pressure_levels,
    arches_default_surface_variables,
)
from geoarches.utils import normalization


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


def test_load_normalization_stats_pangu():
    norm_stats = normalization.NormalizationStatistics(norm_scheme="pangu")
    mean, std = norm_stats.load_normalization_stats()
    assert isinstance(mean, TensorDict)
    assert isinstance(std, TensorDict)
    assert "surface" in mean
    assert "level" in mean
    assert "surface" in std
    assert "level" in std
    assert mean["surface"].shape == (4, 1, 1, 1)
    assert mean["level"].shape == (6, 13, 1, 1)
    assert std["surface"].shape == (4, 1, 1, 1)
    assert std["level"].shape == (6, 13, 1, 1)


def test_load_normalization_stats_graphcast():
    variables = {
        "surface": ["2m_temperature", "10m_u_component_of_wind"],
        "level": ["geopotential", "u_component_of_wind"],
    }
    levels = [500, 850]
    norm_stats = normalization.NormalizationStatistics(
        variables=variables, levels=levels, norm_scheme="graphcast"
    )
    mean, std = norm_stats.load_normalization_stats()
    assert isinstance(mean, TensorDict)
    assert isinstance(std, TensorDict)
    assert "surface" in mean
    assert "level" in mean
    assert "surface" in std
    assert "level" in std
    assert mean["surface"].shape == (2, 1, 1, 1)
    assert mean["level"].shape == (2, 2, 1, 1)
    assert std["surface"].shape == (2, 1, 1, 1)
    assert std["level"].shape == (2, 2, 1, 1)


def test_load_graphcast_timedelta_stats():
    variables = {
        "surface": ["2m_temperature", "10m_u_component_of_wind"],
        "level": ["geopotential", "u_component_of_wind"],
    }
    levels = [500, 850]
    norm_stats = normalization.NormalizationStatistics(
        variables=variables, levels=levels, norm_scheme="graphcast"
    )
    surface_stds, level_stds = norm_stats.load_graphcast_timedelta_stats()
    assert surface_stds.shape == (2, 1, 1, 1)
    assert level_stds.shape == (2, 2, 1, 1)


def test_compute_loss_coeffs_pangu():
    norm_stats = normalization.NormalizationStatistics(norm_scheme="pangu")
    loss_coeffs = norm_stats.compute_loss_coeffs(latitude=121)
    assert isinstance(loss_coeffs, TensorDict)
    assert "surface" in loss_coeffs
    assert "level" in loss_coeffs
    assert loss_coeffs["surface"].shape == (4, 1, 121, 1)
    assert loss_coeffs["level"].shape == (6, 13, 121, 1)


def test_compute_loss_coeffs_graphcast():
    levels = [500, 850]
    norm_stats = normalization.NormalizationStatistics(levels=levels, norm_scheme="graphcast")
    loss_coeffs = norm_stats.compute_loss_coeffs(latitude=121)
    assert isinstance(loss_coeffs, TensorDict)
    assert "surface" in loss_coeffs
    assert "level" in loss_coeffs
    assert loss_coeffs["surface"].shape == (4, 1, 121, 1)
    assert loss_coeffs["level"].shape == (6, 2, 121, 1)


def test_compute_loss_coeffs_pangu_no_delta():
    norm_stats = normalization.NormalizationStatistics(norm_scheme="pangu")
    loss_coeffs = norm_stats.compute_loss_coeffs(latitude=121, loss_delta_normalization=False)
    assert isinstance(loss_coeffs, TensorDict)
    assert "surface" in loss_coeffs
    assert "level" in loss_coeffs
    assert loss_coeffs["surface"].shape == (4, 1, 121, 1)
    assert loss_coeffs["level"].shape == (6, 13, 121, 1)
