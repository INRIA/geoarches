from datetime import timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import xarray as xr
from geoarches.metrics.label_wrapper import LabelWrapper, convert_metric_dict_to_xarray
from torchmetrics import Metric


@pytest.fixture
def mock_metric():
    # Create a mock Metric class that returns sample tensors for testing.
    mock_metric = MagicMock(spec=Metric)
    mock_metric.compute.return_value = {
        "rmse": torch.tensor([[0.5], [1.0]]),  # (var, lev)
        "mae": torch.tensor([[0.3], [0.8]]),
    }
    return mock_metric


@pytest.fixture
def variable_indices():
    # Sample (var, lev) indices
    return {
        "var1": (0, 0),
        "var2": (1, 0),
    }


class TestVarLevLabel:
    def test_convert_to_labeled_dict(self, mock_metric, variable_indices):
        # Test compute method with labeled dict output
        wrapper = LabelWrapper(
            metric=mock_metric,
            variable_indices=variable_indices,
            lead_time_hours=None,
            rollout_iterations=None,
        )

        wrapper.update()
        output = wrapper.compute()

        torch.testing.assert_close(output["rmse_var1"], torch.tensor(0.5))
        torch.testing.assert_close(output["rmse_var2"], torch.tensor(1.0))
        torch.testing.assert_close(output["mae_var1"], torch.tensor(0.3))
        torch.testing.assert_close(output["mae_var2"], torch.tensor(0.8))


@pytest.fixture
def mock_metric_with_timedelta_dimension():
    # Create a mock Metric class that returns sample tensors for testing.
    mock_metric = MagicMock(spec=Metric)
    mock_metric.compute.return_value = {
        "rmse": torch.tensor(
            [
                # First timestep
                [
                    [0.1, 0.2],  # Variable 1 at 2 levels
                    [0.3, 0.4],  # Variable 2 at 2 levels
                ],
                # Second timestep
                [[0.5, 0.6], [0.7, 0.8]],
                # Third timestep
                [[0.9, 1.0], [1.1, 1.2]],
            ]
        )
    }
    return mock_metric


class TestTimeDeltaLabel:
    def test_convert_to_labeled_dict(self, mock_metric_with_timedelta_dimension, variable_indices):
        # Test compute method with labeled dict output
        wrapper = LabelWrapper(
            metric=mock_metric_with_timedelta_dimension,
            variable_indices=variable_indices,
            lead_time_hours=None,
            rollout_iterations=None,
        )

        wrapper.update()
        output = wrapper.compute()

        # Output includes timedelta dimension.
        torch.testing.assert_close(output["rmse_var1"], torch.tensor([0.1, 0.5, 0.9]))
        torch.testing.assert_close(output["rmse_var2"], torch.tensor([0.3, 0.7, 1.1]))

    def test_convert_to_labeled_dict_with_explicit_timedelta_dimension(
        self, mock_metric_with_timedelta_dimension, variable_indices
    ):
        # Test compute method with labeled dict output
        wrapper = LabelWrapper(
            metric=mock_metric_with_timedelta_dimension,
            variable_indices=variable_indices,
            lead_time_hours=6,  # Separate timedelta dimension.
            rollout_iterations=3,
        )

        wrapper.update()
        output = wrapper.compute()

        # Output includes timedelta dimension.
        torch.testing.assert_close(output["rmse_var1_6h"], torch.tensor(0.1))
        torch.testing.assert_close(output["rmse_var1_12h"], torch.tensor(0.5))
        torch.testing.assert_close(output["rmse_var1_18h"], torch.tensor(0.9))
        torch.testing.assert_close(output["rmse_var2_6h"], torch.tensor(0.3))
        torch.testing.assert_close(output["rmse_var2_12h"], torch.tensor(0.7))
        torch.testing.assert_close(output["rmse_var2_18h"], torch.tensor(1.1))


def test_convert_metric_dict_to_xarray():
    labeled_dict = {
        # T2m
        "mse_T2m_24h": 1.0,
        "mse_T2m_48h": 2.0,
        "var_T2m_24h": 3.0,
        "var_T2m_48h": 4.0,
        # U10
        "mse_U10_24h": 5.0,
        "mse_U10_48h": 6.0,
        "var_U10_24h": 7.0,
        "var_U10_48h": 8.0,
    }

    xr_dataset = convert_metric_dict_to_xarray(
        labeled_dict, extra_dimensions=["prediction_timedelta"]
    )

    xr.testing.assert_equal(
        xr_dataset,
        xr.Dataset(
            data_vars={
                "T2m": xr.DataArray(
                    data=np.array([[1, 2], [3, 4]], dtype=np.float32),
                    dims=("metric", "prediction_timedelta"),
                ),
                "U10": xr.DataArray(
                    data=np.array([[5, 6], [7, 8]], dtype=np.float32),
                    dims=("metric", "prediction_timedelta"),
                ),
            },
            coords={
                "metric": ["mse", "var"],
                "prediction_timedelta": [
                    timedelta(hours=24),
                    timedelta(hours=48),
                ],
            },
        ),
    )


def test_convert_metric_dict_to_xarray_without_timedelta_dimension():
    labeled_dict = {
        # T2m
        "mse_T2m": 1.0,
        "var_T2m": 3.0,
        # U10
        "mse_U10": 5.0,
        "var_U10": 7.0,
    }

    xr_dataset = convert_metric_dict_to_xarray(labeled_dict)

    xr.testing.assert_equal(
        xr_dataset,
        xr.Dataset(
            data_vars={
                "T2m": xr.DataArray(
                    data=np.array([1, 3], dtype=np.float32),
                    dims=("metric"),
                ),
                "U10": xr.DataArray(
                    data=np.array([5, 7], dtype=np.float32),
                    dims=("metric"),
                ),
            },
            coords={
                "metric": ["mse", "var"],
            },
        ),
    )
