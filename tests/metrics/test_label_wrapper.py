from datetime import timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import xarray as xr
from torchmetrics import Metric

from geoarches.metrics.label_wrapper import (
    LabelDictWrapper,
    LabelXarrayWrapper,
    add_timedelta_index,
    convert_metric_dict_to_xarray,
)


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
        wrapper = LabelDictWrapper(
            metric=mock_metric,
            variable_indices=variable_indices,
        )

        wrapper.update()
        output = wrapper.compute()

        torch.testing.assert_close(output["rmse_var1"], torch.tensor(0.5))
        torch.testing.assert_close(output["rmse_var2"], torch.tensor(1.0))
        torch.testing.assert_close(output["mae_var1"], torch.tensor(0.3))
        torch.testing.assert_close(output["mae_var2"], torch.tensor(0.8))

    def test_convert_to_xarray(self, mock_metric, variable_indices):
        # Test compute method with labeled dict output
        wrapper = LabelXarrayWrapper(
            metric=mock_metric,
            coord_names=["variable", "level"],
            coords=[
                ["var1", "var2"],
                [1],
            ],
        )

        wrapper.update()
        output = wrapper.compute()

        xr.testing.assert_equal(
            output,
            xr.Dataset(
                data_vars={
                    "rmse": xr.DataArray(
                        data=np.array([[0.5], [1.0]], dtype=np.float32),
                        dims=("variable", "level"),
                    ),
                    "mae": xr.DataArray(
                        data=np.array([[0.3], [0.8]], dtype=np.float32),
                        dims=("variable", "level"),
                    ),
                },
                coords={
                    "variable": ["var1", "var2"],
                    "level": [1],
                },
            ),
        )


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
        wrapper = LabelDictWrapper(
            metric=mock_metric_with_timedelta_dimension,
            variable_indices=variable_indices,
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
        wrapper = LabelDictWrapper(
            metric=mock_metric_with_timedelta_dimension,
            variable_indices=add_timedelta_index(
                variable_indices,
                lead_time_hours=6,  # Separate timedelta dimension.
                rollout_iterations=3,
            ),
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
                "mse": xr.DataArray(
                    data=np.array([[1, 2], [5, 6]], dtype=np.float32),
                    dims=("variable", "prediction_timedelta"),
                ),
                "var": xr.DataArray(
                    data=np.array([[3, 4], [7, 8]], dtype=np.float32),
                    dims=("variable", "prediction_timedelta"),
                ),
            },
            coords={
                "variable": ["T2m", "U10"],
                "prediction_timedelta": [
                    timedelta(hours=24),
                    timedelta(hours=48),
                ],
            },
        ),
    )


def test_convert_metric_dict_to_xarray_with_bins_dimension():
    labeled_dict = {
        # T2m
        "rankhist_T2m_1_24h": 1.0,
        "rankhist_T2m_1_48h": 2.0,
        "rankhist_T2m_2_24h": 3.0,
        "rankhist_T2m_2_48h": 4.0,
    }

    xr_dataset = convert_metric_dict_to_xarray(
        labeled_dict, extra_dimensions=["bins", "prediction_timedelta"]
    )

    xr.testing.assert_equal(
        xr_dataset,
        xr.Dataset(
            data_vars={
                "rankhist": xr.DataArray(
                    data=np.array([[[1, 2], [3, 4]]], dtype=np.float32),
                    dims=("variable", "bins", "prediction_timedelta"),
                ),
            },
            coords={
                "variable": ["T2m"],
                "prediction_timedelta": [timedelta(hours=24), timedelta(hours=48)],
                "bins": [1, 2],
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
                "mse": xr.DataArray(
                    data=np.array([1, 5], dtype=np.float32),
                    dims=("variable"),
                ),
                "var": xr.DataArray(
                    data=np.array([3, 7], dtype=np.float32),
                    dims=("variable"),
                ),
            },
            coords={
                "variable": ["T2m", "U10"],
            },
        ),
    )
