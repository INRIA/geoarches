from unittest.mock import MagicMock

import numpy as np
import torch
import xarray as xr
from torchmetrics import Metric

from geoarches.metrics.metric_base import TensorDictMetricBase


class TestTensorDictMetricBase:
    def test_calls_update(self):
        mock_level_metric = MagicMock(spec=Metric)
        mock_surface_metric = MagicMock(spec=Metric)
        metric = TensorDictMetricBase(level=mock_level_metric, surface=mock_surface_metric)

        fake_targets = {"level": torch.tensor([1]), "surface": torch.tensor([2])}
        fake_preds = {"level": torch.tensor([3]), "surface": torch.tensor([4])}
        metric.update(fake_targets, fake_preds)

        mock_level_metric.update.assert_called_once_with(
            targets=torch.tensor([1]), preds=torch.tensor([3])
        )
        mock_surface_metric.update.assert_called_once_with(
            targets=torch.tensor([2]), preds=torch.tensor([4])
        )

    def test_aggregate_computed_labeled_dicts(self):
        mock_level_metric = MagicMock(spec=Metric)
        mock_level_metric.compute.return_value = {
            "rmse_var1": torch.tensor(1.0),
            "mae_var1": torch.tensor(3.0),
        }
        mock_surface_metric = MagicMock(spec=Metric)
        mock_surface_metric.compute.return_value = {
            "rmse_var2": torch.tensor(2.0),
            "mae_var2": torch.tensor(4.0),
        }
        metric = TensorDictMetricBase(level=mock_level_metric, surface=mock_surface_metric)

        fake_inputs = {"level": torch.zeros(0), "surface": torch.zeros(0)}
        metric.update(fake_inputs, fake_inputs)
        output = metric.compute()

        mock_level_metric.update.assert_called_once()
        mock_surface_metric.update.assert_called_once()
        assert output == {
            "rmse_var1": torch.tensor(1.0),
            "rmse_var2": torch.tensor(2.0),
            "mae_var1": torch.tensor(3.0),
            "mae_var2": torch.tensor(4.0),
        }

    def test_suffix_metric_key_to_avoid_overwriting_elements(self):
        mock_level_metric = MagicMock(spec=Metric)
        mock_level_metric.compute.return_value = {
            "rmse": torch.tensor(1.0),
        }
        mock_surface_metric = MagicMock(spec=Metric)
        mock_surface_metric.compute.return_value = {
            "rmse": torch.tensor(2.0),
        }
        metric = TensorDictMetricBase(level=mock_level_metric, surface=mock_surface_metric)

        fake_inputs = {"level": torch.zeros(0), "surface": torch.zeros(0)}
        metric.update(fake_inputs, fake_inputs)
        output = metric.compute()

        mock_level_metric.update.assert_called_once()
        mock_surface_metric.update.assert_called_once()
        assert output == {
            "rmse": torch.tensor(1.0),
            "rmse_surface": torch.tensor(2.0),  # Added suffix to metric key.
        }

    def test_aggregate_computed_xr_datasets(self):
        mock_level_metric = MagicMock(spec=Metric)
        mock_level_metric.compute.return_value = (
            xr.Dataset(
                data_vars={
                    "var1": xr.DataArray(
                        data=np.array([0.5, 0.3], dtype=np.float32),
                        dims=("metric"),
                    ),
                    "var2": xr.DataArray(
                        data=np.array([0.7, 0.8], dtype=np.float32),
                        dims=("metric"),
                    ),
                },
                coords={
                    "metric": ["rmse", "mae"],
                },
            ),
        )
        mock_surface_metric = MagicMock(spec=Metric)
        mock_surface_metric.compute.return_value = (
            xr.Dataset(
                data_vars={
                    "var3": xr.DataArray(
                        data=np.array([1.5, 1.3], dtype=np.float32),
                        dims=("metric"),
                    ),
                    "var4": xr.DataArray(
                        data=np.array([1.7, 1.8], dtype=np.float32),
                        dims=("metric"),
                    ),
                },
                coords={
                    "metric": ["rmse", "mae"],
                },
            ),
        )
        metric = TensorDictMetricBase(level=mock_level_metric, surface=mock_surface_metric)
        fake_inputs = {"level": torch.zeros(0), "surface": torch.zeros(0)}
        metric.update(fake_inputs, fake_inputs)
        output = metric.compute()
        xr.testing.assert_equal(
            output,
            xr.Dataset(
                data_vars={
                    "var1": xr.DataArray(
                        data=np.array([0.5, 0.3], dtype=np.float32),
                        dims=("metric"),
                    ),
                    "var2": xr.DataArray(
                        data=np.array([0.7, 0.8], dtype=np.float32),
                        dims=("metric"),
                    ),
                    "var3": xr.DataArray(
                        data=np.array([1.5, 1.3], dtype=np.float32),
                        dims=("metric"),
                    ),
                    "var4": xr.DataArray(
                        data=np.array([1.7, 1.8], dtype=np.float32),
                        dims=("metric"),
                    ),
                },
                coords={
                    "metric": ["rmse", "mae"],
                },
            ),
        )
