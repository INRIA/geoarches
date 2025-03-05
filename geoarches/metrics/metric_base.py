# Base class for metrics.
from typing import Callable, Dict, List

import torch
import torch.nn as nn
import xarray as xr
from tensordict.tensordict import TensorDict
from torchmetrics import Metric


def compute_lat_weights(latitude_resolution: int) -> torch.tensor:
    """Compute latitude coefficients for latititude weighted metrics.

    Assumes latitude coordinates are equidistant and ordered from -90 to 90.

    Args:
        latitude_resolution: latititude dimension size.
    """
    if latitude_resolution == 1:
        return torch.tensor(1.0)
    lat_coeffs_equi = torch.tensor(
        [
            torch.cos(x)
            for x in torch.arange(
                -torch.pi / 2, torch.pi / 2 + 1e-6, torch.pi / (latitude_resolution - 1)
            )
        ]
    )
    lat_coeffs_equi = lat_coeffs_equi / lat_coeffs_equi.mean()
    return lat_coeffs_equi[:, None]


def compute_lat_weights_weatherbench(latitude_resolution: int) -> torch.tensor:
    """Calculate the area overlap as a function of latitude.
    The weatherbench version gives slightly different coeffs
    """
    latitudes = torch.linspace(-90, 90, latitude_resolution)
    points = torch.deg2rad(latitudes)
    pi_over_2 = torch.tensor([torch.pi / 2], dtype=torch.float32)
    bounds = torch.concatenate([-pi_over_2, (points[:-1] + points[1:]) / 2, pi_over_2])
    upper = bounds[1:]
    lower = bounds[:-1]
    # normalized cell area: integral from lower to upper of cos(latitude)
    weights = torch.sin(upper) - torch.sin(lower)
    weights = weights / weights.mean()
    return weights[:, None]


class MetricBase:
    def __init__(
        self,
        compute_lat_weights_fn: Callable[[int], torch.tensor] = compute_lat_weights_weatherbench,
    ):
        """
        Args:
            compute_lat_weights_fn: Function to compute latitude weights given latitude shape.
                Used for error and variance calculations. Expected shape of weights: [..., lat, 1].
        """
        self.compute_lat_weights_fn = compute_lat_weights_fn
        super().__init__()

    def wmse(self, x: torch.Tensor, y: torch.Tensor | int = 0):
        """Latitude weighted mse error.

        Args:
            x: preds with shape (..., lat, lon)
            y: targets with shape (..., lat, lon)
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return (x - y).pow(2).mul(lat_coeffs).nanmean((-2, -1))

    def wmae(self, x: torch.Tensor, y: torch.Tensor | int = 0):
        """Latitude weighted mae error.

        Args:
            x: preds with shape (..., lat, lon)
            y: targets with shape (..., lat, lon)
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return (x - y).abs().mul(lat_coeffs).nanmean((-2, -1))

    def wvar(self, x: torch.Tensor, dim: int = 1):
        """Latitude weighted variance along axis.

        Args:
            x: preds with shape (..., lat, lon)
            dim: over which dimension to compute variance.
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return x.var(dim).mul(lat_coeffs).nanmean((-2, -1))

    def weighted_mean(self, x: torch.Tensor):
        """Latitude weighted mean over grid.

        Args:
            x: preds with shape (..., lat, lon)
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return x.mul(lat_coeffs).nanmean((-2, -1))


class TensorDictMetricBase(Metric):
    """Wrapper around metric that handles targets and preds that are TensorDicts.

    Keeps track of a metric per item in the TensorDict.
    Warning: not compatible with metric.forward() - only used update() and compute().
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: mapping of key to metric.
                Key should match the key in the TensorDict.
                Metric should be an instantiation of a metric class that accepts tensors.

        Example:
            preds = TensorDict(level=torch.tensor(...), surface=torch.tensor(...))
            targets = TensorDict(level=torch.tensor(...), surface=torch.tensor(...))
            metric = TensorDictMetricBase(level=BrierSkillScore(), surface=BrierSkillScore())
            metric.update(targets, preds)
        """
        super().__init__()
        self.metrics = nn.ModuleDict(kwargs)

    def update(self, targets: TensorDict, preds: TensorDict | List[TensorDict]) -> None:
        """Update internal metrics.

        Returns:
            None
        """
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=1)

        for key, metric in self.metrics.items():
            metric.update(targets[key], preds[key])

    def compute(self) -> Dict[str, torch.Tensor]:
        """Return aggregated collections of the computed metrics.

        If metrics return dictionary: returns one aggregated dict.
        If metrics return xarray: returns metrged xarray dataset.
        """
        out_dict = dict()
        out_xarrays = []

        for metric in self.metrics.values():
            # Collect returned values from each metric.
            return_values = metric.compute()
            if not isinstance(return_values, tuple):
                return_values = [return_values]
            for return_value in return_values:
                # Handle returned dictionary.
                if isinstance(return_value, dict):
                    out_dict.update(return_value)
                # Handle returned xarray dataset.
                elif isinstance(return_value, xr.Dataset):
                    out_xarrays.append(return_value)
                else:
                    raise ValueError(
                        f"TensorDictMetricBase cannot handle metric return type: {type(return_value)}"
                    )
        if out_xarrays:
            out_xarray = xr.merge(out_xarrays)
            if out_dict:
                return out_dict, out_xarray
            return out_xarray
        return out_dict

    def reset(self):
        """
        Reset states of all metrics.
        """
        for metric in self.metrics.values():
            metric.reset()
