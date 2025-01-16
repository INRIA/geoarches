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
    The weatherbench version gives slightly different coeffs.
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
    """Implement latitude-weighted base functions."""

    def __init__(
        self,
        compute_lat_weights_fn: Callable[[int], torch.tensor] = compute_lat_weights_weatherbench,
    ):
        """
        Args:
            variable_indices: dict used to extract indices from output tensor.
            compute_lat_weights_fn: Function to compute latitude weights given latitude shape.
                Used for error and variance calculations. Expected shape of weights: [..., lat, 1].
        """
        super().__init__()
        self.compute_lat_weights_fn = compute_lat_weights_fn

    def wmse(self, x: torch.Tensor, y: torch.Tensor | int = 0):
        """Latitude weighted mse error.

        Args:
            x: preds with shape (..., lat, lon)
            y: targets with shape (..., lat, lon)
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return (x - y).pow(2).mul(lat_coeffs).mean((-2, -1))

    def wmae(self, x: torch.Tensor, y: torch.Tensor | int = 0):
        """Latitude weighted mae error.

        Args:
            x: preds with shape (..., lat, lon)
            y: targets with shape (..., lat, lon)
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return (x - y).abs().mul(lat_coeffs).mean((-2, -1))

    def wvar(self, x: torch.Tensor, dim: int = 1):
        """Latitude weighted variance along axis.

        Args:
            x: preds with shape (..., lat, lon)
            dim: over which dimension to compute variance.
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return x.var(dim).mul(lat_coeffs).mean((-2, -1))

    def weighted_mean(self, x: torch.Tensor):
        """Latitude weighted mean over grid.

        Args:
            x: preds with shape (..., lat, lon)
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return x.mul(lat_coeffs).mean((-2, -1))


class TensorDictMetricBase(Metric):
    """Wrapper around metric to enable handling of targets and preds that are TensorDicts.

    Assumes metric should accept tensor target and pred.
    Keeps track of a metric instantiation per item in the TensorDict.

    Warning: not compatible with metric.forward() - only use update() and compute().
    See https://github.com/Lightning-AI/torchmetrics/issues/987#issuecomment-2419846736.
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
            metric.update(targets=targets[key], preds=preds[key])

    def compute(self) -> Dict[str, torch.Tensor]:
        """Return aggregated collections of the computed metrics.

        Elements from each metric are aggregated. Handles multiple return values per metric.
        Assumes all metrics return the same number of outputs.
        """
        aggregated_outputs = []

        for key, metric in self.metrics.items():
            # Collect returned values from each metric.
            outputs = metric.compute()
            if not isinstance(outputs, tuple):
                outputs = [outputs]
            for i, output in enumerate(outputs):
                # Handle returned dictionary.
                if isinstance(output, dict):
                    if len(aggregated_outputs) - 1 < i:
                        aggregated_outputs.append({})
                    if aggregated_outputs[i].keys().isdisjoint(output.keys()):
                        aggregated_outputs[i].update(output)
                    else:
                        aggregated_outputs[i].update({f"{k}_{key}": v for k, v in output.items()})
                # Handle returned xarray dataset.
                elif isinstance(output, xr.Dataset):
                    if len(aggregated_outputs) - 1 < i:
                        aggregated_outputs.append([])
                    aggregated_outputs[i].append(output)

        for output in aggregated_outputs:
            if isinstance(output, list):
                merged_dataset = xr.merge(output)
                aggregated_outputs[i] = merged_dataset

        if len(aggregated_outputs) == 1:
            return aggregated_outputs[0]
        return aggregated_outputs

    def reset(self):
        """
        Reset states of all metrics.
        """
        for metric in self.metrics.values():
            metric.reset()
