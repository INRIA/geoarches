import itertools
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import Tensor
from torchmetrics import Metric


class LabelDictWrapper(Metric):
    """Wrapper class around metric for extracting metric outputs into a labelled dictionary.
    Helpful for WandB which needs to log single values.

    Expects the metric to return a dictionary holding computed metrics:
        - keys: metric_name
        - values: torch tensors with shape (..., *(variable_index))
                  variable_index is passed in with param `variable_indices`

    LabelDictWrapper returns a dictionary of computed metrics:
        - keys: <metric_name>_<variable_name>
        - value: torch tensors with shape (...)

    Warning: this class is not compatible with forward(), only use update() and compute().
    See https://github.com/Lightning-AI/torchmetrics/issues/987#issuecomment-2419846736.

    Example:
        metric = LabelDictWrapper(EnsembleMetrics(preprocess=preprocess_fn),
                              variable_indices=dict(T2m_24h=(0, 2, 0), T2m_48h=(1, 2, 0), U10_24h=(0, 0, 0)), U10_48h=(1, 0, 0)))
        targets, preds = torch.tensor(batch, timedelta, var, lev, lat, lon), torch.tensor(batch, nmem, timedelta, var, lev, lat, lon)
        metric.update(targets, preds)
        labeled_dict = metric.compute()  # EnsembleMetrics returns {"mse": torch.tensor(timedelta, var, lev) }
        labelled_dict = {"mse_T2m_24h": ..., "mse_T2m_48h": ..., "mse_U10_24h": ..., "mse_U10_48h": ...}

    Args:
        metric: base metric that should be wrapped. It is assumed that the metric outputs a
            dict mapping metric name to tensors that have shape (..., *(variable_index)).
        variable_indices: Mapping from variable name to index (ie. var, lev) into tensor holding computed metric.
                ie. dict(T2m=(2, 0), U10=(0, 0), V10=(1, 0), SP=(3, 0)).
    """

    def __init__(
        self,
        metric: Metric,
        variable_indices: Dict[str, tuple],
    ):
        super().__init__()
        if not isinstance(metric, Metric):
            raise ValueError(
                f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}"
            )
        self.metric = metric
        self.variable_indices = variable_indices

    def _convert(self, raw_metric_dict: Dict[str, Tensor]):
        # Label metrics.
        labeled_dict = dict()
        for var, index in self.variable_indices.items():
            for metric_name, metric in raw_metric_dict.items():
                labeled_dict[f"{metric_name}_{var}"] = metric.__getitem__((..., *index))
        return labeled_dict

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.metric.update(*args, **kwargs)

    def compute(self) -> Dict[str, Tensor]:
        return self._convert(self.metric.compute())

    def reset(self) -> None:
        """Reset metric."""
        self.metric.reset()
        super().reset()


def add_timedelta_index(
    variable_indices: dict[str, tuple],
    lead_time_hours: None | int = None,
    rollout_iterations: None | int = None,
):
    """Add prediction_timedelta dimension to variable indices for LabelDictWrapper.

    For example: if variable indexes are (var, lev).
    Returns indexes with (timedelta, var, lev).
    Means that LabelDictWrapper expects metric to return metrics with shape (..., timedelta, var, lev).

    Args:
        variable_indices: Mapping from variable name to index (ie. var, lev).
        lead_time_hours: time delta between timesteps in multistep rollout.
        rollout_iterations: Number of rollout iterations in multistep predictions. ie. Size of prediction_timdelta dimension.
    """
    if lead_time_hours is None or rollout_iterations is None:
        return variable_indices
    indices = {}
    for var, index in variable_indices.items():
        for i in range(rollout_iterations):
            lead_time = lead_time_hours * (i + 1)
            indices[f"{var}_{lead_time}h"] = (i, *index)
    return indices


def convert_metric_dict_to_xarray(
    labeled_dict: Dict[str, Tensor], extra_dimensions: List[str] = []
):
    """
    Convert labeled dict with metrics (output of LabelDictWrapper) to labeled xarray.

    Example:
        labelled_dict = {"mse_T2m_24h": 1.0, "mse_T2m_48h": 2.0}
        ds = convert_metric_dict_to_xarray(labelled_dict, extra_dimensions=['prediction_timedelta'])

    Args:
        labeled_dict: Mapping to metric tenors. Expects keys to be formatted as "<metric>_<var>_<dim1>_<dim2>_..._"
            where the separator between dimensions is an underscore.
        extra_dimensions: list of dimension names, if any extra beyond <metric> and <var>.
    """
    if "metadata" in labeled_dict:
        del labeled_dict["metadata"]

    def _convert_coord(name, value):
        if "timedelta" in name:
            return pd.to_timedelta(value)
        if "bin" in name:
            return int(value)
        else:
            return value

    # Collect coordinates.
    variables = set()
    metrics = set()
    coords = defaultdict(set)
    for label in labeled_dict:
        labels = label.split("_")
        if len(labels) - 2 != len(extra_dimensions):
            raise ValueError(
                f"Expected length of extra_dimensions for key {label} to be: {len(labels) - 2}. Got extra_dimensions={extra_dimensions}."
            )
        metrics.add(labels[0])
        variables.add(labels[1])
        for i, dim in enumerate(extra_dimensions):
            coords[dim].add(labels[i + 2])

    dimension_shape = [len(coord) for coord in (metrics, *coords.values())]
    # Sort coordinates.
    metrics = sorted(list(metrics))
    for k, coord in coords.items():
        coords[k] = sorted(list(coord), key=lambda x: _convert_coord(k, x))

    # Aggregate data arrays by variable.
    dimensions = ["metric"] + extra_dimensions
    data_arrays = {}
    for var in variables:
        data = []
        for dims in itertools.product(metrics, *coords.values()):
            metric, other_dims = dims[0], dims[1:]
            other_dims = "_" + "_".join(other_dims) if other_dims else ""
            key = f"{metric}_{var}{other_dims}"
            data.append(labeled_dict[key])
        data = np.array(data).reshape(dimension_shape)
        data_arrays[var] = (dimensions, data)

    # Prepare coordinates.
    for k, coord in coords.items():
        coords[k] = [_convert_coord(k, x) for x in coord]
    coords["metric"] = metrics

    return xr.Dataset(data_vars=data_arrays, coords=coords)


class LabelXarrayWrapper(Metric):
    """Wrapper class around metric for extracting metric outputs into a labelled xarray.
    Helpful for easier analysis.

    Expects the metric to return a dictionary holding computed metrics:
        - keys: metric name
        - values: torch tensor with shape (dim1,  dim2, ...)

    LabelXarrayWrapper returns xarray of computed metrics:
        - with variable as data variables and (metric, dim1, dim2) as dimensions.

    Warning: this class is not compatible with forward(), only use update() and compute().
    See https://github.com/Lightning-AI/torchmetrics/issues/987#issuecomment-2419846736.

    Example:
        metric = LabelXarrayWrapper(EnsembleMetrics(preprocess=preprocess_fn),
                                    dims = ['variable', 'level'],
                                    coords= (['T2m','U10m'], [500, 750, 800])
        targets, preds = torch.tensor(batch, var, lev, lat, lon), torch.tensor(batch, var, lev, lat, lon)
        metric.update(targets, preds)
        xr_dataset = metric.compute()  # EnsembleMetrics internally returns {"mse": torch.tensor(var, lev) }

    Args:
        metric: base metric that should be wrapped. It is assumed that the metric outputs a
            dict mapping metric name to tensors that have shape (dim1, dim2, ...).
        dims: Names of the dimensions returned by metric (same order as tensor shape and `coords`).
        coords: Values for the dimensions returned by metric (same order as tensor shape and `dims`).
    """

    def __init__(
        self,
        metric: Metric,
        dims: Sequence[str],
        coords: Sequence[Sequence],
    ):
        super().__init__()
        if not isinstance(metric, Metric):
            raise ValueError(
                f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {metric}."
            )
        if "variable" not in dims:
            raise ValueError(
                "One dimension needs to be named 'variable'. It will be used as variable in xarray dataset."
            )
        self.metric = metric
        self.dims = dims
        self.coords = coords

    def _convert(self, metric_dict: dict[str, torch.tensor]) -> xr.Dataset:
        ds = xr.Dataset(
            data_vars={metric: (self.dims, val) for metric, val in metric_dict.items()},
            coords={dim: coord for dim, coord in zip(self.dims, self.coords)},
        )
        ds = ds.to_array(dim="metric")  # Stack metrics into new dim.
        ds = ds.to_dataset(dim="variable")  # Split into separate variables along var dimension.
        return ds

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.metric.update(*args, **kwargs)

    def compute(self) -> dict[str, torch.tensor]:
        return self._convert(self.metric.compute())
