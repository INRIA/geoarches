from functools import partial
from importlib import resources
from typing import Callable

import torch
import xarray as xr
from einops import rearrange
from torchmetrics import Metric

from geoarches.dataloaders import era5
from geoarches.metrics.label_wrapper import LabelDictWrapper, add_timedelta_index

from .. import stats as geoarches_stats
from . import metric_base
from .metric_base import MetricBase, TensorDictMetricBase


class BrierSkillScore(Metric, MetricBase):
    """
    Calculate BrierSkillScore as defined in the GenCast Paper.
    It scores probabilities of events predicted by the ensemble distribution, normalized by the reference climatology.
    Optionally, preprocesses preds and targets passed to update() ie. useful for binarization.

    The implementation relies on the assumption that the lat_weights are normalized by the mean (so that the sum of the weights is equal to latitude dim).

    Accepted tensor shapes:
        targets: (batch, ..., lat, lon)
        preds: (batch, nmembers, ..., lat, lon)

    Return dictionary of metrics reduced over batch, lat, lon.
    """

    def __init__(
        self,
        data_shape: tuple,
        preprocess: Callable | None = None,
        compute_lat_weights_fn: Callable[[int], torch.tensor] = metric_base.compute_lat_weights,
    ):
        """
        Since the brier score operates on binary variables, the metric optionally accepts a function to preprocess/binarize predictions and targets.
        See example usage in tests.

        Args:
            data_shape: Shape of tensor to hold computed metric.
                e.g. if targets are shape (batch, timedelta, var, lev, lat, lon) then data_shape = (timedelta, var, lev).
                This class computes metric across batch, lat, lon dimensions.
            preprocess: Takes as input targets and predictions and returns boolean/int tensor if the condition for the event is met.
                Assumes input is (batch, ..., lat, lon).
                Returns same shape as given targets and predictions.
                If not set, this function assumes the predictions and targets are already binarized.
            compute_lat_weights_fn: Function to compute latitude weights given latitude shape.
                Used for error and variance calculations. Expected shape of weights: [..., lat, 1].
                See function example in metric_base.MetricBase.
                Default function assumes latitudes are ordered -90 to 90.
        """
        Metric.__init__(self)
        MetricBase.__init__(self, compute_lat_weights_fn)
        self.preprocess = preprocess

        self.add_state("nsamples", default=torch.tensor(0), dist_reduce_fx="sum")

        for metric_name in (
            "brierscore",
            "fbrierscore",
            "clim_prob",
        ):
            self.add_state(metric_name, default=torch.zeros(data_shape), dist_reduce_fx="sum")

    def update(self, targets: torch.Tensor, preds: torch.Tensor | list[torch.Tensor]) -> None:
        """Update internal state with a batch of targets and predictions.

        Expects inputs to this function to be denormalized.

        Args:
            targets: Target tensor. Expected input shape is (batch, ..., lat, lon)
            preds: Tensor or list of tensors holding ensemble member predictions.
                   If tensor, expected input shape is (batch, nmembers, ..., lat, lon). If list, (batch, ..., lat, lon).
        Returns:
            None
        """
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=1)

        targets, preds = targets.float(), preds.float()
        self.nsamples += preds.shape[0]
        nmembers = preds.shape[1]

        if self.preprocess:
            targets, preds = self.preprocess(targets, preds)
            targets, preds = targets.float(), preds.float()

        pred_ensemble_mean = preds.mean(1)
        mse = self.wmse(pred_ensemble_mean, targets).sum(0)
        weighted_prod = self.weighted_mean(pred_ensemble_mean * (1 - pred_ensemble_mean)).sum(0)

        self.brierscore = self.brierscore + mse
        self.fbrierscore = self.fbrierscore + mse - weighted_prod / (nmembers - 1)

        self.clim_prob = self.clim_prob + self.weighted_mean(targets).sum(0)

    def compute(self) -> torch.Tensor:
        """Compute final metrics utilizing internal states."""
        brierscore = self.brierscore / self.nsamples
        clim_prob = self.clim_prob / self.nsamples
        # Clim brier score can be formulated as (avg of targets^2) - (average of targets)^2
        # However, since targets are binarized, avg of targets^2 = avg of targets, which is clim_prob.
        clim_brierscore = clim_prob - clim_prob.pow(2)

        metrics = dict(
            brierscore=self.brierscore / self.nsamples,
            fbrierscore=self.fbrierscore / self.nsamples,
            brierclimscore=clim_brierscore,
            brierskillscore=(1 - brierscore / clim_brierscore),
        )

        return metrics


def _binarize(high_quantiles, low_quantiles, target, pred=None):
    """
    Binarize for exceedance of the 99.99th, 99.9th, and 99th percentile events and below the 0.01, 0.1
    and 1st percentiles across all lead times.

    Args:
        high_quantiles: high quantiles per lat/lon. Shape: (var q lat lon)
        low_quantiles: low quantiles per lat/lon. Shape: (var q lat lon)
        target: (b, ..., var, lev, lat, lon) where ... can hold extra_dims such as timedelta (lead time).
        pred: (b, members, ..., var, lev, lat, lon).
    """
    # (quantiles, bs, var, lev, lat, lon)
    high_quantiles = rearrange(high_quantiles, "var q lev lat lon -> q 1 var lev lat lon")
    low_quantiles = rearrange(low_quantiles, "var q lev lat lon -> q 1 var lev lat lon")

    # Binarize.
    extra_dims = len(target.shape) - 5
    for _ in range(extra_dims):
        high_quantiles = high_quantiles.unsqueeze(2)
        low_quantiles = low_quantiles.unsqueeze(2)

    target = torch.concat(
        [target > high_quantiles.to(target.device), target < low_quantiles.to(target.device)]
    )
    target = rearrange(target, "q b ... var lev lat lon -> b ... q var lev lat lon")

    if pred is None:
        return target

    pred = torch.concat(
        [
            pred > high_quantiles.unsqueeze(2).to(pred.device),
            pred < low_quantiles.unsqueeze(2).to(pred.device),
        ]
    )
    pred = rearrange(pred, "q b mem ... var lev lat lon -> b mem ... q var lev lat lon")

    return target, pred


class Era5BrierSkillScore(TensorDictMetricBase):
    """Wrapper around BrierSkillScore for calculating brier skill score for weather bench surface and level variables.

    Calculates for surface variables: u and v component of wind, temperature, mean sea level pressure.

    Accepted tensor shapes:
        targets: (batch, ..., var, level, lat, lon)
        preds: (batch, nmembers, ..., var, level, lat, lon)

    Return dictionary of metrics reduced over batch, lat, lon:
        - key: metric_name
        - value: tensor with shape (..., q, var, level)

    Aggregation function compute() returns a dict holding tensors per variable,
    with length (timedelta) where timedelta is lead time (number of model rollouts).
    """

    def __init__(
        self,
        quantiles_filepath="era5-quantiles-2016_2022.nc",
        high_quantile_levels=[0.99, 0.999, 0.9999],
        low_quantiles_levels=[0.01, 0.001, 0.0001],
        surface_variables=era5.surface_variables,
        level_variables=era5.level_variables,
        pressure_levels=era5.pressure_levels,
        lead_time_hours: None | int = None,
        rollout_iterations: None | int = None,
        return_raw_dict: bool = False,
        save_memory: bool = False,
    ):
        """
        Args:
            quantiles_filepath: File to load quantile values from.
            high_quantiles: Quantiles to check for events where data > high_quantile.
            low_quantiles: Quantiles to check for events where data < low_quantile.
            surface_variables: Names of surface variables (to select quantiles).
            level_variables: Names of level variables (to select quantiles).
            pressure_levels: pressure levels to select from quantiles.
            lead_time_hours: Timedelta between timestamps in multistep rollout.
                Set to explicitly handle predictions from multistep rollout.
                This option labels each timestep separately in output metric dict.
                Assumes that data shape of predictions/targets are [batch, ..., multistep, var, lev, lat, lon].
                FYI when set to None, Era5BrierSkillScore still handles natively any extra dimensions in targets/preds.
            rollout_iterations: Size of timedelta dimension (number of rollout iterations in multistep predictions).
                Set to explicitly handle metrics computed on predictions from multistep rollout.
                See param `lead_time_hours`.
            return_raw_dict: Whether to also return the raw output from the metrics.
        """
        # Quantiles for each var across gridpoints and times.
        with resources.as_file(resources.files(geoarches_stats).joinpath(quantiles_filepath)) as f:
            q = xr.open_dataset(f).transpose(..., "latitude", "longitude")
        self.surface_high_quantiles = torch.from_numpy(
            q[era5.surface_variables]
            .sel({"quantile": high_quantile_levels}, method="nearest")
            .to_array()
            .to_numpy()
        ).unsqueeze(-3)  # Add level dimension.
        self.surface_low_quantiles = torch.from_numpy(
            q[era5.surface_variables]
            .sel({"quantile": low_quantiles_levels}, method="nearest")
            .to_array()
            .to_numpy()
        ).unsqueeze(-3)  # Add level dimension.
        self.level_high_quantiles = torch.from_numpy(
            q[level_variables]
            .sel({"quantile": high_quantile_levels, "level": pressure_levels}, method="nearest")
            .to_array()
            .to_numpy()
        )
        self.level_low_quantiles = torch.from_numpy(
            q[level_variables]
            .sel({"quantile": low_quantiles_levels, "level": pressure_levels}, method="nearest")
            .to_array()
            .to_numpy()
        )

        # Variable indices include quantile (var, lev) --> (quantile, var, lev).
        # Enable LabelDictWrapper to extract metrics properly from BrierSkillScore output.
        def _add_quantile_index(variable_indices):
            out = {}
            for var, var_lev_idx in variable_indices.items():
                for quantile_idx, quantile in enumerate(
                    high_quantile_levels + low_quantiles_levels
                ):
                    out[f"{var}_{quantile * 100}%"] = (quantile_idx, *var_lev_idx)
            return out

        # Initialize separate metrics for level vars and surface vars.
        kwargs = {}
        if surface_variables:
            kwargs["surface"] = LabelDictWrapper(
                BrierSkillScore(
                    data_shape=(len(surface_variables), 1),
                    preprocess=partial(
                        _binarize, self.surface_high_quantiles, self.surface_low_quantiles
                    ),
                ),
                variable_indices=add_timedelta_index(
                    _add_quantile_index(era5.get_surface_variable_indices(surface_variables)),
                    lead_time_hours=lead_time_hours,
                    rollout_iterations=rollout_iterations,
                ),
                return_raw_dict=return_raw_dict,
            )
        if level_variables:
            kwargs["level"] = LabelDictWrapper(
                BrierSkillScore(
                    data_shape=(len(level_variables), len(pressure_levels)),
                    preprocess=partial(
                        _binarize, self.level_high_quantiles, self.level_low_quantiles
                    ),
                ),
                variable_indices=add_timedelta_index(
                    _add_quantile_index(
                        era5.get_headline_level_variable_indices(pressure_levels, level_variables)
                    ),
                    lead_time_hours=lead_time_hours,
                    rollout_iterations=rollout_iterations,
                ),
                return_raw_dict=return_raw_dict,
            )
        super().__init__(**kwargs)
