from typing import Callable, Dict, List

import torch
from torchmetrics import Metric

from geoarches.dataloaders import era5
from geoarches.metrics.label_wrapper import LabelDictWrapper, add_timedelta_index

from . import metric_base
from .metric_base import MetricBase, TensorDictMetricBase


class EnsembleMetrics(Metric, MetricBase):
    """
    Metrics for ensemble prediction including:
    - ensemble mean RMSE
    - ensemble spread
    - spread skill ratio
    - CRPS

    Accepted tensor shapes:
        targets: (batch, ..., lat, lon)
        preds: (batch, nmembers, ..., lat, lon)

    Return dictionary of metrics reduced over batch, lat, lon.
    """

    def __init__(
        self,
        data_shape: tuple,
        compute_lat_weights_fn: Callable[
            [int], torch.tensor
        ] = metric_base.compute_lat_weights_weatherbench,
        # lead_time_hours: None | int = None,
        # rollout_iterations: int = 1,
        save_memory: bool = False,
        # variable_indices: Dict[str, tuple] | None = None,
    ):
        """
        Args:
            data_shape: Shape of tensor holding computed metric.
                e.g. if targets are shape (batch, timedelta, var, lev, lat, lon) then data_shape = (timedelta, var, lev).
                This class computes metric across batch, lat, lon dimensions.
            compute_lat_weights_fn: Function to compute latitude weights given latitude shape.
                Used for error and variance calculations. Expected shape of weights: [..., lat, 1].
                See function example in metric_base.MetricBase.
                Default function assumes latitudes are ordered -90 to 90.
            save_memory: compute dispersion in memory-concious fashion (avoid broadcasting on added member dimension).
                         Recommended when number of ensemble members exceeds 10.
        """
        Metric.__init__(self)
        MetricBase.__init__(
            self,
            compute_lat_weights_fn=compute_lat_weights_fn,
            # variable_indices=variable_indices,
            # lead_time_hours=lead_time_hours,
            # rollout_iterations=rollout_iterations,
        )

        # Call `self.add_state`for every internal state that is needed for the metrics computations.
        # `dist_reduce_fx` indicates the function that should be used to reduce.
        self.add_state("nsamples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("nmembers", default=torch.tensor(0), dist_reduce_fx="sum")

        metric_states = ["mse", "var", "mae", "dispersion", "energy_rmse", "energy_dispersion"]
        for state in metric_states:
            self.add_state(state, default=torch.zeros(data_shape), dist_reduce_fx="sum")

        self.save_memory = save_memory

    def update(self, targets: torch.Tensor, preds: torch.Tensor | List[torch.Tensor]) -> None:
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

        self.nsamples += preds.shape[0]
        self.nmembers += preds.shape[0] * preds.shape[1]  # Total member predictions

        pred_ensemble_mean = preds.mean(1)

        # for auto-broadcast
        self.mse = self.mse + self.wmse(targets, pred_ensemble_mean).sum(0)
        self.var = self.var + self.wvar(preds).sum(0)

        # For CRPS.
        self.mae = self.mae + self.wmae(preds, targets.unsqueeze(1)).mean(1).sum(0)

        self.energy_rmse = self.energy_rmse + self.wmse(preds, targets.unsqueeze(1)).sqrt().mean(
            1
        ).sum(0)

        if self.save_memory:
            nmembers = preds.shape[1]
            dispersion = 0
            energy_dispersion = 0
            for i in range(nmembers):
                dispersion += (preds[:, [i]] - preds).abs().sum(1)  # in-place addition
                sq_diff = (preds[:, [i]] - preds).pow(2)
                energy_dispersion += self.weighted_mean(sq_diff).sqrt().sum(1)  # in-place addition

            dispersion /= nmembers**2
            dispersion = self.weighted_mean(dispersion).sum(0)
            self.dispersion = self.dispersion + dispersion

            energy_dispersion /= nmembers**2
            energy_dispersion = energy_dispersion.sum(0)
            self.energy_dispersion = self.energy_dispersion + energy_dispersion
        else:
            # Faster but takes more memory due to expensive broadcasting.
            self.dispersion = self.dispersion + (
                self.wmae(preds.unsqueeze(1), preds.unsqueeze(2)).mean((1, 2)).sum(0)
            )
            self.energy_dispersion = self.energy_dispersion + (
                self.wmse(preds.unsqueeze(1), preds.unsqueeze(2)).sqrt().mean((1, 2)).sum(0)
            )

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute final metrics utilizing internal states.
        Returns:
            Dict: mapping metric name to tensor holding computed metric.
                  holds one tensor per variable and metric pair ie. mse_wind_speed.
        """
        nmembers = self.nmembers / self.nsamples  # nmembers per sample
        spread_skill_ratio_coeff = (1 + 1 / nmembers) ** 0.5

        # f{metric} is {metric} without statistical bias due to ensemble size.
        # for frmse see
        # https://github.com/google-research/weatherbench2/blob/main/weatherbench2/metrics.py#L500

        assert self.nsamples > 0, "nsamples is zero, division by zero."
        assert nmembers > 0, "nmembers is zero, division by zero."

        metrics = dict(
            mse=self.mse / self.nsamples,
            frmse=(self.mse / self.nsamples - self.var / self.nsamples / nmembers).sqrt(),
            var=self.var / self.nsamples,
            spskr=spread_skill_ratio_coeff * (self.var / self.mse).sqrt(),  # this is unbiased
            crps=(self.mae - 0.5 * self.dispersion) / self.nsamples,
            fcrps=(self.mae - 0.5 * self.dispersion * nmembers / (nmembers - 1)) / self.nsamples,
            energyscore=(self.energy_rmse - 0.5 * self.energy_dispersion) / self.nsamples,
            fenergyscore=(
                self.energy_rmse - 0.5 * self.energy_dispersion * nmembers / (nmembers - 1)
            )
            / self.nsamples,
        )

        return metrics


class Era5EnsembleMetrics(TensorDictMetricBase):
    """Wrapper class around EnsembleMetrics for computing over surface and level variables.

    Handles batches coming from Era5 Dataloader.

    Accepted tensor shapes:
        targets: (batch, ..., timedelta, var, level, lat, lon)
        preds: (batch, nmembers, ..., timedelta, var, level, lat, lon)

    Metrics are reduced over batch, lat, lon.
    Returns a labelled dictionary for each (timedelta, var, lev).
    """

    def __init__(
        self,
        surface_variables=era5.surface_variables,
        level_variables=era5.level_variables,
        pressure_levels=era5.pressure_levels,
        save_memory: bool = False,
        lead_time_hours: None | int = None,
        rollout_iterations: None | int = None,
        return_raw_dict: bool = False,
    ):
        """
        Args:
            surface_variables: Names of surface variables (to select quantiles).
            level_variables: Names of level variables (used to get `variable_indices`).
            pressure_levels: pressure levels in data (used to get `variable_indices`).
            save_memory: compute dispersion in memory-concious fashion (avoid broadcasting on added member dimension).
                         Recommended when number of ensemble members exceeds 10.
            lead_time_hours: Timedelta between timestamps in multistep rollout.
                Set to explicitly handle predictions from multistep rollout.
                This option labels each timestep separately in output metric dict.
                Assumes that data shape of predictions/targets are [batch, ..., multistep, var, lev, lat, lon].
                FYI when set to None, Era5EnsembleMetrics still handles natively any extra dimensions in targets/preds.
            rollout_iterations: Size of timedelta dimension (number of rollout iterations in multistep predictions).
                Set to explicitly handle metrics computed on predictions from multistep rollout.
                See param `lead_time_hours`.
            return_raw_dict: Whether to also return the raw output from the metrics.
        """
        # Initialize separate metrics for level vars and surface vars.
        kwargs = {}
        if surface_variables:
            surface_ensemble_metric = EnsembleMetrics(data_shape=(len(surface_variables), 1))
            # Wrap metric with LabelDictWrapper to return dictionary with labelled metrics (for wandb logging).
            kwargs["surface"] = LabelDictWrapper(
                surface_ensemble_metric,
                variable_indices=add_timedelta_index(
                    era5.get_surface_variable_indices(surface_variables),
                    lead_time_hours=lead_time_hours,
                    rollout_iterations=rollout_iterations,
                ),
                return_raw_dict=return_raw_dict,
            )
        if level_variables:
            level_ensemble_metric = EnsembleMetrics(
                data_shape=(len(level_variables), len(pressure_levels)),
                save_memory=save_memory,  # Save memory on level vars only.
            )
            kwargs["level"] = LabelDictWrapper(
                level_ensemble_metric,
                variable_indices=add_timedelta_index(
                    era5.get_headline_level_variable_indices(pressure_levels, level_variables),
                    lead_time_hours=lead_time_hours,
                    rollout_iterations=rollout_iterations,
                ),
                return_raw_dict=return_raw_dict,
            )
        super().__init__(**kwargs)
