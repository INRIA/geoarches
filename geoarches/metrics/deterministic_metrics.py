from typing import Callable, Dict

import torch
from torchmetrics import Metric

from geoarches.dataloaders import era5
from geoarches.metrics.label_wrapper import LabelDictWrapper, add_timedelta_index

from .metric_base import MetricBase, TensorDictMetricBase, compute_lat_weights_weatherbench

lat_coeffs_equi = torch.tensor(
    [torch.cos(x) for x in torch.arange(-torch.pi / 2, torch.pi / 2 + 1e-6, torch.pi / 120)]
)
lat_coeffs_equi = (lat_coeffs_equi / lat_coeffs_equi.mean())[None, None, :, None]


def wrmse(pred, gt, weights=None):
    """Weighted root mean square error.

    Expects inputs of shape: [..., lat, lon]

    Args:
        pred: predictions
        gt: targets
        weights: weights for the latitudes
    """
    if weights is None:
        weights = lat_coeffs_equi.to(pred.device)

    err = (pred - gt).pow(2).mul(weights).mean((-2, -1)).sqrt()
    return err


def headline_wrmse(pred, gt, denormalize_function=None):
    """RMSE for the top variables in WeatherBench.

    Input shape should be (batch, leadtime, var, level, lat, lon)

    Args:
        pred: TensorDict with surface and level tensors for predictions.
        batch: TensorDict with surface and level tensors for targets.
        prefix: string prefix for the keys of the surface and level tensors in `pred` and `batch`.

    """
    pred = denormalize_function(pred)
    gt = denormalize_function(gt)

    surface_wrmse = wrmse(pred["surface"], gt["surface"])
    level_wrmse = wrmse(pred["level"], gt["level"])

    metrics = dict(
        T2m=surface_wrmse[..., 2, 0],
        SP=surface_wrmse[..., 3, 0],
        U10m=surface_wrmse[..., 0, 0],
        V10m=surface_wrmse[..., 1, 0],
        Z500=level_wrmse[..., 0, 7],
        T850=level_wrmse[..., 3, 10],
        Q700=1000 * level_wrmse[..., 4, 9],
        U850=level_wrmse[..., 1, 10],
        V850=level_wrmse[..., 2, 10],
    )

    return metrics


class DeterministicRMSE(Metric, MetricBase):
    """
    Metrics for deterministic prediction

    """

    def __init__(
        self,
        data_shape: tuple,
        compute_lat_weights_fn: Callable[[int], torch.tensor] = compute_lat_weights_weatherbench,
    ):
        """
        Args:

            variable_indices: Mapping from variable name to (var, lev) index into tensor holding computed metric.
                ie. dict(T2m=(2, 0), U10=(0, 0), V10=(1, 0), SP=(3, 0)).
                Will index into metric tensor with (..., *index) to handle extra dimensions such as multistep.
            compute_lat_weights_fn: Function to compute latitude weights given latitude shape.
                Used for error and variance calculations. Expected shape of weights: [..., lat, 1].
                See function example in metric_base.MetricBase.
                Default function assumes latitudes are ordered -90 to 90.
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
        self.add_state("mse", default=torch.zeros(data_shape), dist_reduce_fx="sum")
        self.add_state(
            "rmse_before_time_avg", default=torch.zeros(data_shape), dist_reduce_fx="sum"
        )

    def update(self, targets: torch.Tensor, preds: torch.Tensor) -> None:
        """Update internal state with a batch of targets and predictions.

        Expects inputs to this function to be denormalized.

        Args:
            targets: Target tensor. Expected input shape is (batch, ..., var, level, lat, lon)
            preds: Tensor. Expected input shape is (batch, ..., var, level, lat, lon)
        Returns:
            None
        """

        self.nsamples += preds.shape[0]

        # for auto-broadcast
        self.mse = self.mse + self.wmse(targets, preds).sum(0)
        self.rmse_before_time_avg = self.rmse_before_time_avg + self.wmse(
            targets, preds
        ).sqrt().sum(0)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute final metrics utilizing internal states.
        Returns:
            Dict: mapping metric name to tensor holding computed metric.
                  holds one tensor per variable and metric pair ie. mse_wind_speed.
        """
        all_metrics = dict(
            rmse_before_time_avg=self.rmse_before_time_avg / self.nsamples,
            mse=self.mse / self.nsamples,
            rmse=(self.mse / self.nsamples).sqrt(),
        )

        # out = dict()
        # for var, index in self.indices.items():
        #     for metric_name, metric in all_metrics.items():
        #         out[f"{metric_name}_{var}"] = metric.__getitem__((..., *index)).item()

        return all_metrics


class Era5DeterministicMetrics(TensorDictMetricBase):
    """Wrapper class around EnsembleMetrics for computing over surface and level variables.

    Handles batches coming from Era5 Dataloader.

    Accepted tensor shapes:
        targets: (batch, ..., timedelta, var, level, lat, lon)
        preds: (batch, nmembers, ..., timedelta, var, level, lat, lon)

    Return dictionary of metrics reduced over batch, lat, lon.
    """

    def __init__(
        self,
        compute_lat_weights_fn: Callable[[int], torch.tensor] = compute_lat_weights_weatherbench,
        pressure_levels=era5.pressure_levels,
        num_level_variables=len(era5.level_variables),
        lead_time_hours: int = 24,
        rollout_iterations: int = 1,
    ):
        """
        Args:
            pressure_levels: pressure levels in data (used to get `variable_indices`).
            level_data_shape: (var, lev) shape for level variables.
            num_level_variables: Number of level variables (used to compute data_shape).
            rollout_iterations: Number of rollout iterations in multistep predictions.
                this option labels each timestep separately in output metric dict.
                Assumes that data shape of predictions/targets are [batch, ..., multistep, var, lev, lat, lon]


        """
        super().__init__(
            surface=LabelDictWrapper(
                DeterministicRMSE(
                    data_shape=(len(era5.surface_variables), 1),
                    compute_lat_weights_fn=compute_lat_weights_fn,
                ),
                variable_indices=add_timedelta_index(
                    era5.get_surface_variable_indices(),
                    lead_time_hours=lead_time_hours,
                    rollout_iterations=rollout_iterations,
                ),
            ),
            level=LabelDictWrapper(
                DeterministicRMSE(
                    data_shape=(num_level_variables, len(pressure_levels)),
                    compute_lat_weights_fn=compute_lat_weights_fn,
                ),
                variable_indices=add_timedelta_index(
                    era5.get_headline_level_variable_indices(pressure_levels),
                    lead_time_hours=lead_time_hours,
                    rollout_iterations=rollout_iterations,
                ),
            ),
        )
