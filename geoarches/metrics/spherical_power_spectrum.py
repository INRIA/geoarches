from datetime import timedelta
from typing import Callable

import pyshtools as pysh
import torch
from einops import rearrange
from torchmetrics import Metric

from geoarches.dataloaders import era5
from geoarches.metrics.label_wrapper import LabelXarrayWrapper

from .metric_base import TensorDictMetricBase


def _remove_south_pole_lat(arr: torch.tensor) -> torch.tensor:
    """Remove 90 S lat from data because library requires nlon = nlat*2)

    Assumes: data tensor has shape (..., lat, lon)
    """
    return arr[..., :-1, :]


class PowerSpectrum(Metric):
    """
    Calculate spherical power spectrum on both targets and preds separately.

    Accepted tensor shapes:
        targets: (batch, ..., lat, lon)
        preds: (batch, nmembers, ..., lat, lon)

    Averages over batch and members.
    """

    def __init__(
        self,
        preprocess: Callable | None = None,
        compute_target_spectrum: bool = True,
    ):
        """
        Args:
            preprocess: Takes as input targets or predictions and returns processed tensor.
            compute_target_spectrim: Whether to also compute spectrum on groundtruth.
                Turn off to save computation.
        """
        Metric.__init__(self)
        self.preprocess = preprocess
        self.compute_target_spectrum = compute_target_spectrum

        self.add_state("nsamples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("pred_spectrum", default=torch.tensor(0), dist_reduce_fx="sum")
        if self.compute_target_spectrum:
            self.add_state("target_spectrum", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, targets: torch.Tensor, preds: torch.Tensor | list[torch.Tensor]) -> None:
        """Update internal state with a batch of predictions.

        Expects inputs to this function to be denormalized.

        Args:
            targets: Target tensor. Expected input shape is (batch, ..., lat, lon)
            preds: Tensor or list of tensors holding ensemble member predictions.
                   If tensor, expected input shape is (batch, nmembers, ..., lat, lon). If list, (batch, ..., lat, lon).
        """
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=1)

        self.nsamples += preds.shape[0]

        if self.preprocess:
            targets = self.preprocess(targets)
            preds = self.preprocess(preds)

        def _compute_spectrum(grid):
            """Compute power spectrum on lat x lon grid."""
            return torch.from_numpy(pysh.SHGrid.from_array(grid).expand().spectrum())

        def _compute_spectrum_over_batch(x):
            shape = x.shape[:-2]
            x = rearrange(x, "b ... lat lon -> (b ...) lat lon")
            spectrum = torch.stack([_compute_spectrum(grid) for grid in x])
            return spectrum.reshape((*shape, spectrum.shape[-1]))

        self.pred_spectrum = self.pred_spectrum + _compute_spectrum_over_batch(preds).mean(1).sum(
            0
        )
        if self.compute_target_spectrum:
            self.target_spectrum = self.target_spectrum + _compute_spectrum_over_batch(
                targets
            ).sum(0)

    def compute(self) -> torch.Tensor:
        """Compute final metrics utliizing internal states."""
        output = dict(power_spectrum=self.pred_spectrum / self.nsamples)
        if self.compute_target_spectrum:
            output["ref_power_spectrum"] = self.target_spectrum / self.nsamples
        return output


class Era5PowerSpectrum(TensorDictMetricBase):
    """Wrapper class around PowerSpectrum for computing over surface and level variables.

    Handles batches coming from Era5 Dataloader.

    Accepted tensor shapes:
        targets: (batch, timedelta, var, level, lat, lon)
        preds: (batch, nmembers, timedelta, var, level, lat, lon)
    """

    def __init__(
        self,
        compute_target_spectrum: bool = False,
        surface_variables: str = era5.surface_variables,
        level_variables: str = era5.level_variables,
        pressure_levels: str = era5.pressure_levels,
        lead_time_hours: None | int = None,
        rollout_iterations: None | int = None,
        return_raw_dict: bool = False,
    ):
        """
        Args:
            compute_target_spectrum: Whether to also compute spectrum on groundtruth.
                Turn off to save computation.
            surface_variables: Names of level variables.
            level_variables: Names of surface variables.
            pressure_levels: pressure levels in data.
            lead_time_hours: timedelta (in hours) between prediction times.
            rollout_iterations: number of multistep rollout for predictions.
                (ie. lead time of 24 hours for 3 days, lead_time_hours=24, rollout_iterations=3)
            return_raw_dict: Whether to also return the raw output from the metrics.
        """
        # Whether to include prediction_timdelta dimension.
        if rollout_iterations:
            surface_coord_names = ["prediction_timedelta", "variable", "degree"]
            level_coord_names = ["prediction_timedelta", "variable", "level", "degree"]

            timedeltas = [timedelta((i + 1) * lead_time_hours) for i in range(rollout_iterations)]
            surface_coords = [timedeltas, surface_variables]
            level_coords = [timedeltas, level_variables, pressure_levels]
        else:
            surface_coord_names = ["variable", "degree"]
            level_coord_names = ["variable", "level", "degree"]
            surface_coords = [surface_variables]
            level_coords = [level_variables, pressure_levels]

        # Initialize separate metrics for level vars and surface vars.
        kwargs = {}
        if surface_variables:
            kwargs["surface"] = LabelXarrayWrapper(
                PowerSpectrum(preprocess=lambda x: _remove_south_pole_lat(x.squeeze(-3))),
                coord_names=surface_coord_names,
                coords=surface_coords,
                return_raw_dict=return_raw_dict,
            )
        if level_variables:
            kwargs["level"] = LabelXarrayWrapper(
                PowerSpectrum(preprocess=_remove_south_pole_lat),
                coord_names=level_coord_names,
                coords=level_coords,
                return_raw_dict=return_raw_dict,
            )
        super().__init__(**kwargs)
