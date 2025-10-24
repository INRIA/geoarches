import importlib
from pathlib import Path
from typing import Dict, List

import torch
import xarray as xr
from tensordict.tensordict import TensorDict

import geoarches.stats as geoarches_stats
from geoarches.dataloaders.era5_constants import (
    arches_default_level_variables,
    arches_default_pressure_levels,
    arches_default_surface_variables,
)
from geoarches.metrics.metric_base import compute_lat_weights, compute_lat_weights_weatherbench

# Stats path
geoarches_stats_path = importlib.resources.files(geoarches_stats)

# Default loss weights used for ArchesWeather
default_var_weights = {
    "surface": [0.1, 0.1, 1.0, 0.1],
    "level": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}


class NormalizationStatistics:
    def __init__(
        self,
        norm_file: str = "pangu_norm_stats.nc",
        variables: Dict[str, List[str]] = None,
        levels: List[int] = arches_default_pressure_levels,
        loss_weight_per_variable: Dict[str, List[float]] = default_var_weights,
        diff_delta_stats_path: str | None = None,
    ):
        """
        Initializes the normalization module with the specified normalization scheme, variables,
        pressure levels, and loss weights per variable.

        The module supports two normalization schemes: 'graphcast' and 'pangu'.
        Pangu normalization is the scheme used for the models presented in the papers.
        Graphcast normalization scheme allows the use of more variables and pressure levels,
        but it is not the default scheme used in the ArchesWeather models.

        The normalization module will load the mean and standard deviation statistics
        for the specified variables and pressure levels from the precomputed stats files.
        Further, the modules computes the loss coefficients based on the provided variables,
        pressure levels, and loss weights per variable.
        The loss coefficients are computed based on the area weights, surface and level variables,
        and the vertical coefficients derived from the pressure levels.
        The normalization module also supports delta normalization, which is used to normalize
        the loss coefficients based on the standard deviation of the difference between successive states.

        Parameters
        ----------
        norm_file : str
            The path to the normalization statistics file. Can either be a relative path with respect to
            geoarches/stats, or an absolute path.
        variables : dict, optional
            A dictionary containing the variables to be normalized. The keys should be 'surface' and 'level',
            and the values should be lists of variable names. If None, the default surface and level variables
            of archesweather will be used.
        levels : list, optional
            A list of pressure levels to be used for normalization. If None, the default pressure levels
            of archesweather will be used.
        loss_weight_per_variable : dict, optional
            A dictionary containing the loss weights for each variable. The keys should be 'surface' and 'level',
            and the values should be lists of corresponding weights (in the same order as the variable lists).
            If None, the default weights defined in `default_var_weights` will be used.
        """
        if variables is None:
            variables = {
                "surface": arches_default_surface_variables,
                "level": arches_default_level_variables,
            }
        print("##### VARIABLES: ", variables, " #####")
        print("##### LEVELS: ", levels, " #####")

        self.norm_file_path = norm_file
        if not Path(norm_file).is_absolute():
            self.norm_file_path = geoarches_stats_path / norm_file
        if not Path(self.norm_file_path).exists():
            raise ValueError(f"Normalization file {self.norm_file_path} does not exist.")

        print(self.norm_file_path)
        self.diff_delta_stats_path = diff_delta_stats_path

        # If passed through hydra, need to convert from OmegaConf objects to lists.
        self.variables = {k: list(vars) for k, vars in variables.items()}
        self.levels = list(levels)

        self.loss_weight_per_variable = loss_weight_per_variable

        self.mean = None
        self.std = None
        self.diff_std = None
        self.loss_coeffs = None

    def load_normalization_stats(self):
        with xr.open_dataset(self.norm_file_path) as stats_ds:
            stats = {
                "surface_mean": torch.from_numpy(
                    stats_ds[self.variables["surface"]].sel(statistic="mean").to_array().to_numpy()
                )[..., None, None, None],
                "surface_std": torch.from_numpy(
                    stats_ds[self.variables["surface"]].sel(statistic="std").to_array().to_numpy()
                )[..., None, None, None],
                "level_mean": torch.from_numpy(
                    stats_ds[self.variables["level"]]
                    .sel(statistic="mean")
                    .sel(level=self.levels)
                    .to_array()
                    .to_numpy()
                )[..., None, None],
                "level_std": torch.from_numpy(
                    stats_ds[self.variables["level"]]
                    .sel(statistic="std")
                    .sel(level=self.levels)
                    .to_array()
                    .to_numpy()
                )[..., None, None],
            }

        self.mean = TensorDict(
            surface=stats["surface_mean"],
            level=stats["level_mean"],
        )
        self.std = TensorDict(
            surface=stats["surface_std"],
            level=stats["level_std"],
        )

        return self.mean, self.std

    def load_timedelta_stats(self):
        """Loads the standard deviation of the difference between successive states."""

        path = (
            self.diff_delta_stats_path or self.norm_file_path
        )  # Default to norm file if no delta path provided
        with xr.open_dataset(path) as stats_ds:
            surface_stds = torch.from_numpy(
                stats_ds[self.variables["surface"]].sel(statistic="diff_std").to_array().to_numpy()
            )[..., None, None, None]
            level_stds = torch.from_numpy(
                stats_ds[self.variables["level"]]
                .sel(statistic="diff_std")
                .sel(level=self.levels)
                .to_array()
                .to_numpy()
            )[..., None, None]

            return surface_stds, level_stds

    def compute_loss_coeffs(
        self,
        latitude=121,
        pow=2,
        loss_delta_normalization=True,
        use_weatherbench_lat_coeffs=False,
        area_weights=None,
    ):
        if area_weights is None:
            compute_weights_fn = (
                compute_lat_weights_weatherbench
                if use_weatherbench_lat_coeffs
                else compute_lat_weights
            )

            area_weights = compute_weights_fn(latitude)

        pressure_levels = torch.tensor(self.levels).float()
        vertical_coeffs = (pressure_levels / pressure_levels.mean()).reshape(-1, 1, 1)

        n_surface_vars = len(self.variables["surface"])
        n_level_vars = len(self.variables["level"])

        surf_weights = torch.tensor([self.loss_weight_per_variable["surface"]]).reshape(
            -1, 1, 1, 1
        )
        level_weights = torch.tensor([self.loss_weight_per_variable["level"]]).reshape(-1, 1, 1, 1)

        total_coeff = sum(surf_weights) + sum(level_weights)

        surface_coeffs = n_surface_vars * surf_weights
        level_coeffs = n_level_vars * level_weights

        loss_coeffs = TensorDict(
            surface=area_weights * surface_coeffs / total_coeff,
            level=area_weights * level_coeffs * vertical_coeffs / total_coeff,
        )

        # Get standard deviation for normalization
        if self.std is not None:
            data_std = self.std
        else:
            _, data_std = self.load_normalization_stats()
            self.std = data_std

        if loss_delta_normalization:
            if self.diff_std is not None:
                delta_surface_stds, delta_level_stds = self.diff_std
            else:
                delta_surface_stds, delta_level_stds = self.load_timedelta_stats()
                self.diff_std = (delta_surface_stds, delta_level_stds)

            if data_std["surface"].shape[0] != delta_surface_stds.shape[0]:
                raise ValueError(
                    f"Surface stds shape mismatch: {data_std['surface'].shape} vs {delta_surface_stds.shape}"
                )
            if data_std["level"].shape[0] != delta_level_stds.shape[0]:
                raise ValueError(
                    f"Level stds shape mismatch: {data_std['level'].shape} vs {delta_level_stds.shape}"
                )

            loss_delta_scaler = TensorDict(
                surface=data_std["surface"] / delta_surface_stds,
                level=data_std["level"] / delta_level_stds,
            )
        else:
            loss_delta_scaler = TensorDict(
                surface=torch.ones_like(loss_coeffs["surface"]),
                level=torch.ones_like(loss_coeffs["level"]),
            )

        self.loss_coeffs = loss_coeffs

        return loss_coeffs, loss_delta_scaler
