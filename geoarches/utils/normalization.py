import importlib
from typing import Dict, List

import torch
import xarray as xr
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from tensordict import TensorDict

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
        variables: Dict[str, List[str]] = None,
        levels: List[int] = arches_default_pressure_levels,
        norm_scheme: str = "pangu",
        loss_weight_per_variable: Dict[str, Dict[str, float]] = default_var_weights,
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
        variables : dict, optional
            A dictionary containing the variables to be normalized. The keys should be 'surface' and 'level',
            and the values should be lists of variable names. If None, the default surface and level variables
            of archesweather will be used.
        levels : list, optional
            A list of pressure levels to be used for normalization. If None, the default pressure levels
            of archesweather will be used.
        norm_scheme : str, optional
            The normalization scheme to be used. It can be either 'graphcast' or 'pangu'.
            If None, no normalization will be applied.
        loss_weight_per_variable : dict, optional
            A dictionary containing the loss weights for each variable. The keys should be 'surface' and 'level',
            and the values should be dictionaries with variable names as keys and their corresponding weights as values.
            If None, the default weights defined in `default_var_weights` will be used.
        Raises
        ------
        ValueError
            If the provided normalization scheme is not supported. Supported schemes are 'graphcast' and '
            'pangu'.
        AssertionError
            If the normalization scheme is 'pangu' and the provided variables or levels do not match
            the default values required for this scheme.
        Notes
        -----
        The default surface variables for the 'pangu' normalization scheme are:
        - 10m_u_component_of_wind
        - 10m_v_component_of_wind
        - 2m_temperature
        - mean_sea_level_pressure
        The default level variables for the 'pangu' normalization scheme are:
        - geopotential
        - u_component_of_wind
        - v_component_of_wind
        - temperature
        - specific_humidity
        - vertical_velocity
        The default pressure levels for the 'pangu' normalization scheme are:
        - 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
        """

        print("##### NORM SCHEME: ", norm_scheme, " #####")

        if variables is None:
            variables = {
                "surface": arches_default_surface_variables,
                "level": arches_default_level_variables,
            }
        print("##### VARIABLES: ", variables, " #####")

        if norm_scheme and norm_scheme not in ["graphcast", "pangu"]:
            raise ValueError(
                f"Normalization scheme {norm_scheme} not supported. Choose from ['graphcast', 'pangu']"
            )
        self.norm_scheme = norm_scheme

        self.variables = (
            OmegaConf.to_object(variables) if isinstance(variables, DictConfig) else variables
        )
        self.levels = OmegaConf.to_object(levels) if isinstance(levels, ListConfig) else levels

        if norm_scheme == "pangu":
            assert self.variables["surface"] == arches_default_surface_variables, (
                "Pangu normalization scheme requires the default surface variables./n"
                "Surf. Vars: 10m_u_component_of_wind, 10m_v_component_of_wind, 2m_temperature, "
                "mean_sea_level_pressure"
            )
            assert self.variables["level"] == arches_default_level_variables, (
                "Pangu normalization scheme requires the default level variables./n"
                "Level Vars: geopotential, u_component_of_wind, v_component_of_wind, "
                "temperature, specific_humidity, vertical_velocity"
            )
            assert self.levels == arches_default_pressure_levels, (
                "Pangu normalization scheme requires the default pressure levels./n"
                "Pressure Levels: 50, 100, 150, 200, 250, 300, 400, "
                "500, 600, 700, 850, 925, 1000"
            )

        self.loss_weight_per_variable = loss_weight_per_variable

        self.mean = None
        self.std = None
        self.loss_coeffs = None

    def _graphcast_normalization_stats(self):
        norm_file_path = geoarches_stats_path / "graphcast_norm_stats.nc"
        stats_ds = xr.open_dataset(norm_file_path)

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

        data_mean = TensorDict(
            surface=stats["surface_mean"],
            level=stats["level_mean"],
        )

        data_std = TensorDict(
            surface=stats["surface_std"],
            level=stats["level_std"],
        )

        return data_mean, data_std

    def _pangu_normalization_stats(self):
        # include vertical component by default
        norm_file_path = geoarches_stats_path / "pangu_norm_stats2_with_w.pt"
        pangu_stats = torch.load(norm_file_path, weights_only=True)

        data_mean = TensorDict(
            surface=pangu_stats["surface_mean"],
            level=pangu_stats["level_mean"],
        )

        data_std = TensorDict(
            surface=pangu_stats["surface_std"],
            level=pangu_stats["level_std"],
        )

        return data_mean, data_std

    def load_normalization_stats(self):
        if self.norm_scheme is None:
            return None, None

        if self.norm_scheme == "pangu":
            mean, std = self._pangu_normalization_stats()
        elif self.norm_scheme == "graphcast":
            mean, std = self._graphcast_normalization_stats()

        self.mean = mean
        self.std = std

        return mean, std

    def load_graphcast_timedelta_stats(self):
        """Loads the standard deviation of the difference between successive states."""
        file = geoarches_stats_path / "graphcast_diffs_stats.nc"
        stats_ds = xr.open_dataset(file)

        surface_stds = torch.from_numpy(
            stats_ds[self.variables["surface"]].sel(statistic="std").to_array().to_numpy()
        )[..., None, None, None]
        level_stds = torch.from_numpy(
            stats_ds[self.variables["level"]]
            .sel(statistic="std")
            .sel(level=self.levels)
            .to_array()
            .to_numpy()
        )[..., None, None]

        return surface_stds, level_stds

    def compute_loss_coeffs(
        self, latitude=121, pow=2, loss_delta_normalization=True, use_weatherbench_lat_coeffs=False
    ):
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

        if loss_delta_normalization:
            if self.norm_scheme == "graphcast":
                delta_surface_stds, delta_level_stds = self.load_graphcast_timedelta_stats()
            else:
                # For Pangu, we use the precomputed stats
                delta_surface_stds = torch.tensor([3.8920, 4.5422, 2.0727, 584.0980]).reshape(
                    -1, 1, 1, 1
                )
                delta_level_stds = torch.tensor(
                    [5.9786e02, 7.4878e00, 8.9492e00, 2.7132e00, 9.5222e-04, 0.3]
                ).reshape(-1, 1, 1, 1)

            assert data_std["surface"].shape[0] == delta_surface_stds.shape[0], (
                "Surface stds shape mismatch"
            )
            assert data_std["level"].shape[0] == delta_level_stds.shape[0], (
                "Level stds shape mismatch"
            )

            loss_delta_scaler = TensorDict(
                surface=data_std["surface"] / delta_surface_stds,
                level=data_std["level"] / delta_level_stds,
            )

            loss_coeffs = loss_coeffs * loss_delta_scaler.pow(pow)

        print(
            f"Loss coefficients computed with normalization scheme:\
            {self.norm_scheme}, pow: {pow}, delta normalization: {loss_delta_normalization},\
            use_weatherbench_lat_coeffs: {use_weatherbench_lat_coeffs}"
        )

        self.loss_coeffs = loss_coeffs

        return loss_coeffs
