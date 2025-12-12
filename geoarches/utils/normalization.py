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
        levels: List[int] | None = arches_default_pressure_levels,
        residual_stats_path: str | None = None,
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
            A list of pressure levels to be used for normalization. If None, assumes stats are not by levels.
        residual_stats_path : str, optional
            The path to the statistics file containing the standard deviation of the difference between predicted
            successive states. If None, the normalization of training data (e.g. ERA5) file will be used.
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

        self.residual_stats_path = residual_stats_path
        if self.residual_stats_path is not None:
            if not Path(self.residual_stats_path).is_absolute():
                self.residual_stats_path = geoarches_stats_path / self.residual_stats_path
            if not Path(self.residual_stats_path).exists():
                raise ValueError(f"Residual stats file {self.residual_stats_path} does not exist.")

        # If passed through hydra, need to convert from OmegaConf objects to lists.
        self.variables = {k: list(vars) for k, vars in variables.items()}
        self.levels = list(levels) if levels else None

        self.mean = None
        self.std = None
        self.diff_std = None
        self.loss_coeffs = None
        self.state_scaler = None

    def load_normalization_stats(self):
        """
        Loads the mean and standard deviation statistics for the specified variables and pressure levels
        from the precomputed stats file.
        """
        mean, std = {}, {}
        with xr.open_dataset(self.norm_file_path) as stats_ds:
            for var_type, var_names in self.variables.items():
                # Select by levels if level dimension is present.
                indexers = {}
                has_lev = True if "level" in stats_ds[var_names].dims else False
                if has_lev:
                    if self.levels is None:
                        raise ValueError(
                            "Found level dimension in dataset, but levels are not provided."
                        )
                    indexers = {"level": self.levels}
                mean[var_type] = torch.from_numpy(
                    stats_ds[var_names].sel(statistic="mean", **indexers).to_array().to_numpy()
                )[..., None, None]
                std[var_type] = torch.from_numpy(
                    stats_ds[var_names].sel(statistic="std", **indexers).to_array().to_numpy()
                )[..., None, None]
                # Add level dimension if not present.
                if not has_lev:
                    mean[var_type] = mean[var_type][..., None]
                    std[var_type] = std[var_type][..., None]

        self.mean = TensorDict(mean)
        self.std = TensorDict(std)

        return self.mean, self.std

    def load_timedelta_stats(self):
        """
        Loads the standard deviation of the difference between successive states.
        Depending on self.residual_stats_path, it loads either the delta stats of the model predictions
        or the delta stats of the training data (e.g., ERA5).
        """

        path = (
            self.residual_stats_path or self.norm_file_path
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

    def compute_state_scaler(
        self, state_normalization="delta", downweight_vertical_velocity=False
    ):
        """
        Computes the delta scalers for normalizing the model predictions based on the specified
        state normalization scheme.
        Parameters
        ----------
        state_normalization : str
            The state normalization scheme to be used. Can be either 'delta' or 'pred'.
            'delta' normalization uses the standard deviation of the difference between successive states.
            'pred' normalization uses the standard deviation of deltas of the model predictions.
        downweight_vertical_velocity : bool
            Whether to downweight the vertical velocity variable in the level variables.
            This is useful when the vertical velocity is not as important as other variables.
        Returns
        -------
        delta_scaler : TensorDict
            A TensorDict containing the delta scalers for surface and level variables.
        """

        # Handle assertions for state normalization
        if state_normalization == "pred":
            assert self.residual_stats_path is not None, (
                "residual_stats_path must be provided for 'pred' normalization."
            )
        elif state_normalization == "delta":
            assert self.residual_stats_path is None, (
                "residual_stats_path must be None for 'delta' normalization."
            )
        else:
            raise ValueError(
                f"Invalid state_normalization: {state_normalization}. Must be 'delta' or 'pred'."
            )

        # Load delta stats
        delta_surface_stds, delta_level_stds = self.load_timedelta_stats()

        # Get standard deviation for as we scale with state data std
        if self.std is not None:
            data_std = self.std
        else:
            _, data_std = self.load_normalization_stats()
            self.std = data_std

        # Check shapes
        if data_std["surface"].shape[0] != delta_surface_stds.shape[0]:
            raise ValueError(
                f"Surface stds shape mismatch: {data_std['surface'].shape} vs {delta_surface_stds.shape}"
            )
        if data_std["level"].shape[0] != delta_level_stds.shape[0]:
            raise ValueError(
                f"Level stds shape mismatch: {data_std['level'].shape} vs {delta_level_stds.shape}"
            )

        # Compute delta scalers
        state_scaler = TensorDict(
            surface=data_std["surface"] / delta_surface_stds,
            level=data_std["level"] / delta_level_stds,
        )

        # Downweight vertical velocity if specified
        if downweight_vertical_velocity:
            # Downweight vertical velocity (assumed to be the last level variable)
            vv_index = self.variables["level"].index("vertical_velocity")
            state_scaler["level"][vv_index] *= 0.3  # Downweight by a factor of 3

        self.state_scaler = state_scaler

        return state_scaler

    def compute_loss_coeffs(
        self,
        loss_weight_per_variable: Dict[str, List[float]] = default_var_weights,
        latitude: int = 121,
        use_weatherbench_lat_coeffs: bool = False,
    ):
        """Computes loss coefficients per variable, level, and latitude.

        Args:
            loss_weight_per_variable : A dictionary containing the loss weights for each variable.
                The keys match keys of the variables, i.e. 'surface' and 'level',
                and the values should be lists of corresponding weights (in the same order as the variable lists).
                If None, the default weights defined in `default_var_weights` will be used.
            latitude : Number of latitude points in the data (Needed to compute latitude weighting).
            use_weatherbench_lat_coeffs : bool
                Whether to use the WeatherBench latitude coefficients for area weighting.
        """
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

        surf_weights = torch.tensor([loss_weight_per_variable["surface"]]).reshape(-1, 1, 1, 1)
        level_weights = torch.tensor([loss_weight_per_variable["level"]]).reshape(-1, 1, 1, 1)

        total_coeff = sum(surf_weights) + sum(level_weights)

        surface_coeffs = n_surface_vars * surf_weights
        level_coeffs = n_level_vars * level_weights

        loss_coeffs = TensorDict(
            surface=area_weights * surface_coeffs / total_coeff,
            level=area_weights * level_coeffs * vertical_coeffs / total_coeff,
        )

        self.loss_coeffs = loss_coeffs

        return loss_coeffs
