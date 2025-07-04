import json 
import torch 

import importlib
import geoarches.stats as geoarches_stats
from tensordict import TensorDict

geoarches_stats_path = importlib.resources.files(geoarches_stats)

from geoarches.dataloaders.era5 import arches_default_surface_variables, arches_default_level_variables, arches_default_pressure_levels, pressure_levels
from geoarches.metrics.metric_base import compute_lat_weights, compute_lat_weights_weatherbench

default_var_weights = {
    "surface": {
        "T2m": 1.0,
        "U10m": .1,
        "V10m": .1,
        "SP": .1,
    },
    "level": {
        "Z": 1.0,
        "U": 1.0,
        "V": 1.0,
        "T": 1.0,
        "Q": 1.0,
        "W": 1.0,
    },
}

class TensorDictNormalizationStats:
    def __init__(self, variables=None, levels=None, norm_scheme=None, loss_weight_per_variable=None):
        print("##### NORM SCHEME: ", norm_scheme, " #####")
        
        if variables is None:
            variables = {
                "surface": arches_default_surface_variables,
                "level": arches_default_level_variables,
            }

        print("##### VARIABLES: ", variables, " #####"  )

        if levels is None: 
            levels = arches_default_pressure_levels

        if norm_scheme is None:
            self.norm_scheme = "pangu"
        elif norm_scheme not in ["graphcast", "pangu"]:
            raise ValueError(f"Normalization scheme {norm_scheme} not supported. Choose from ['graphcast', 'pangu']")
        else:
            self.norm_scheme = norm_scheme

        self.variables = variables
        self.levels = levels
        self.pl_indices = [pressure_levels.index(p) for p in self.levels]
        
        if loss_weight_per_variable is not None:
            self.loss_weight_per_variable = loss_weight_per_variable
        else:
            self.loss_weight_per_variable = default_var_weights

        self.mean = None
        self.std = None
        self.loss_coeffs = None

    def _graphcast_normalization_stats(self):

        with open(geoarches_stats_path / "gc_stats_mean_by_level.json") as f:
            mean_stats = json.load(f)
        with open(geoarches_stats_path / "gc_stats_stddev_by_level.json") as f:
            std_stats = json.load(f)

        data_mean = {
            "surface": torch.tensor([mean_stats['data_vars'][v]['data'] for v in self.variables["surface"]]).reshape(-1, 1, 1, 1),
            "level": torch.tensor([[mean_stats['data_vars'][v]['data'][j] for j in self.pl_indices] for v in self.variables["level"]]).reshape(-1, len(self.levels), 1, 1),
        }

        data_std = {
            "surface": torch.tensor([[std_stats['data_vars'][v]['data']] for v in self.variables["surface"]]).reshape(-1, 1, 1, 1),
            "level": torch.tensor([[std_stats['data_vars'][v]['data'][j] for j in self.pl_indices] for v in self.variables["level"]]).reshape(-1, len(self.levels), 1, 1),
        }

        return TensorDict(surface=data_mean["surface"], level=data_mean["level"]), \
            TensorDict(surface=data_std["surface"], level=data_std["level"])

    def _pangu_normalization_stats(cls):
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
        if self.norm_scheme == "pangu":
            mean, std = self._pangu_normalization_stats()
        elif self.norm_scheme == "graphcast":
            mean, std = self._graphcast_normalization_stats()

        self.mean = mean
        self.std = std

        return mean, std

    def load_graphcast_timedelta_stats(self):
        file = geoarches_stats_path / "gc_stats_diffs_stddev_by_level.json"
        with open(file) as f:
            diff_stats = json.load(f)
        
        surface_stds = torch.tensor([diff_stats['data_vars'][v]['data'] for v in self.variables['surface']]).reshape(-1, 1, 1, 1)
        level_stds = torch.tensor([[diff_stats['data_vars'][v]['data'][i] for i in self.pl_indices] for v in self.variables['level']]).reshape(-1, len(self.levels), 1, 1)

        return surface_stds, level_stds 

    def compute_loss_coeffs(self, latitude=121, pow=2, loss_delta_normalization=True, use_weatherbench_lat_coeffs=False):
        
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

        surf_weights =  torch.tensor([self.loss_weight_per_variable["surface"]]).reshape(
                -1, 1, 1, 1)
        level_weights = torch.tensor([self.loss_weight_per_variable["level"]]).reshape(
                -1, 1, 1, 1)
        
        total_coeff = sum(surf_weights) + sum(level_weights)

        surface_coeffs = n_surface_vars * surf_weights 
        level_coeffs = n_level_vars * level_weights
        
        loss_coeffs = TensorDict(
            surface=area_weights * surface_coeffs/ total_coeff,
            level=area_weights * level_coeffs * vertical_coeffs / total_coeff,
        )

        # Get standard deviation for normalization
        if self.mean is not None or self.std is not None:
            data_std = self.std
        elif self.norm_scheme == "graphcast":
            _, data_std = self.load_normalization_stats()
        elif self.norm_scheme == "pangu":
            _, data_std = self.load_normalization_stats()
        else:
            raise ValueError(f"Normalization scheme {self.norm_scheme} not supported. Choose from ['graphcast', 'pangu']")
                
        if loss_delta_normalization:
            if self.norm_scheme == "graphcast":
                delta_surface_stds, delta_level_stds = self.load_graphcast_timedelta_stats()
            else:
                # For Pangu, we use the precomputed stats
                delta_surface_stds = torch.tensor([3.8920, 4.5422, 2.0727, 584.0980]).reshape(-1, 1, 1, 1)
                delta_level_stds = torch.tensor(
                        [5.9786e02, 7.4878e00, 8.9492e00, 2.7132e00, 9.5222e-04, 0.3]
                ).reshape(-1, 1, 1, 1)

            assert data_std['surface'].shape[0] == delta_surface_stds.shape[0], "Surface stds shape mismatch"
            assert data_std['level'].shape[0] == delta_level_stds.shape[0], "Level stds shape mismatch"

            loss_delta_scaler = TensorDict(
                surface=data_std['surface'] / delta_surface_stds,
                level=data_std['level'] / delta_level_stds,
            )

            loss_coeffs = loss_coeffs * loss_delta_scaler.pow(pow)
        
        print(
            f"Loss coefficients computed with normalization scheme:\
            {self.norm_scheme}, pow: {pow}, delta normalization: {loss_delta_normalization},\
            use_weatherbench_lat_coeffs: {use_weatherbench_lat_coeffs}")
        
        self.loss_coeffs = loss_coeffs

        return loss_coeffs

