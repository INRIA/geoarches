import importlib.resources

import numpy as np
import pandas as pd
import torch
from tensordict.tensordict import TensorDict

from .. import stats as geoarches_stats
from .netcdf import XarrayDataset
from geoarches.utils.tensordict_utils import apply_nan_to_num, apply_isnan, apply_threshold, replace_nans



class DCPPForecast(XarrayDataset):
    """
    Load DCPP data for the forecast task.

    Loads previous timestep and multiple future timesteps if configured.
    Also handles normalization.
    """

    def __init__(
        self,
        path="/path/to/data/",
        forcings_path="data/",
        domain="train",
        filename_filter=None,
        lead_time_months=1,
        multistep=1,
        load_prev=True,
        load_clim=False,
        norm_scheme="spatial_norm",
        limit_examples: int = 0,
        mask_value=0,
        variables=None,
        surface_variables = None,
        level_variables = None,
        surface_variable_indices = [],
        level_variable_indices = [],
        pressure_levels = [85000, 70000, 50000, 25000],
        filename_filter_type='dcpp',
        forcing_type='dcpp',
    ):
        """
        Args:
            path: Single filepath or directory holding files.
            domain: Specify data split for the filename filters (eg. train, val, test, testz0012..)
            engine: xarray dataset backend.
            filename_filter: To filter files within data directory based on filename.
            lead_time_months: Time difference between current state and previous and future states.
            multistep: How many future states to load. By default, loads one (current time + lead_time_months).
            load_prev: Whether to load state at previous timestamp (current time - lead_time_months).
            load_clim: Whether to load climatology.
            limit_examples: Return set number of examples in dataset
            mask_value: what value to use as mask for nan values in dataset
        """
        self.__dict__.update(locals())  # concise way to update self with input arguments
        self.timedelta = 1        

        if(self.filename_filter_type =='dcpp_alt'):
            train_filter = [x for i, x in enumerate(range(1960,2000))]
            val_filter = [x for i, x in enumerate(range(2000,2010))]
            filename_filters = dict(
                all=(lambda _: True),
                train=lambda x: any(
                    substring in x for substring in [f"{str(x)}_" for x in train_filter]
                ),
                test=lambda x: any(
                    substring in x for substring in [f"{str(x)}_" for x in [2010,2011,2012,2013,2014,2015,2016]]
                ),
                val=lambda x: any(
                    substring in x for substring in [f"{str(x)}_" for x in val_filter]
                ),
                empty=lambda x: False,
            )
        elif(self.filename_filter_type =='dcpp'):
            train_filter = [x for i, x in enumerate(range(1960,2010)) if (i + 1) % 10 != 0]
            test_filter = [1969,1979,1989,1999,2009]
            filename_filters = dict(
                all=(lambda _: True),
                train=lambda x: any(
                    substring in x for substring in [f"{str(x)}_" for x in train_filter]
                ),
                val=lambda x: any(
                    substring in x for substring in [f"{str(x)}_" for x in [2010,2011,2012,2013,2014,2015,2016]]
                ),
                test=lambda x: any(
                    substring in x for substring in [f"{str(x)}_" for x in test_filter]
                ),
                empty=lambda x: False,
            )            
        elif(self.filename_filter_type=='cmip'):
            train_filter = ['historical','ssp245']
            test_filter = ['ssp370']
            val_filter = ['ssp245'] #for accessibility 
            filename_filters = dict(
                all=(lambda _: True),
                train=lambda x: any(
                    substring in x for substring in [f"{str(x)}.nc" for x in train_filter]
                ),
                val=lambda x: any(
                    substring in x for substring in [f"{str(x)}.nc" for x in val_filter]
                ),
                test=lambda x: any(
                    substring in x for substring in [f"{str(x)}.nc" for x in test_filter]
                ),
                empty=lambda x: False,
            )
        if filename_filter is None:
            filename_filter = filename_filters[domain]
        if variables is None:
            variables = dict(surface=surface_variables, level=level_variables)
        if(self.filename_filter_type=='cmip'):
            stats_file_path = 'cmip_stats.pt'
            dimension_indexers = {"plev": None}

        else:
            stats_file_path = 'dcpp_stats.pt'
            dimension_indexers = {"plev": pressure_levels}
        super().__init__(
            path,
            filename_filter=filename_filter,
            variables=variables,
            limit_examples=limit_examples,
            dimension_indexers=dimension_indexers,
        )



        geoarches_stats_path = importlib.resources.files(geoarches_stats)
        norm_file_path = geoarches_stats_path / stats_file_path
        level_norm_file_path = geoarches_stats_path / stats_file_path

        spatial_norm_stats = torch.load(norm_file_path)
        level_spatial_norm_stats = torch.load(level_norm_file_path)
        
        clim_removed_file_path = geoarches_stats_path / "dcpp_clim_removed_norm_stats.pt"
        clim_removed_norm_stats = torch.load(clim_removed_file_path)
        
        if self.norm_scheme is None:
            self.data_mean = TensorDict(
                surface=torch.tensor(0),
                level=torch.tensor(0),
            )
            self.data_std = TensorDict(
                surface=torch.tensor(1),
                level=torch.tensor(1),
            )
        elif self.norm_scheme == "spatial_norm":
            self.data_mean = TensorDict(
                surface=torch.stack([spatial_norm_stats["surface_mean"][i] for i in surface_variable_indices]),
                level=torch.stack([level_spatial_norm_stats["level_mean"][i] for i in level_variable_indices]),
            )
            self.data_std = TensorDict(
                surface=torch.stack([spatial_norm_stats["surface_std"][i] for i in surface_variable_indices]),
                level=torch.stack([level_spatial_norm_stats["level_std"][i] for i in level_variable_indices]),
            )
        elif self.norm_scheme == "mean_only_spatial_norm":
            self.data_mean = TensorDict(
                surface=torch.stack([spatial_norm_stats["surface_mean"][i] for i in surface_variable_indices]),
                level=torch.stack([level_spatial_norm_stats["level_mean"][i] for i in level_variable_indices]),
            )

            self.data_std = TensorDict(
                surface=torch.stack([spatial_norm_stats["surface_std"][i] for i in surface_variable_indices]).nanmean(axis=(-1,-2),keepdim=True),
                level=torch.stack([level_spatial_norm_stats["level_std"][i] for i in level_variable_indices]).nanmean(axis=(-1,-2),keepdim=True),
            )
        elif self.norm_scheme == "clim_removed":
            self.data_mean = TensorDict(
                surface=clim_removed_norm_stats["surface_mean"],
                level=clim_removed_norm_stats["level_mean"],
            )
            self.data_mean.batch_size = [12] #this is so the tensordict can be indexed by one value for both surface/level
            self.data_std = TensorDict(
                surface=clim_removed_norm_stats["surface_std"],
                level=clim_removed_norm_stats["level_std"],
            )
            self.data_std.batch_size = [12]

        self.surface_variables = surface_variables
        self.level_variables = [
            a + ' ' + str(p//100)
            for a in level_variables
            for p in pressure_levels
        ]

        self.atmos_forcings = torch.tensor(np.load(f'{forcings_path}/ghg_forcings_normed.npy'))
        self.solar_forcings = torch.tensor(np.load(f'{forcings_path}/solar_forcings_normed.npy'))
        times_seconds = [v[2].item() // 10**9 for k,v in self.id2pt.items()]
        self.next_timestamp_map = {k:v for k,v in list(zip(times_seconds,times_seconds[1:]))}

        #override netcdf functionality
        self.timestamps = sorted(self.timestamps, key=lambda x: (x[0],x[1]))  # sort by timestamp

    def convert_to_tensordict(self, xr_dataset):
        """
        input xarr should be a single time slice
        """
        tdict = super().convert_to_tensordict(xr_dataset)
        tdict["surface"] = tdict["surface"].unsqueeze(-3)
        return tdict

    def __len__(self):
        # Take into account previous and/or future timestamps loaded for one example.
        offset = self.multistep + self.load_prev
        return super().__len__() - offset * self.lead_time_months // self.timedelta

    def __getitem__(self, i, normalize=True):
        out = {}
        # Shift index forward if need to load previous timestamp.
        i = i + self.load_prev * self.lead_time_months // self.timedelta

        out = dict()
        #  load current state
        out["state"] = super().__getitem__(i)
        out["timestamp"] = torch.tensor(
            self.id2pt[i][2].item() // 10**9,
            dtype=torch.int64,
        )  # time in seconds
        times = pd.to_datetime(out["timestamp"].cpu().numpy(),unit='s').tz_localize(None)
        current_month = torch.tensor(times.month)-1 % 12
        current_year = torch.tensor(times.year) - 1961 # -1961 to zero index
        out['forcings'] = torch.concatenate([self.atmos_forcings[current_year,:], self.solar_forcings[(current_year*12)+current_month,:]])

        t = self.lead_time_months  # multistep

        out["next_state"] = super().__getitem__(i + t // self.timedelta)
        # Load multiple future timestamps if specified.
        if self.multistep > 1:
            future_states = []
            for k in range(1, self.multistep + 1):
                future_states.append(super().__getitem__(i + k * t // self.timedelta))
            out["future_states"] = torch.stack(future_states, dim=0)

        if self.load_prev:
            if(self.load_prev > 1):
                prev_states = []
                for k in range(0,self.load_prev):
                    prev_states.append(super().__getitem__(i - (self.lead_time_months*k+1) // self.timedelta))
                out["prev_state"] = torch.stack(prev_states, dim=0)
            else:
                out["prev_state"] = super().__getitem__(i - self.lead_time_months // self.timedelta)
                prev_timestamp = self.id2pt[i - self.lead_time_months // self.timedelta][2].item() // 10**9
                times = pd.to_datetime(prev_timestamp,unit='s').tz_localize(None)
        if normalize:
            out = self.normalize(out,month=current_month)
    
        # need to replace nans with mask_value
        mask = {k: (apply_isnan(v) if "state" in k else v) for k, v in out.items()}

        out = {k: apply_nan_to_num(v) if "state" in k else v for k, v in out.items()}
        out = {k: (v*mask[k] if "state" in k else v) for k, v in out.items()}

        return out

    def normalize(self, batch,stateless=False,month=None):
        device = list(batch.values())[0].device
        means = self.data_mean.to(device)
        stds = self.data_std.to(device)

        if(self.norm_scheme == 'clim_removed'):
            return {k: ((v - means[month]) / stds[month] if "state" in k else v) for k, v in batch.items()}
        elif(stateless):
            return (batch - means) / stds
        else:
            dict_out =  {k: ((v - means) / stds if "state" in k else v) for k, v in batch.items()}
            dict_out = {k: apply_threshold(v) if "state" in k else v for k, v in dict_out.items()}
            return dict_out

    def denormalize(self, batch,stateless=False,month=None):
        if(stateless):
            device = batch.device
        else:
            device = list(batch.values())[0].device
        means = self.data_mean.to(device)
        stds = self.data_std.to(device)
        if(self.norm_scheme == 'clim_removed' and stateless):
            return batch * stds[month] + means[month]
        elif(self.norm_scheme == 'clim_removed'):
            return {k: (v * stds[month] + means[month] if "state" in k else v) for k, v in batch.items()}
        elif(stateless):
            return (batch * stds) + means
        else:
            return {k: ((v * stds) + means if "state" in k else v) for k, v in batch.items()}

