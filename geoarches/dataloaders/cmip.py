import importlib.resources

import pandas as pd
import torch
from tensordict.tensordict import TensorDict

from geoarches.utils.tensordict_utils import (
    apply_nan_to_num,
    get_non_nan_mask,
    replace_inf_and_large_values,
)

from .. import stats as geoarches_stats
from .netcdf import XarrayDataset


class CMIPForecast(XarrayDataset):
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
        norm_scheme="spatial",
        limit_examples: int = 0,
        mask_value=0,
        variables=None,
        surface_variables=None,
        level_variables=None,
        add_orography=False,
        spatial_forcing_variables=[
            "methane",
            "cfc11",
            "carbon",
            "nitrous",
            "load_ASNO3M",
            "load_CSNO3M",
            "load_CINO3M",
            "load_SO4",
            "load_AIBCM",
            "load_ASBCM",
        ],
        non_spatial_forcing_variables=[
            "CO2",
            "CFC12eq",
            "CFC12eq",
            "CH4",
            "N2O",
            "ssi_0",
            "ssi_1",
            "ssi_2",
            "ssi_3",
        ],
        pressure_levels=[],
        depth_levels=[
            0.50576,
            3.8562799,
            8.092519,
            13.991038,
            22.757616,
            35.740204,
            53.850636,
            77.61116,
            108.03028,
            147.40625,
        ],
        train_filter=["historical", "ssp245", "ssp585", "ssp126", "ssp434", "piControl"],
        test_filter=["ssp370"],
        val_filter=["ssp245"],
        zg_log=False,
        full_ozone=False
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
        filename_filters = dict(
            all=(lambda _: True),
            train=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.nc" for y in train_filter]
            ),
            val=lambda x: any(
                substring in x for substring in [f"{str(y)}_interpolation.nc" for y in val_filter]
            ),
            test=lambda x: any(
                substring in x for substring in [f"{str(y)}_interpolation.nc" for y in test_filter]
            ),
            val1=lambda x: any(
                substring in x for substring in [f"{str(y)}_interpolation.nc" for y in ["ssp119"]]
            ),
            val2=lambda x: any(
                substring in x for substring in [f"{str(y)}_interpolation.nc" for y in ["ssp370"]]
            ),
            val3=lambda x: any(
                substring in x for substring in [f"{str(y)}_interpolation.nc" for y in ["ssp534-over"]]
            ),
            abrupt=lambda x: any(
                substring in x for substring in [f"{str(y)}_interpolation.nc" for y in ["abrupt-4xCO2"]]
            ),
            empty=lambda x: False,
        )
        if filename_filter is None:
            filename_filter = filename_filters[domain]
        if variables is None:
            variables = dict(
                surface=surface_variables,
                level=level_variables,
                lev=["thetao"],
                spatial_forcings=spatial_forcing_variables,
                non_spatial_forcings=non_spatial_forcing_variables,
            )

        dimension_indexers = {"plev": ("plev", pressure_levels), "lev": ("lev", depth_levels)}
        
        self.master_list_spatial_forcing_variables = [
            "methane",
            "cfc11",
            "carbon",
            "nitrous",
            "load_ASNO3M",
            "load_CSNO3M",
            "load_CINO3M",
            "load_SO4",
            "load_AIBCM",
            "load_ASBCM",
        ]
        if(self.full_ozone):
            variables['ozone'] = ['ozone']
        # else:
        #     variables['ozone'] = ["ozone_0","ozone_1","ozone_2","ozone_3","ozone_4","ozone_5"]
            # self.master_list_spatial_forcing_variables.append('ozone')
        if(not self.full_ozone):
            self.master_list_spatial_forcing_variables = self.master_list_spatial_forcing_variables + ["ozone_0",
            "ozone_1",
            "ozone_2",
            "ozone_3",
            "ozone_4",
            "ozone_5"]
        self.master_list_non_spatial_forcing_variables = [
            "ssi_0",
            "ssi_1",
            "ssi_2",
            "ssi_3",
            "ssi_4",
            "ssi_5",
            "exp_id",
        ]
        super().__init__(
            path,
            filename_filter=filename_filter,
            variables=variables,
            limit_examples=limit_examples,
            dimension_indexers=dimension_indexers,
            timestamp_key=lambda x: (x[0], x[1]),
        )
        if self.norm_scheme == 'zeroes':
            self.data_mean = TensorDict(
                surface=torch.tensor(0),
                level=torch.tensor(0),
                lev=torch.tensor(0),
                spatial_forcings=torch.tensor(0),
                non_spatial_forcings=torch.tensor(0)
            )
            self.data_std = TensorDict(
                surface=torch.tensor(1),
                level=torch.tensor(1),
                lev=torch.tensor(1),
                spatial_forcings=torch.tensor(0),
                non_spatial_forcings=torch.tensor(0)
            )
        else:
            if self.full_ozone: 
                stats_file_path = "cmip_stats_full_ozone.pt"
            elif norm_scheme == 'full_ocean':
                stats_file_path = "cmip_stats_full_ocean.pt"
            elif norm_scheme == 'zg_log':
                stats_file_path = 'cmip_stats_zg_log.pt'
            elif ("piControl" in train_filter) and (norm_scheme == 'spatial_new'):
                stats_file_path = "cmip_stats_test_welfords.pt"
            elif "piControl" in train_filter:
                stats_file_path = "cmip_stats_piControl.pt"
            else:
                stats_file_path = "cmip_stats_test_welfords_without_piControl.pt"
            geoarches_stats_path = importlib.resources.files(geoarches_stats)
            norm_file_path = geoarches_stats_path / stats_file_path
            spatial_norm_stats = torch.load(norm_file_path)
            # a hack to add exp_id without needing to normalize it.
            spatial_norm_stats["non_spatial_forcings_mean"] = torch.concat(
                [spatial_norm_stats["non_spatial_forcings_mean"], torch.tensor([0])]
            )
            spatial_norm_stats["non_spatial_forcings_std"] = torch.concat(
                [spatial_norm_stats["non_spatial_forcings_std"], torch.tensor([1])]
            )
            self.data_mean = TensorDict(
                surface=spatial_norm_stats["surface_mean"],
                level=spatial_norm_stats["level_mean"],
                lev=spatial_norm_stats["lev_mean"],
                spatial_forcings=spatial_norm_stats["spatial_forcings_mean"],
                non_spatial_forcings=spatial_norm_stats["non_spatial_forcings_mean"],
            )


            self.data_std = TensorDict(
                surface=spatial_norm_stats["surface_std"].nanmean(axis=(-1, -2), keepdim=True),
                level=spatial_norm_stats["level_std"].nanmean(axis=(-1, -2), keepdim=True),
                lev=spatial_norm_stats["lev_std"].nanmean(axis=(-1, -2), keepdim=True),
                spatial_forcings=spatial_norm_stats["spatial_forcings_std"].nanmean(
                    axis=(-1, -2), keepdim=True
                ),
                non_spatial_forcings=spatial_norm_stats["non_spatial_forcings_std"],
            )
            #this is redundant with norm_scheme == zg_log
            
            if(self.zg_log):
                geoarches_stats_path = importlib.resources.files(geoarches_stats)
                norm_file_path = geoarches_stats_path / 'cmip_stats_zg_log.pt'
                spatial_norm_stats = torch.load(norm_file_path)
                if(norm_scheme == 'full_ocean'):
                    self.data_mean['level'][-1] = spatial_norm_stats['level_mean'][-1,1:]
                    self.data_std['level'][-1] = spatial_norm_stats['level_std'][-1,1:].nanmean(axis=(-1, -2),keepdims=True)
                else:
                    self.data_mean['level'][-1] = spatial_norm_stats['level_mean'][-1]
                    self.data_std['level'][-1] = spatial_norm_stats['level_std'][-1].nanmean(axis=(-1, -2),keepdims=True)
            if(len(level_variables) == 4):
                self.data_mean['level'] = self.data_mean['level'][:len(level_variables)]     
                self.data_std['level'] = self.data_std['level'][:len(level_variables)]            
            if(not self.full_ozone):
                self.data_mean["spatial_forcings"] = [
                    self.data_mean["spatial_forcings"][i]
                    for i, val in enumerate(self.spatial_forcing_variables)
                    if val in self.master_list_spatial_forcing_variables
                ]
                self.data_mean["non_spatial_forcings"] = [
                    self.data_mean["non_spatial_forcings"][i]
                    for i, val in enumerate(self.non_spatial_forcing_variables)
                    if val in self.master_list_non_spatial_forcing_variables
                ]
                self.data_std["spatial_forcings"] = [
                    self.data_std["spatial_forcings"][i]
                    for i, val in enumerate(self.spatial_forcing_variables)
                    if val in self.master_list_spatial_forcing_variables
                ]
                self.data_std["non_spatial_forcings"] = [
                    self.data_std["non_spatial_forcings"][i]
                    for i, val in enumerate(self.non_spatial_forcing_variables)
                    if val in self.master_list_non_spatial_forcing_variables
                ]
            else:
                #accidentally added orography to full_ozone normalization
                self.data_mean["spatial_forcings"] = self.data_mean["spatial_forcings"][:-1]
                self.data_mean["non_spatial_forcings"] = self.data_mean["non_spatial_forcings"][:-1]
                self.data_std["spatial_forcings"] = self.data_std["spatial_forcings"][:-1]
                self.data_std["non_spatial_forcings"] = self.data_std["non_spatial_forcings"][:-1]


        self.surface_variables = surface_variables
        self.level_variables = [
            a + " " + str(p // 100) for a in level_variables for p in pressure_levels
        ]
        times_seconds = [v[2].item() // 10**9 for k, v in self.id2pt.items()]
        self.next_timestamp_map = {k: v for k, v in list(zip(times_seconds, times_seconds[1:]))}

        # override netcdf functionality
        self.timestamps = sorted(self.timestamps, key=lambda x: (x[0], x[1]))  # sort by timestamp
        self.orography = torch.load(f"{forcings_path}/orography.pt")[
            None
        ]  # add dim for stacking later
        self.orography = (self.orography * self.orography.mean()) / self.orography.std()

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
        times = pd.to_datetime(out["timestamp"].cpu().numpy(), unit="s").tz_localize(None)
        current_month = torch.tensor(times.month) - 1 % 12
        current_year = torch.tensor(times.year)

        t = self.lead_time_months  # multistep
        out["next_state"] = super().__getitem__(i + t // self.timedelta)
        # Load multiple future timestamps if specified.
        if self.multistep > 1:
            future_states = []
            future_timestamps = []
            for k in range(1, self.multistep + 1):
                state = super().__getitem__(i + k * t // self.timedelta)
                future_states.append(state)
                future_timestamps.append(
                    torch.tensor(
                        self.id2pt[i + k * t // self.timedelta][2].item() // 10**9,
                        dtype=torch.int64,
                    )
                )
            out["future_states"] = torch.stack(future_states, dim=0)
            out["future_timestamps"] = torch.stack(future_timestamps, dim=0)
        if self.load_prev > 1:
            prev_states = []
            for k in range(0, self.load_prev):
                prev_states.append(
                    super().__getitem__(i - (self.lead_time_months * k + 1) // self.timedelta)
                )
            out["prev_state"] = torch.stack(prev_states, dim=0)
        else:
            out["prev_state"] = super().__getitem__(
                i - self.lead_time_months // self.timedelta
            )
            # prev_timestamp = (
            #     self.id2pt[i - self.lead_time_months // self.timedelta][2].item() // 10**9
            # )
            # times = pd.to_datetime(prev_timestamp, unit="s").tz_localize(None)
        if(self.full_ozone):
            out['state']['spatial_forcings'] = torch.concatenate([out['state']['spatial_forcings'],out['state']['ozone'][0]])
            out['prev_state']['spatial_forcings'] = torch.concatenate([out['prev_state']['spatial_forcings'],out['prev_state']['ozone'][0]])
            out['next_state']['spatial_forcings'] = torch.concatenate([out['next_state']['spatial_forcings'],out['next_state']['ozone'][0]])
    
            # else:
            #     out['state']['spatial_forcings'] = torch.concatenate([out['state']['spatial_forcings'],out['state']['ozone'][0]])
            del out['state']['ozone']
            del out['prev_state']['ozone']
            del out['next_state']['ozone']

            
        # if(self.full_ozone):
        #     #need to collapse 
        #     def reshape_spatial_forcings(t_dict):
        #         t_dict['spatial_forcings'] = t_dict['spatial_forcings'].reshape(t_dict['spatial_forcings'].shape[0]+t_dict['spatial_forcings'].shape[-1],t_dict['spatial_forcings'].shape[-3],t_dict['spatial_forcings'].shape[-2])
        #         return t_dict
        #     out = {k: reshape_spatial_forcings(v) if "state" in k else v for k, v in out.items()}
            # out['state']['spatial_forcings'] = out['state']['spatial_forcings'].reshape(out['state']['spatial_forcings'].shape[0]*out['state']['spatial_forcings'].shape[1],out['state']['spatial_forcings'].shape[-2],out['state']['spatial_forcings'].shape[-1])
        out = {k: replace_inf_and_large_values(v,1e30,torch.nan) if "state" in k else v for k, v in out.items()}
        if(self.zg_log):
            out['state']['level'][-1] = torch.log(out['state']['level'][-1])
            out['next_state']['level'][-1] = torch.log(out['next_state']['level'][-1])
            if(self.load_prev > 1):
                out['prev_state']['level'][:,-1] = torch.log(out['prev_state']['level'][:,-1])
            else:
                out['prev_state']['level'][-1] = torch.log(out['prev_state']['level'][-1])

        if normalize:
            out = self.normalize(out, month=current_month)
            out = {k: apply_nan_to_num(v) if "state" in k else v for k, v in out.items()}
        if(self.add_orography):
            out['state']['spatial_forcings'] = torch.concatenate([out['state']['spatial_forcings'],self.orography],dim=0)
            if(self.load_prev > 1):
                out['prev_state']['spatial_forcings'] = torch.stack([torch.concatenate([state['spatial_forcings'],self.orography],dim=0) for state in out['prev_state']])
            if(self.multistep > 1):
                out['future_states']['spatial_forcings'] = torch.stack([torch.concatenate([state['spatial_forcings'],self.orography],dim=0) for state in out['future_states']])
    
        # out['prev_state']['spatial_norm'] = torch.concatenate([out['prev_state']['spatial_norm'],self.orography],dim=1)
        # need to replace nans with mask_value
        # mask = {k: (get_non_nan_mask(v) if "state" in k else v) for k, v in out.items()}
        # out = {k: (v * mask[k] if "state" in k else v) for k, v in out.items()}
        return out

    def normalize(self, batch, stateless=False, month=None):
        device = list(batch.values())[0].device
        means = self.data_mean.to(device)
        stds = self.data_std.to(device)
        if stateless:
            batch = (batch - means) / stds
            # batch = replace_inf_and_large_values(batch, 1e30)
            return batch
        else:
            dict_out = {k: ((v - means) / stds if "state" in k else v) for k, v in batch.items()}
            # dict_out = {
            #     k: (replace_inf_and_large_values(v, 1e30) if "state" in k else v)
            #     for k, v in dict_out.items()
            # }  # sometimes nans are actually 1e38 need to check this
            return dict_out

    # def denormalize(self, batch, stateless=False, month=None):
    #     if stateless:
    #         device = batch.device
    #     else:
    #         device = list(batch.values())[0].device
    #     means = self.data_mean.to(device)
    #     stds = self.data_std.to(device)
    #     for key in ['non_spatial_forcings', 'spatial_forcings']:
    #         if key in means:
    #             del means[key]
    #         if key in stds:
    #             del stds[key]
    #         if key in batch['state']:
    #             del batch['state'][key]
    #         if key in batch['next_state']:
    #             del batch['next_state'][key]
    #         if key in batch['prev_state']:
    #             del batch['prev_state'][key]
    #     if stateless:
    #         return (batch * stds) + means
    #     else:
    #         return {k: ((v * stds) + means if "state" in k else v) for k, v in batch.items()}
    def denormalize(self, batch, stateless=False, month=None):
        if stateless:
            device = batch.device
        else:
            device = list(batch.values())[0].device

        means = self.data_mean.to(device)
        stds = self.data_std.to(device)

        skip_keys = {"non_spatial_forcings", "spatial_forcings"}

        if stateless:
            result = {}
            for k, v in batch.items():
                if k not in skip_keys:
                    result[k] = (v * stds[k]) + means[k]
                else:
                    result[k] = v
            return result
        else:
            result = {}
            for k, v in batch.items():
                if "state" in k:
                    result[k] = {
                        kk: ((vv * stds[kk]) + means[kk]) if kk not in skip_keys else vv
                        for kk, vv in v.items()
                    }
                else:
                    result[k] = v
            return result
