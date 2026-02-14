import importlib.resources

import torch
from tensordict.tensordict import TensorDict

from geoarches.utils.tensordict_utils import (
    apply_nan_to_num,
    get_non_nan_mask,
    replace_inf_and_large_values,
)

from .. import stats as geoarches_stats
from .netcdf import XarrayDataset


class CMIPForecastLeadTime(XarrayDataset):
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
        lead_times=None,
        return_all_lead_times=False,
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
            # "cfc11",
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
        full_ozone=False,
        aerosol_log=False
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
        # Normalize lead_times: allow user to pass a list of lead times (months)
        # Backwards compatible behaviour:
        # - if `lead_times` is provided it is used (list of ints)
        # - else if `multistep` > 1, create lead_times as [lead_time_months * (i+1) for i in range(multistep)]
        # - else use [lead_time_months]
        if lead_times is not None:
            # ensure a sorted list of ints
            self.lead_times = sorted([int(x) for x in lead_times])
        else:
            if multistep is not None and multistep > 1:
                self.lead_times = [lead_time_months * (i + 1) for i in range(multistep)]
            else:
                self.lead_times = [lead_time_months]
        filename_filters = dict(
            all=(lambda _: True),
            train=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.memmap" for y in train_filter]
            ),
            val=lambda x: any(
                substring in x for substring in [f"{str(y)}_interpolation.memmap" for y in val_filter]
            ),
            test=lambda x: any(
                substring in x for substring in [f"{str(y)}_interpolation.memmap" for y in test_filter]
            ),
            val1=lambda x: any(
                substring in x for substring in [f"{str(y)}_interpolation.memmap" for y in ["ssp119"]]
            ),
            val2=lambda x: any(
                substring in x for substring in [f"{str(y)}_interpolation.memmap" for y in ["ssp370"]]
            ),
            val3=lambda x: any(
                substring in x for substring in [f"{str(y)}_interpolation.memmap" for y in ["ssp534-over"]]
            ),
            abrupt=lambda x: any(
                substring in x for substring in [f"{str(y)}_interpolation.memmap" for y in ["abrupt-4xCO2"]]
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
            # "cfc11",
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
                # return torch.load(f'{geoarches_stats_path}/cmip_stats_{getattr(cfg, "norm_scheme", None)}.pt')
            stats_file_path = f'cmip_stats_{self.norm_scheme}.pt'
            # if self.full_ozone: 
            #     stats_file_path = "cmip_stats_full_ozone.pt"
            # elif norm_scheme == 'zg_non_log':
            #     stats_file_path = "cmip_stats_zg_non_log.pt"

            # elif norm_scheme == 'full_ocean':
            #     stats_file_path = "cmip_stats_full_ocean.pt"
            # elif norm_scheme == 'zg_log':
            #     stats_file_path = 'cmip_stats_zg_log.pt'
            # elif norm_scheme == 'longer_temps':
            #     stats_file_path = 'cmip_stats_longer_temp.pt'

            # elif ("piControl" in train_filter) and (norm_scheme == 'spatial_new'):
            #     stats_file_path = "cmip_stats_test_welfords.pt"
            # elif "piControl" in train_filter:
            #     stats_file_path = "cmip_stats_piControl.pt"
            # else:
            #     stats_file_path = "cmip_stats_test_welfords_without_piControl.pt"
            geoarches_stats_path = importlib.resources.files(geoarches_stats)
            norm_file_path = geoarches_stats_path / stats_file_path
            spatial_norm_stats = torch.load(norm_file_path,weights_only=False)
            # a hack to add exp_id without needing to normalize it.
            spatial_norm_stats["non_spatial_forcings_mean"] = torch.concat(
                [spatial_norm_stats["non_spatial_forcings_mean"], torch.tensor([0])]
            )
            spatial_norm_stats["non_spatial_forcings_std"] = torch.concat(
                [spatial_norm_stats["non_spatial_forcings_std"], torch.tensor([1])]
            )
            if ('full_ocean' in path):
                self.data_mean = TensorDict(
                    surface=spatial_norm_stats["surface_mean"],
                    lev=spatial_norm_stats["lev_mean"],
                    spatial_forcings=spatial_norm_stats["spatial_forcings_mean"],
                    non_spatial_forcings=spatial_norm_stats["non_spatial_forcings_mean"],
                )
                self.data_std = TensorDict(
                    surface=spatial_norm_stats["surface_std"].nanmean(axis=(-1, -2), keepdim=True),
                    lev=spatial_norm_stats["lev_std"].nanmean(axis=(-1, -2), keepdim=True),
                    spatial_forcings=spatial_norm_stats["spatial_forcings_std"].nanmean(
                        axis=(-1, -2), keepdim=True
                    ),
                    non_spatial_forcings=spatial_norm_stats["non_spatial_forcings_std"],
                )
            else:
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
                # if(len(level_variables) == 4): #special case when zg is removed, need to be more precise here
                #     self.data_mean['level'] = self.data_mean['level'][:len(level_variables)]     
                #     self.data_std['level'] = self.data_std['level'][:len(level_variables)]
            #this is redundant with norm_scheme == zg_log
            # if(self.zg_log):
            #     geoarches_stats_path = importlib.resources.files(geoarches_stats)
            #     norm_file_path = geoarches_stats_path / 'cmip_stats_zg_log.pt'
            #     spatial_norm_stats = torch.load(norm_file_path,weights_only=False)
            #     if(norm_scheme == 'full_ocean'):
            #         self.data_mean['level'][-1] = spatial_norm_stats['level_mean'][-1,1:]
            #         self.data_std['level'][-1] = spatial_norm_stats['level_std'][-1,1:].nanmean(axis=(-1, -2),keepdims=True)
            #     else:
            #         self.data_mean['level'][-1] = spatial_norm_stats['level_mean'][-1]
            #         self.data_std['level'][-1] = spatial_norm_stats['level_std'][-1].nanmean(axis=(-1, -2),keepdims=True)
            # if(len(level_variables) == 4): #special no zg case
            #     self.data_mean['level'] = self.data_mean['level'][:len(level_variables)]     
            #     self.data_std['level'] = self.data_std['level'][:len(level_variables)]       

            if(not self.full_ozone):
                if(len(self.spatial_forcing_variables) ==0):
                    self.data_mean["spatial_forcings"] = 0
                    self.data_std['spatial_forcings'] = 1
                else:
                    self.data_mean["spatial_forcings"] = [
                        self.data_mean["spatial_forcings"][self.master_list_spatial_forcing_variables.index(val)]
                        for i, val in enumerate(self.spatial_forcing_variables)
                        if val in self.master_list_spatial_forcing_variables
                    ]

                    self.data_std["spatial_forcings"] = [
                        self.data_std["spatial_forcings"][self.master_list_spatial_forcing_variables.index(val)]
                        for i, val in enumerate(self.spatial_forcing_variables)
                        if val in self.master_list_spatial_forcing_variables
                    ]

                #is this an error? 
                # self.data_std["non_spatial_forcings"] = torch.tensor([
                #     self.data_std["non_spatial_forcings"][self.master_list_non_spatial_forcing_variables.index(val)]
                #     for i, val in enumerate(self.non_spatial_forcing_variables)
                #     if val in self.master_list_non_spatial_forcing_variables
                # ])
            else:
                #accidentally added orography to full_ozone normalization
                self.data_mean["spatial_forcings"] = self.data_mean["spatial_forcings"][:-1]
                self.data_mean["non_spatial_forcings"] = self.data_mean["non_spatial_forcings"][:-1]
                self.data_std["spatial_forcings"] = self.data_std["spatial_forcings"][:-1]
                self.data_std["non_spatial_forcings"] = self.data_std["non_spatial_forcings"][:-1]
        self.data_std["non_spatial_forcings"] = torch.tensor([
            self.data_std["non_spatial_forcings"][self.master_list_non_spatial_forcing_variables.index(val)]
            for i, val in enumerate(self.non_spatial_forcing_variables)
            if val in self.master_list_non_spatial_forcing_variables
        ])
        self.data_mean["non_spatial_forcings"] = torch.tensor([
            self.data_mean["non_spatial_forcings"][self.master_list_non_spatial_forcing_variables.index(val)]
            for i, val in enumerate(self.non_spatial_forcing_variables)
            if val in self.master_list_non_spatial_forcing_variables
        ])
        self.surface_variables = surface_variables
        # self.level_variables = [
        #     a + " " + str(p // 100) for a in level_variables for p in pressure_levels
        # ]
        times_seconds = [v[2].item() // 10**9 for k, v in self.id2pt.items()]
        self.next_timestamp_map = {k: v for k, v in list(zip(times_seconds, times_seconds[1:]))}

        # override netcdf functionality
        self.timestamps = sorted(self.timestamps, key=lambda x: (x[0], x[1]))  # sort by timestamp
        self.orography = torch.load(f"{forcings_path}/orography.pt")[
            None
        ]  # add dim for stacking later
       
        self.orography = (self.orography - self.orography.mean()) / self.orography.std()

    def convert_to_tensordict(self, xr_dataset):
        """
        input xarr should be a single time slice
        """
        tdict = super().convert_to_tensordict(xr_dataset)
        tdict["surface"] = tdict["surface"].unsqueeze(-3)
        return tdict

    def __len__(self):
        # Take into account previous and/or future timestamps loaded for one example.
        # Convert lead times (months) to index steps using timedelta
        ref_step = min(self.lead_times) // self.timedelta
        max_future_step = max(self.lead_times) // self.timedelta
        offset_steps = max_future_step + (self.load_prev * ref_step)
        return super().__len__() - offset_steps

    
    def __getitem__(self, i, normalize=True):

        # Shift index if previous state is requested. Use reference step (smallest lead).
        ref_step = min(self.lead_times) // self.timedelta
        i = i + self.load_prev * ref_step

        out = TensorDict()
        
        out["state"] = super().__getitem__(i).clone()

        out["timestamp"] = torch.tensor(
            self.id2pt[i][2].item() // 10**9,
            dtype=torch.int64,
        )
        
        # Choose a single lead time at random from the configured lead_times.
        lead_steps = [lt // self.timedelta for lt in self.lead_times]
        if self.return_all_lead_times:
            next_states = []
            for step in lead_steps:
                next_states.append(super().__getitem__(i + step).clone())
            
            out["next_state"] = torch.stack(next_states, dim=0)
            # Expose which lead time (months) was selected for this example
            out["lead_time_months"] = torch.tensor(self.lead_times, dtype=torch.int64)
            out["lead_time"] = torch.tensor(lead_steps, dtype=torch.int64)
        else:
            if len(lead_steps) == 1:
                chosen_idx = 0
            else:
                chosen_idx = int(torch.randint(low=0, high=len(lead_steps), size=(1,)).item())
            chosen_step = lead_steps[chosen_idx]
            out["next_state"] = super().__getitem__(i + chosen_step).clone()
            out["lead_time_months"] = torch.tensor(self.lead_times[chosen_idx], dtype=torch.int64)
            out["lead_time"] = torch.tensor([chosen_step], dtype=torch.int64)

        # Previous state (use reference step)
        out["prev_state"] = super().__getitem__(i - ref_step).clone()
        # if(self.one_year_lead_time):

        #     #add year placeholder, this kind of doubles importance of ta for now, but need to have a place holder for next year pred for autoregression
        #     year_tas = out['state']['surface'][self.surface_variables.index('tas')][None]
        #     year_ta = out['state']['level'][self.level_variables.index('ta')][None]
        #     out['state']['surface'] = torch.concatenate([out['state']['surface'],year_tas],dim=0)
        #     out['state']['level'] = torch.concatenate([out['state']['level'],year_ta],dim=0)

        #     next_year = super().__getitem__(i+ self.lead_time_months // self.timedelta+12)
        #     next_year_tas = next_year['surface'][self.surface_variables.index('tas')][None]
        #     next_year_ta = next_year['level'][self.level_variables.index('ta')][None]
        #     out['next_state']['surface'] = torch.concatenate([out['next_state']['surface'],next_year_tas],dim=0)
        #     out['next_state']['level'] = torch.concatenate([out['next_state']['level'],next_year_ta],dim=0)

        #     prev_year = super().__getitem__(i - self.lead_time_months // self.timedelta - 12)
        #     prev_year_tas = prev_year['surface'][self.surface_variables.index('tas')][None]
        #     prev_year_ta = prev_year['level'][self.level_variables.index('ta')][None]
        #     out['prev_state']['surface'] = torch.concatenate([out['prev_state']['surface'],prev_year_tas],dim=0)
        #     out['prev_state']['level'] = torch.concatenate([out['prev_state']['level'],prev_year_ta],dim=0)

        # Replace inf/large values IN-PLACE instead of cloning
        for k, v in out.items():
            if "state" in k:
                for key, tensor in v.items():
                    # In-place replacement  
                    mask = (tensor.abs() > 1e30) | torch.isinf(tensor)
                    tensor[mask] = torch.nan
        # if(self.aerosol_log):
        #     out['state']['spatial_forcings'][3:9] = torch.log(out['state']['spatial_forcings'][3:9])
        #     out['next_state']['spatial_forcings'][3:9] = torch.log(out['next_state']['spatial_forcings'][3:9])
        #     out['prev_state']['spatial_forcings'][3:9] = torch.log(out['prev_state']['spatial_forcings'][3:9])

        # # Normalize in-place if possible
        if normalize:
            out = self.normalize(out)
            
            for k, v in out.items():
                if "state" in k:
                    for key, value in v.items():
                        # Ensure contiguous before in-place operation
                        if not value.is_contiguous():
                            v[key] = value.contiguous()
                            value = v[key]
                        torch.nan_to_num(value, out=value)
        # Handle orography
        if self.add_orography:
            for state_key in ['state', 'prev_state', 'next_state']:
                if state_key in out:
                    old_spatial = out[state_key]['spatial_forcings']
                    if old_spatial.ndim == 3:
                        out[state_key]['spatial_forcings'] = torch.cat(
                            [old_spatial, self.orography], dim=0
                        ).contiguous()
                    elif old_spatial.ndim == 4:
                        # Stacked case (S, C, H, W)
                        S = old_spatial.shape[0]
                        # self.orography is (1, H, W). Expand to (S, 1, H, W)
                        oro_expanded = self.orography.unsqueeze(0).expand(S, -1, -1, -1)
                        out[state_key]['spatial_forcings'] = torch.cat(
                            [old_spatial, oro_expanded], dim=1
                        ).contiguous()
                    del old_spatial

        return out
        
    def normalize(self, batch, stateless=False, month=None):
        device = batch['state'].device
        # means = self.data_mean.to(device)

        if not hasattr(self, '_cached_means') or self._cached_means.device != device:
            self._cached_means = self.data_mean.to(device, non_blocking=True)
            self._cached_stds = self.data_std.to(device, non_blocking=True)
        
        means = self._cached_means
        stds = self._cached_stds
        
        if stateless:
            # Return normalized batch without modifying original
            return (batch - means) / stds
        else:
            # Normalize in-place where possible
            for k, v in batch.items():
                if "state" in k:
                    # Method 1: In-place operations (if v allows it)
                    if isinstance(v, dict) or isinstance(v, TensorDict):
                        for inner_k, inner_v in v.items():
                            # Compute normalized value
                            mean_key = means.get(inner_k, 0)
                            std_key = stds.get(inner_k, 1)
                            
                            # Normalize with minimal intermediate tensors
                            normalized = inner_v.sub(mean_key).div_(std_key)
                            batch[k][inner_k] = normalized
                    else:
                        # Simple tensor case
                        normalized = v.sub(means).div_(stds)
                        batch[k] = normalized
            
            return batch

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
