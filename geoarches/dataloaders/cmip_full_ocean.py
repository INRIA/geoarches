import importlib.resources
import datetime
import torch
from tensordict.tensordict import TensorDict

from geoarches.utils.tensordict_utils import (
    apply_nan_to_num,
    get_non_nan_mask,
    replace_inf_and_large_values,
)

from .. import stats as geoarches_stats
from .netcdf import XarrayDataset
def timestamp_to_index(ts, start_year=1852, start_month=1): #the file starts after two years
    # Convert seconds from epoch to a datetime object
    dt = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=ts)
    # Calculate the monthly index
    index = (dt.year - start_year) * 12 + (dt.month - start_month)
    # print(index,dt.year,dt.month)
    return index

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
        add_climate_indices=False,
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


        dimension_indexers = {"plev": ("plev", pressure_levels), "lev": ("lev", depth_levels)}
        super().__init__(
            path,
            filename_filter=filename_filter,
            variables=variables,
            limit_examples=limit_examples,
            dimension_indexers=dimension_indexers,
            timestamp_key=lambda x: (x[0], x[1]), # sort by file, then index of time in file. 
            add_climate_indices=add_climate_indices
        )

        self.variables = dict(
            surface=surface_variables,
            level=level_variables,
            lev=["thetao"],
            spatial_forcings=spatial_forcing_variables,
            non_spatial_forcings=non_spatial_forcing_variables,
        )
        self.surface_variable_master_list = ['tos','siconc','sithick','zos','net_flux','evspsbl','huss']
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
                spatial_forcings=torch.tensor(1),
                non_spatial_forcings=torch.tensor(1)
            )
        else:
            stats_file_path = f'cmip_stats_{self.norm_scheme}.pt'
            geoarches_stats_path = importlib.resources.files(geoarches_stats)
            norm_file_path = geoarches_stats_path / stats_file_path
            spatial_norm_stats = torch.load(norm_file_path,weights_only=False)
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

        self.data_mean["spatial_forcings"] = self.data_mean["spatial_forcings"][:-1]
        self.data_std["spatial_forcings"] = self.data_std["spatial_forcings"][:-1]
        #remove ice         
        self.data_mean["surface"] = [
            self.data_mean["surface"][self.surface_variable_master_list.index(val)] for i,val 
            in enumerate(self.variables['surface'])
            if val in self.surface_variable_master_list
        ]
        self.data_std["surface"] = [
            self.data_std["surface"][self.surface_variable_master_list.index(val)] for i,val 
            in enumerate(self.variables['surface'])
            if val in self.surface_variable_master_list
        ]

        self.valid_indices = []
        ref_step = min(self.lead_times) // self.timedelta
        prev_offset = self.load_prev * ref_step
        max_future_offset = max(self.lead_times) // self.timedelta


        #here we need to check the line_id of each file, therefore cannot just clip the whole length. 
        #see netcdf, as if add_el_nino is activated, it also clips the first 24 steps
        for k in range(len(self.timestamps)):
            # Check prev
            if k - prev_offset < 0 or self.timestamps[k - prev_offset][0] != self.timestamps[k][0]:
                continue
            
            # Check future
            if k + max_future_offset >= len(self.timestamps) or self.timestamps[k + max_future_offset][0] != self.timestamps[k][0]:
                continue
                
            self.valid_indices.append(k)

        self.orography = torch.load(f"{forcings_path}/orography.pt")[
            None
        ]  # add dim for stacking later
       
        self.orography = (self.orography - self.orography.mean()) / self.orography.std()
        self.climate_indices = torch.load(f'{forcings_path}/climate_indices.pt')

    def convert_to_tensordict(self, xr_dataset):
        """
        input xarr should be a single time slice
        """
        tdict = super().convert_to_tensordict(xr_dataset)
        tdict["surface"] = tdict["surface"].unsqueeze(-3)
        return tdict

    def __len__(self):
        return len(self.valid_indices)

    
    def __getitem__(self, i, normalize=True):

        idx = self.valid_indices[i]
        out = TensorDict()
        
        out["state"] = super().__getitem__(idx).clone()

        out["timestamp"] = torch.tensor(
            self.id2pt[idx][2].item() // 10**9,
            dtype=torch.int64,
        )
        
        # Choose a single lead time at random from the configured lead_times.
        lead_steps = [lt // self.timedelta for lt in self.lead_times]
        if self.return_all_lead_times:
            next_states = []
            for step in lead_steps:

                next_states.append(super().__getitem__(idx + step).clone())
            
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
            if(chosen_step ==12):
                timestamp_to_index(self.id2pt[idx][2].item() // 10**9)
                timestamp_to_index(self.id2pt[idx + chosen_step][2].item() // 10**9)
            out["next_state"] = super().__getitem__(idx + chosen_step).clone()
            out["lead_time_months"] = torch.tensor(self.lead_times[chosen_idx], dtype=torch.int64)
            out["lead_time"] = torch.tensor([chosen_step], dtype=torch.int64)

        # Previous state (use reference step)
        ref_step = min(self.lead_times) // self.timedelta
        out["prev_state"] = super().__getitem__(idx - ref_step).clone()

        # Replace inf/large values IN-PLACE instead of cloning
        for k, v in out.items():
            if "state" in k:
                for key, tensor in v.items():
                    # In-place replacement  
                    mask = (tensor.abs() > 1e30) | torch.isinf(tensor)
                    tensor[mask] = torch.nan

        surface_filter_indices = [
            self.surface_variable_master_list.index(val) for i,val 
            in enumerate(self.variables['surface'])
            if val in self.surface_variable_master_list
        ]
        if(self.return_all_lead_times):
            out["next_state"]['surface'] = out["next_state"]["surface"][:,surface_filter_indices]
            out["state"]['surface'] = out["state"]["surface"][surface_filter_indices]
            out["prev_state"]['surface'] = out["prev_state"]["surface"][surface_filter_indices]
        else:
            out["next_state"]['surface'] = out["next_state"]["surface"][surface_filter_indices]
            out["state"]['surface'] = out["state"]["surface"][surface_filter_indices]
            out["prev_state"]['surface'] = out["prev_state"]["surface"][surface_filter_indices]            
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

        #need to get timestamp -> which index 
        # index = timestamp_to_index(self.id2pt[idx][2].item() // 10**9)
        # if self.add_climate_indices:
        #     for state_key in ['state', 'prev_state', 'next_state']:
        #         if state_key in out:
        #             for climate_index in range(6):
        #                 old_non_spatial = out[state_key]['non_spatial_forcings']
        #                 out[state_key]['non_spatial_forcings'] = torch.cat(
        #                     [old_non_spatial, self.climate_indices[:,climate_index,index]]  , dim=0
        #                 )

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
