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

import psutil
import os

def print_cpu_memory_usage(position):
    mem = psutil.virtual_memory()
    
    used_mb = (mem.total - mem.available) / (1024 ** 2)  # all processes combined
    total_mb = mem.total / (1024 ** 2)
    available_mb = mem.available / (1024 ** 2)
  # Current process memory info
    process = psutil.Process(os.getpid())
    process_mem_mb = process.memory_info().rss / (1024 ** 2)  # Resident Set Size in MB

    print(f"{position} | System Memory: used={used_mb:.2f} MB | available={available_mb:.2f} MB | total={total_mb:.2f} MB | Process Memory: {process_mem_mb:.2f} MB")


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
        full_ozone=False,
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

    def __getitem__old(self, i, normalize=True):
        # print_cpu_memory_usage('beginning of getitem')

        # Shift index forward if need to load previous timestamp.
        i = i + self.load_prev * self.lead_time_months // self.timedelta 
        out = TensorDict() #this fixed a several days long memory leak ......
        # out = dict()
        #  load current state
        out["state"] = super().__getitem__(i)
        out["timestamp"] = torch.tensor(
            self.id2pt[i][2].item() // 10**9,
            dtype=torch.int64,
        )  # time in seconds
        # timestamps = out["timestamp"].detach().to("cpu").clone().numpy()
        # times = pd.to_datetime(timestamps, unit="s").tz_localize(None)

        # times = pd.to_datetime(out["timestamp"].cpu().numpy(), unit="s").tz_localize(None)
        # current_month = torch.tensor(times.month) - 1 % 12
        # current_year = torch.tensor(times.year)

        # print_cpu_memory_usage('before get nextstate')
#         out['state'] =   TensorDict(
#     {
#         "lev": torch.randn(1, 10, 144, 144),
#         "level": torch.randn(5, 17, 144, 144),
#         "non_spatial_forcings": torch.randn(6),
#         "ozone": torch.randn(1, 66, 144, 144),
#         "spatial_forcings": torch.randn(9, 144, 144),
#         "surface": torch.randn(8, 1, 144, 144),
#     },
#     batch_size=[],
# )
        t = self.lead_time_months  # multistep
        out["next_state"] = super().__getitem__(i + t // self.timedelta)
        # print_cpu_memory_usage('after get nextstate')
#         out['next_state'] = TensorDict(
#     {
#         "lev": torch.randn(1, 10, 144, 144),
#         "level": torch.randn(5, 17, 144, 144),
#         "non_spatial_forcings": torch.randn(6),
#         "ozone": torch.randn(1, 66, 144, 144),
#         "spatial_forcings": torch.randn(9, 144, 144),
#         "surface": torch.randn(8, 1, 144, 144),
#     },
#     batch_size=[],
# )

        # Load multiple future timestamps if specified.
        # if self.multistep > 1:
        #     future_states = []
        #     future_timestamps = []
        #     for k in range(1, self.multistep + 1):
        #         state = super().__getitem__(i + k * t // self.timedelta)
        #         future_states.append(state)
        #         future_timestamps.append(
        #             torch.tensor(
        #                 self.id2pt[i + k * t // self.timedelta][2].item() // 10**9,
        #                 dtype=torch.int64,
        #             )
        #         )
        #     out["future_states"] = torch.stack(future_states, dim=0)
        #     out["future_timestamps"] = torch.stack(future_timestamps, dim=0)
        # out['prev_state'] = TorchDict({'level':torch.rand(1,
        # if self.load_prev > 1:
        #     prev_states = []
        #     for k in range(0, self.load_prev):
        #         prev_states.append(
        #             super().__getitem__(i - (self.lead_time_months * k + 1) // self.timedelta)
        #         )
        #     out["prev_state"] = torch.stack(prev_states, dim=0)
        # else:
        out["prev_state"] = super().__getitem__(
            i - self.lead_time_months // self.timedelta
        )
#         out['prev_state'] = TensorDict(
#     {
#         "lev": torch.randn(1, 10, 144, 144),
#         "level": torch.randn(5, 17, 144, 144),
#         "non_spatial_forcings": torch.randn(6),
#         "ozone": torch.randn(1, 66, 144, 144),
#         "spatial_forcings": torch.randn(9,144, 144),
#         "surface": torch.randn(8, 1, 144, 144),
#     },
#     batch_size=[],
# )
#             # prev_timestamp = (
            #     self.id2pt[i - self.lead_time_months // self.timedelta][2].item() // 10**9
            # )
            # times = pd.to_datetime(prev_timestamp, unit="s").tz_localize(None)
        # print_cpu_memory_usage('after get prev state')

        # if self.full_ozone:
        #     # Create new tensors detached and cloned (to break any reference to original storage)
        #     out['state']['spatial_forcings'] = torch.cat(
        #         [out['state']['spatial_forcings'], out['state']['ozone'][0].detach().clone()], dim=0
        #     )
        #     out['prev_state']['spatial_forcings'] = torch.cat(
        #         [out['prev_state']['spatial_forcings'], out['prev_state']['ozone'][0].detach().clone()], dim=0
        #     )
        #     out['next_state']['spatial_forcings'] = torch.cat(
        #         [out['next_state']['spatial_forcings'], out['next_state']['ozone'][0].detach().clone()], dim=0
        #     )
        #     del out['next_state']['ozone']
        #     del out['prev_state']['ozone']
        #     del out['state']['ozone']
        # def copy_two_layers_deep(data, skip_key='ozone'):
        #     new_data = TensorDict()
        #     for outer_k, outer_v in data.items():
        #         if isinstance(outer_v, TensorDict):
        #             # Second layer is a dict → clone its tensors except the skipped key
        #             new_data[outer_k] = {
        #                 inner_k: inner_v.clone()
        #                 for inner_k, inner_v in outer_v.items()
        #                 if inner_k != skip_key
        #             }
        #         elif torch.is_tensor(outer_v):
        #             # Second layer is directly a tensor → clone it
        #             new_data[outer_k] = outer_v.clone()
        #         else:
        #             # Unrecognized type (keep as is or shallow copy)
        #             new_data[outer_k] = outer_v
        #     return new_data
        # # print(out)
        # out = copy_two_layers_deep(out)
        # print(out)
            # out = TensorDict({
            #     outer_k: {
            #         inner_k: inner_v.clone()
            #         for inner_k, inner_v in outer_v.items()
            #         if or (inner_k != 'ozone')
            #     }
            #     for outer_k, outer_v in out.items() 
            # })
            # Explicitly drop old tensors to free memory
            # del out['state']['ozone']
            # del out['prev_state']['ozone']
            # del out['next_state']['ozone']
            
            # # Optionally force garbage collection (useful if in a tight training loop)
            # import gc
            # gc.collect()
            # else:
            #     out['state']['spatial_forcings'] = torch.concatenate([out['state']['spatial_forcings'],out['state']['ozone'][0]])
        # import gc        
        # gc.collect()
        # torch.cuda.empty_cache()
        # if self.full_ozone:
        #     # Detach and ensure these are on CPU or safe device
        #     def safe_concat(a, b):
        #         # Ensure tensors are detached from any graph and contiguous
        #         a = a.detach() if a.requires_grad else a
        #         b = b.detach() if b.requires_grad else b
        #         return torch.cat([a, b], dim=0).contiguous()
        
        #     # Concatenate ozone and spatial forcings safely
        #     out['state']['spatial_forcings'] = safe_concat(out['state']['spatial_forcings'], out['state']['ozone'][0])
        #     out['prev_state']['spatial_forcings'] = safe_concat(out['prev_state']['spatial_forcings'], out['prev_state']['ozone'][0])
        #     out['next_state']['spatial_forcings'] = safe_concat(out['next_state']['spatial_forcings'], out['next_state']['ozone'][0])
        
        #     # Explicitly delete unused keys and clear cache
        #     for key in ['state', 'prev_state', 'next_state']:
        #         if 'ozone' in out[key]:
        #             del out[key]['ozone']
        # if(self.full_ozone):
        #     #need to collapse 
        #     def reshape_spatial_forcings(t_dict):
        #         t_dict['spatial_forcings'] = t_dict['spatial_forcings'].reshape(t_dict['spatial_forcings'].shape[0]+t_dict['spatial_forcings'].shape[-1],t_dict['spatial_forcings'].shape[-3],t_dict['spatial_forcings'].shape[-2])
        #         return t_dict
        #     out = {k: reshape_spatial_forcings(v) if "state" in k else v for k, v in out.items()}
            # out['state']['spatial_forcings'] = out['state']['spatial_forcings'].reshape(out['state']['spatial_forcings'].shape[0]*out['state']['spatial_forcings'].shape[1],out['state']['spatial_forcings'].shape[-2],out['state']['spatial_forcings'].shape[-1])
        out = TensorDict({k: replace_inf_and_large_values(v,1e30,torch.nan) if "state" in k else v for k, v in out.items()})
        # print_cpu_memory_usage('before zg_log')
        # if(self.zg_log):
        #     out['state']['level'][-1] = torch.log(out['state']['level'][-1]).detach()
        #     out['next_state']['level'][-1] = torch.log(out['next_state']['level'][-1]).detach()
        #     if(self.load_prev > 1):
        #         out['prev_state']['level'][:,-1] = torch.log(out['prev_state']['level'][:,-1]).detach()
        #     else:
        #         out['prev_state']['level'][-1] = torch.log(out['prev_state']['level'][-1]).detach()
        # print_cpu_memory_usage('after zg_log')
        # import sys
        # print(sys.getrefcount(out['state']['spatial_forcings']))
        # if normalize:
        #     out = self.normalize(out, month=current_month)
        #     out = {k: apply_nan_to_num(v) if "state" in k else v for k, v in out.items()}

        if normalize:
            out = self.normalize(out)

            # print_cpu_memory_usage('before nan_to_num')

            for k, v in out.items():
                if "state" in k:
                    for key, value in v.items():
                        torch.nan_to_num(value, out=value)
        # print_cpu_memory_usage('after normalize')

        # if(self.add_orography):
        #     out['state']['spatial_forcings'] = torch.concatenate([out['state']['spatial_forcings'],self.orography],dim=0).detach()
        #     if(self.load_prev > 1):
        #         out['prev_state']['spatial_forcings'] = torch.stack([torch.concatenate([state['spatial_forcings'],self.orography],dim=0) for state in out['prev_state']])
        #     if(self.multistep > 1):
        #         out['future_states']['spatial_forcings'] = torch.stack([torch.concatenate([state['spatial_forcings'],self.orography],dim=0) for state in out['future_states']])
        # if self.add_orography:
        # Update current state
        if self.add_orography:
            # If you're always adding orography as an additional channel
            out['state'] = out['state'].update({
                'spatial_forcings': torch.stack([
                    out['state']['spatial_forcings'],
                    self.orography
                ], dim=0).detach()
            })
            # Update previous states (if list/sequence of TensorDicts)
            if self.load_prev > 1:
                updated_prev = [
                    state.update({
                        'spatial_forcings': torch.cat(
                            [state['spatial_forcings'], self.orography], dim=0
                        )
                    })
                    for state in out['prev_state']
                ]
                out['prev_state'] = torch.stack(updated_prev)
        
            # Update future states (if multistep)
            if self.multistep > 1:
                updated_future = [
                    state.update({
                        'spatial_forcings': torch.cat(
                            [state['spatial_forcings'], self.orography], dim=0
                        )
                    })
                    for state in out['future_states']
                ]
                out['future_states'] = torch.stack(updated_future)
        # out['prev_state']['spatial_norm'] = torch.concatenate([out['prev_state']['spatial_norm'],self.orography],dim=1)
        # need to replace nans with mask_value
        # mask = {k: (get_non_nan_mask(v) if "state" in k else v) for k, v in out.items()}
        # out = {k: (v * mask[k] if "state" in k else v) for k, v in out.items()}
        # print_cpu_memory_usage('end of getitem')

        return out

    
    def __getitem__(self, i, normalize=True):

        # print_cpu_memory_usage('beginning of getitem')
    
        i = i + self.load_prev * self.lead_time_months // self.timedelta 
        out = TensorDict()
        
        out["state"] = super().__getitem__(i)
        
        out["timestamp"] = torch.tensor(
            self.id2pt[i][2].item() // 10**9,
            dtype=torch.int64,
        )
        
        t = self.lead_time_months
        out["next_state"] = super().__getitem__(i + t // self.timedelta)
        out["prev_state"] = super().__getitem__(i - self.lead_time_months // self.timedelta)

        # out['next_state'] = TensorDict(
        # {
        #     "lev": torch.randn(1, 10, 144, 144),
        #     "level": torch.randn(5, 17, 144, 144),
        #     "non_spatial_forcings": torch.randn(6),
        #     # "ozone": torch.randn(1, 66, 144, 144),
        #     "spatial_forcings": torch.randn(75, 144, 144),
        #     "surface": torch.randn(8, 1, 144, 144),
        # },
        # batch_size=[],
        # )

        # out['state'] = TensorDict(
        # {
        #     "lev": torch.randn(1, 10, 144, 144),
        #     "level": torch.randn(5, 17, 144, 144),
        #     "non_spatial_forcings": torch.randn(6),
        #     # "ozone": torch.randn(1, 66, 144, 144),
        #     "spatial_forcings": torch.randn(75, 144, 144),
        #     "surface": torch.randn(8, 1, 144, 144),
        # },
        # batch_size=[],
        # )

        # out['prev_state'] = TensorDict(
        # {
        #     "lev": torch.randn(1, 10, 144, 144),
        #     "level": torch.randn(5, 17, 144, 144),
        #     "non_spatial_forcings": torch.randn(6),
        #     # "ozone": torch.randn(1, 66, 144, 144),
        #     "spatial_forcings": torch.randn(75, 144, 144),
        #     "surface": torch.randn(8, 1, 144, 144),
        # },
        # batch_size=[],
        # )






        
        # Handle ozone concatenation with explicit cleanup
        # if self.full_ozone:
        #     for state_key in ['state', 'prev_state', 'next_state']:
        #         old_spatial = out[state_key]['spatial_forcings']
        #         ozone_slice = out[state_key]['ozone'][0]
                
        #         # Create new concatenated tensor
        #         new_spatial = torch.cat([old_spatial, ozone_slice], dim=0).contiguous()
                
        #         # Update in place and delete old references
        #         out[state_key]['spatial_forcings'] = new_spatial
        #         del old_spatial, ozone_slice
            
        #     # Remove ozone from output (create new dict without ozone)
        #     def remove_ozone(state_dict):
        #         return TensorDict({k: v for k, v in state_dict.items() if k != 'ozone'})
            
        #     out['state'] = remove_ozone(out['state'])
        #     out['prev_state'] = remove_ozone(out['prev_state'])
        #     out['next_state'] = remove_ozone(out['next_state'])
    
        # Replace inf/large values IN-PLACE instead of cloning
        for k, v in out.items():
            if "state" in k:
                for key, tensor in v.items():
                    # In-place replacement
                    mask = (tensor.abs() > 1e30) | torch.isinf(tensor)
                    tensor[mask] = torch.nan
    
        # # Normalize in-place if possible
        # print(out)
        # print(self.data_mean)
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
                    out[state_key]['spatial_forcings'] = torch.cat(
                        [old_spatial, self.orography], dim=0
                    ).contiguous()
                    del old_spatial
    
        # print_cpu_memory_usage('end of getitem')
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
