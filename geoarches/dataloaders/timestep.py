import importlib.resources

import pandas as pd
import torch
from tensordict.tensordict import TensorDict

from geoarches.utils.tensordict_utils import (
    apply_isnan,
    apply_nan_to_num,
    apply_threshold,
)

from .. import stats as geoarches_stats
from .netcdf import XarrayDataset


class CMIPTimeStepForecast(XarrayDataset):
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
        norm_scheme="spatial",
        limit_examples: int = 0,
        surface_variables=None,
        level_variables=None,
        pressure_levels=[],
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
            limit_examples: Return set number of examples in dataset
            mask_value: what value to use as mask for nan values in dataset
        """
        self.__dict__.update(locals())  # concise way to update self with input arguments
        self.timedelta = 1
        train_filter = [x for x in range(1940, 2012)]
        val_filter = [x for x in range(2012, 2013)]
        test_filter = [x for x in range(2013, 2014)]

        filename_filters = dict(
            all=(lambda _: True),
            train=lambda x: any(substring in x for substring in [str(y) for y in train_filter]),
            val=lambda x: any(substring in x for substring in [str(y) for y in val_filter]),
            test=lambda x: any(substring in x for substring in [str(y) for y in test_filter]),
            empty=lambda x: False,
        )
        if filename_filter is None:
            filename_filter = filename_filters[domain]
        if variables is None:
            variables = dict(surface=surface_variables, level=level_variables, lev=["thetao"])
        dimension_indexers = {"plev": pressure_levels}

        super().__init__(
            path,
            filename_filter=filename_filter,
            variables=variables,
            limit_examples=limit_examples,
            dimension_indexers=dimension_indexers,
        )
        if self.norm_scheme != "spatial":
            self.data_mean = TensorDict(
                surface=torch.tensor(0),
                level=torch.tensor(0),
            )
            self.data_std = TensorDict(
                surface=torch.tensor(1),
                level=torch.tensor(1),
            )
        else:
            stats_file_path = "cmip_stats.pt"
            geoarches_stats_path = importlib.resources.files(geoarches_stats)
            norm_file_path = geoarches_stats_path / stats_file_path
            spatial_norm_stats = torch.load(norm_file_path)
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
        self.surface_variables = surface_variables
        self.level_variables = [
            a + " " + str(p // 100) for a in level_variables for p in pressure_levels
        ]
        times_seconds = [v[2].item() // 10**9 for k, v in self.id2pt.items()]
        self.next_timestamp_map = {k: v for k, v in list(zip(times_seconds, times_seconds[1:]))}

        # override netcdf functionality
        self.timestamps = sorted(self.timestamps, key=lambda x: (x[0], x[1]))  # sort by timestamp
        self.orography = torch.load(f"{forcings_path}/orography_forcings/orography.pt")[
            None
        ]  # add dim for stacking later

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
                future_states.append(super().__getitem__(i + k * t // self.timedelta))
                future_timestamps.append(
                    torch.tensor(
                        self.id2pt[i + k * t // self.timedelta][2].item() // 10**9,
                        dtype=torch.int64,
                    )
                )
            out["future_states"] = torch.stack(future_states, dim=0)
            out["future_timestamps"] = torch.stack(future_timestamps, dim=0)
        if self.load_prev:
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
                prev_timestamp = (
                    self.id2pt[i - self.lead_time_months // self.timedelta][2].item() // 10**9
                )
                times = pd.to_datetime(prev_timestamp, unit="s").tz_localize(None)
        # out = {k: apply_threshold_to_nan(v) if "state" in k else v for k, v in out.items()}

        if normalize:
            out = self.normalize(out, month=current_month)

        # need to replace nans with mask_value
        mask = {k: (apply_isnan(v) if "state" in k else v) for k, v in out.items()}
        out = {k: apply_nan_to_num(v) if "state" in k else v for k, v in out.items()}
        out = {k: (v * mask[k] if "state" in k else v) for k, v in out.items()}

        return out

    def normalize(self, batch, stateless=False, month=None):
        device = list(batch.values())[0].device
        means = self.data_mean.to(device)
        stds = self.data_std.to(device)
        if stateless:
            batch = (batch - means) / stds
            batch = apply_threshold(batch)
            return batch
        else:
            dict_out = {k: ((v - means) / stds if "state" in k else v) for k, v in batch.items()}
            dict_out = {
                k: (apply_threshold(v) if "state" in k else v) for k, v in dict_out.items()
            }  # sometimes nans are actually 1e38 need to check this
            return dict_out

    def denormalize(self, batch, stateless=False, month=None):
        if stateless:
            device = batch.device
        else:
            device = list(batch.values())[0].device
        means = self.data_mean.to(device)
        stds = self.data_std.to(device)
        if stateless:
            return (batch * stds) + means
        else:
            return {k: ((v * stds) + means if "state" in k else v) for k, v in batch.items()}
