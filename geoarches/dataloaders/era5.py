import importlib.resources
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tensordict.tensordict import TensorDict

from .. import stats as geoarches_stats
from .netcdf import XarrayDataset

filename_filters = dict(
    all=(lambda _: True),
    last_train=lambda x: ("2018" in x),
    last_train_z0012=lambda x: ("2018" in x and ("0h" in x or "12h" in x)),
    train=lambda x: not ("2019" in x or "2020" in x or "2021" in x),
    # Splits val and test  are from 2019 and 2020 respectively, but
    # we read the years before and after to account for offsets when
    # loading previous and future timestamps for an example.
    val=lambda x: ("2018" in x or "2019" in x or "2020" in x),
    test=lambda x: ("2019" in x or "2020" in x or "2021" in x),
    test_z0012=lambda x: ("2019" in x or "2020" in x or "2021" in x) and ("0h" in x or "12h" in x),
    test2022_z0012=lambda x: ("2022" in x) and ("0h" in x or "12h" in x),  # check if that works ?
    recent2=lambda x: any([str(y) in x for y in range(2007, 2019)]),
    empty=lambda x: False,
)


pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

level_variables = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "vertical_velocity",
]

surface_variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
]

surface_variables_short = {
    "10m_u_component_of_wind": "U10m",
    "10m_v_component_of_wind": "V10m",
    "2m_temperature": "T2m",
    "mean_sea_level_pressure": "SP",
}

level_variables_short = {
    "geopotential": "Z",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "temperature": "T",
    "specific_humidity": "Q",
    "vertical_velocity": "W",
}


def get_surface_variable_indices(variables=surface_variables):
    """Mapping from surface variable name to (var, lev) index in ERA5 dataset."""
    return {surface_variables_short[var]: (i, 0) for i, var in enumerate(variables)}


def get_level_variable_indices(pressure_levels=pressure_levels, variables=level_variables):
    """Mapping from level variable name to (var, lev) index in ERA5 dataset."""
    out = {}
    for var_idx, var in enumerate(variables):
        var_short = level_variables_short[var]
        for lev_idx, lev in enumerate(pressure_levels):
            out[f"{var_short}{lev}"] = (var_idx, lev_idx)
    return out


def get_headline_level_variable_indices(
    pressure_levels=pressure_levels, level_variables=level_variables
):
    """Mapping for main level variables."""
    out = get_level_variable_indices(pressure_levels, level_variables)
    return {k: v for k, v in out.items() if k in ("Z500", "T850", "Q700", "U850", "V850")}


class Era5Dataset(XarrayDataset):
    """
    This class is just for loading era5-like data, without any future or pred state.
    Does not do any normalization either.
    """

    def __init__(
        self,
        path: str = "data/era5_240/full/",
        domain: str = "train",
        filename_filter: Callable | None = None,
        variables: Dict[str, List[str]] | None = None,
        dimension_indexers: Dict[str, list] | None = None,
        return_timestamp: bool = False,
    ):
        """
        Args:
            path: Single filepath or directory holding files.
            domain: Specify data split for the filename filters (eg. train, val, test, testz0012..).
                Used if `filename_filter` is None.
            filename_filter: To filter files within `path` based on filename.
                By default, filters files based on `domain`.
            variables: Variables to load from dataset. Dict holding variable lists mapped by their keys to be processed into tensordict.
                e.g. {surface:[...], level:[...]}. By default uses standard 6 level and 4 surface vars.
            dimension_indexers: Dict of dimensions to select using Dataset.sel(dimension_indexers).
            return_timestamp: Whether to return tuple of (example, timestamp) from __getitem__().
        """
        if filename_filter is None:
            filename_filter = filename_filters[domain]

        if variables is None:
            variables = dict(surface=surface_variables, level=level_variables)

        super().__init__(
            path,
            filename_filter=filename_filter,
            variables=variables,
            dimension_indexers=dimension_indexers,
            return_timestamp=return_timestamp,
            warning_on_nan=True,
        )

    def convert_to_tensordict(self, xr_dataset):
        """
        input xarr should be a single time slice
        """
        if self.dimension_indexers:
            xr_dataset = xr_dataset.sel(self.dimension_indexers)
            #  Workaround to avoid calling sel() after transponse() to avoid OOM.
            self.already_ran_index_selection = True
        xr_dataset = xr_dataset.transpose(..., "level", "latitude", "longitude")
        tdict = super().convert_to_tensordict(xr_dataset)
        # we don't do operations on xr datasets since it takes more time than on tensors

        # unsqueeze surface (important)
        if "surface" in tdict:
            tdict["surface"] = tdict["surface"].unsqueeze(-3)

        # do we need to flip lats ?
        if xr_dataset.latitude[0] < xr_dataset.latitude[-1]:
            tdict = tdict.apply(lambda x: x.flip(-2))

        # focus on Europe
        data_shape = list(tdict.values())[0].shape
        halflon = data_shape[-1] // 2
        tdict = tdict.apply(lambda x: x.roll(halflon, -1))

        return tdict

    def convert_to_xarray(self, tdict, timestamp, levels=None):
        """
        we dont take prediction timedelta into account here
        timestamp is necessary to convert to xarray
        compressed means we only store 3 levels (500, 700, 850)
        """
        tdict = tdict.cpu()
        halflon = tdict["surface"].shape[-1] // 2
        # rebatch if not batched
        if tdict["surface"].shape == torch.Size([]):
            tdict = tdict[None]

        # roll does not work with named dimensions, use cat
        tdict = tdict.apply(lambda x: x.roll(-halflon, -1))

        # squeeze
        surface = tdict["surface"].squeeze(-3)
        level = tdict["level"]

        times = pd.to_datetime(timestamp.cpu().numpy(), unit="s").tz_localize(None)
        xr_dataset = xr.Dataset(
            data_vars=dict(
                **{
                    v: (["time", "level", "latitude", "longitude"], level[:, i])
                    for (i, v) in enumerate(self.variables["level"])
                },
                **{
                    v: (["time", "latitude", "longitude"], surface[:, i])
                    for (i, v) in enumerate(self.variables["surface"])
                },
            ),
            coords=dict(
                time=times,
                latitude=np.arange(90, -90 - 1e-6, -180 / 120),  # decreasing lats
                longitude=np.arange(0, 360, 360 / 240),
                level=pressure_levels,
            ),
        )
        if levels is not None:
            xr_dataset = xr_dataset.sel(level=levels)

        xr_dataset = xr_dataset.chunk(time=1)
        return xr_dataset

    def convert_trajectory_to_xarray(
        self,
        preds_future,
        timestamp=None,
        denormalize=True,
        levels=None,
    ):
        # here preds is a tensordict with shapes (bs, T, var, lvl, lat, lon)
        if denormalize:
            preds_future = self.denormalize(preds_future)
        step_iterations = preds_future.shape[1]

        xr_timedelta_list = [
            self.convert_to_xarray(preds_future[:, i], timestamp=timestamp, levels=levels)
            for i in range(step_iterations)
        ]
        prediction_timedeltas = [timedelta(days=i) for i in range(1, step_iterations + 1)]
        merged_xr_dataset = xr.concat(
            xr_timedelta_list, pd.Index(prediction_timedeltas, name="prediction_timedelta")
        )
        return merged_xr_dataset


class Era5Forecast(Era5Dataset):
    """
    Load Era5 data for the forecast task.

    Loads previous timestep and multiple future timesteps if configured.
    Also handles normalization.
    """

    def __init__(
        self,
        path: str = "data/era5_240/full/",
        domain: str = "train",
        filename_filter: Callable | None = None,
        timedelta_hours: int = None,
        variables: Dict[str, List[str]] | None = None,
        dimension_indexers: Dict[str, list] | None = None,
        norm_scheme: str | None = "pangu",
        lead_time_hours: int = 24,
        multistep: int = 1,
        load_prev: bool = True,
        load_clim: bool = False,
        switch_recent_data_after_steps: int = 250000,
    ):
        """
        Args:
            path: Single filepath or directory holding files.
            domain: Specify data split for the filename filters (eg. train, val, test, testz0012..)
            filename_filter: To filter files within `path` based on filename.
                By default, filters files based on `domain`.
            lead_time_hours: Time difference between current state and previous and future states.
            multistep: Number of future states to load. By default, loads next state only (current time + lead_time_hours).
            load_prev: Whether to load state at previous timestamp (current time - lead_time_hours).
            load_clim: Whether to load climatology.
            norm_scheme: Normalization scheme to use. Can be None to perform no normalization.
            timedelta_hours: Time difference (hours) between 2 consecutive timestamps. If not expecified,
                             default is 6 or 12, depeding on domain.
            variables: Variables to load from dataset. Dict holding variable lists mapped by their keys to be processed into tensordict.
                e.g. {surface:[...], level:[...] By default uses standard 6 level and 4 surface vars.
            dimension_indexers: Dict of dimensions to select using Dataset.sel(dimension_indexers).
        """
        self.__dict__.update(locals())

        super().__init__(
            path,
            filename_filter=filename_filter,
            domain=domain,
            variables=variables,
            dimension_indexers=dimension_indexers,
        )

        # depending on domain, re-set timestamp bounds

        if domain in ("val", "test", "test_z0012"):
            # re-select timestamps
            year = 2019 if domain.startswith("val") else 2020
            start_time = np.datetime64(f"{year}-01-01T00:00:00")
            if self.load_prev:
                start_time = start_time - self.lead_time_hours * np.timedelta64(1, "h")
            end_time = np.datetime64(
                f"{year+1}-01-01T00:00:00"
            ) + self.multistep * self.lead_time_hours * np.timedelta64(1, "h")
            print("start time", start_time)
            super().set_timestamp_bounds(start_time, end_time)

        if timedelta_hours:
            self.timedelta = timedelta_hours
        else:
            self.timedelta = 6 if "z0012" not in domain else 12
        self.current_multistep = 1

        # include vertical component by default
        geoarches_stats_path = importlib.resources.files(geoarches_stats)
        norm_file_path = geoarches_stats_path / "pangu_norm_stats2_with_w.pt"
        pangu_stats = torch.load(norm_file_path, weights_only=True)

        # normalization,
        if self.norm_scheme == "pangu":
            self.data_mean = TensorDict(
                surface=pangu_stats["surface_mean"],
                level=pangu_stats["level_mean"],
            )
            self.data_std = TensorDict(
                surface=pangu_stats["surface_std"],
                level=pangu_stats["level_std"],
            )

        # variable names
        # TODO: fix this below
        self.surface_variables = ["U10m", "V10m", "T2m", "SP"]
        self.level_variables = [
            a + str(p) for a in ["Z", "U", "V", "T", "Q"] for p in pressure_levels
        ]

        # Load climatology.
        self.clim_path = Path(path).parent.joinpath("era5_240_clim.nc")

    def __len__(self):
        # Take into account previous and/or future timestamps loaded for one example.
        offset = self.multistep + self.load_prev
        return super().__len__() - offset * self.lead_time_hours // self.timedelta

    def __getitem__(self, i, normalize=True):
        out = {}
        # Shift index forward if need to load previous timestamp.
        i = i + self.load_prev * self.lead_time_hours // self.timedelta

        out = dict()
        #  load current state
        out["timestamp"] = torch.tensor(
            self.id2pt[i][2].item() // 10**9,  # how to convert to tensor ?
            dtype=torch.int32,
        )  # time in seconds

        out["state"] = super().__getitem__(i)

        out["lead_time_hours"] = torch.tensor(self.lead_time_hours * int(self.multistep)).int()

        # next obsi. has function of
        T = self.lead_time_hours  # multistep

        if self.multistep > 0:
            out["next_state"] = super().__getitem__(i + T // self.timedelta)

        # Load multiple future timestamps if specified.
        if self.multistep > 1:
            future_states = []
            for k in range(1, self.multistep + 1):
                future_states.append(super().__getitem__(i + k * T // self.timedelta))
            out["future_states"] = torch.stack(future_states, dim=0)

        if self.load_prev:
            out["prev_state"] = super().__getitem__(i - self.lead_time_hours // self.timedelta)

        if self.load_clim:
            clim_xr = xr.open_dataset(self.clim_path)
            timestamp = self.id2pt[i][2]
            doy = np.datetime64(timestamp, "D") - np.datetime64(timestamp, "Y") + 1
            hour = (timestamp.astype("datetime64[h]") - timestamp.astype("datetime64[D]")).astype(
                int
            ) % 24
            climi = clim_xr.sel(dayofyear=doy.astype("int"), hour=hour)
            out["clim_state"] = self.convert_to_tensordict(climi)
            clim_xr.close()

        if normalize and self.norm_scheme:
            out = self.normalize(out)

        return out

    def normalize(self, batch):
        if self.norm_scheme is None:
            return batch

        device = list(batch.values())[0].device

        means = self.data_mean.to(device)
        stds = self.data_std.to(device)

        if "surface" in batch:
            # we can normalize directly
            return (batch - means) / stds
        out = {k: ((v - means) / stds if "state" in k else v) for k, v in batch.items()}

        return out

    def denormalize(self, batch):
        device = list(batch.values())[0].device
        means = self.data_mean.to(device)
        stds = self.data_std.to(device)

        if "surface" in batch:
            # we can denormalize directly
            return batch * stds + means

        out = {k: (v * stds + means if "state" in k else v) for k, v in batch.items()}
        return out

    def iteration_hook(self, model):
        # this is to update dataset based on how many training steps have already been performed
        if model.global_step >= self.switch_recent_data_after_steps:
            start_time = np.datetime64("2007-01-01T00:00:00")
            end_time = np.datetime64("2019-01-01T00:00:00")

            delta_start = int(self.load_prev) * self.lead_time_hours * np.timedelta64(1, "h")
            delta_end = self.multistep * self.lead_time_hours * np.timedelta64(1, "h")

            super().set_timestamp_bounds(start_time - delta_start, end_time + delta_end)
