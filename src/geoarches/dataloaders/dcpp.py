import importlib.resources

import pandas as pd
import torch
from tensordict.tensordict import TensorDict

from .. import stats as geoarches_stats
from .netcdf import NetcdfDataset

filename_filters = dict(
    all=(lambda _: True),
    train=lambda x: any(
        substring in x for substring in [f"_{str(x)}_tos_included.nc" for x in range(1, 9)]
    ),
    val=lambda x: any(
        substring in x for substring in [f"_{str(x)}_tos_included.nc" for x in range(9, 10)]
    ),
    test=lambda x: any(
        substring in x for substring in [f"_{str(x)}_tos_included.nc" for x in range(10, 11)]
    ),
    empty=lambda x: False,
)

pressure_levels = [85000, 70000, 50000]
surface_variables = ["tas", "npp", "nbp", "gpp", "cVeg", "evspsbl", "mrfso", "mrro", "ps", "tos"]
level_variables = ["hur", "hus", "o3", "ta", "ua", "va", "wap", "zg"]


def replace_nans(tensordict, value=0):
    return tensordict.apply(
        lambda x: torch.where(torch.isnan(x), torch.tensor(value, dtype=x.dtype), x)
    )


class DCPPForecast(NetcdfDataset):
    """
    Load DCPP data for the forecast task.

    Loads previous timestep and multiple future timesteps if configured.
    Also handles normalization.
    """

    def __init__(
        self,
        path="data/batch_with_tos/",
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
        self.current_multistep = 1
        if filename_filter is None:
            filename_filter = filename_filters[domain]
        if variables is None:
            variables = dict(surface=surface_variables, level=level_variables)
        dimension_indexers = {"plev": pressure_levels}
        super().__init__(
            path,
            filename_filter=filename_filter,
            variables=variables,
            limit_examples=limit_examples,
            dimension_indexers=dimension_indexers,
        )

        geoarches_stats_path = importlib.resources.files(geoarches_stats)
        norm_file_path = geoarches_stats_path / "dcpp_spatial_norm_stats.pt"
        spatial_norm_stats = torch.load(norm_file_path)

        # normalization,
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
                surface=spatial_norm_stats["surface_mean"],
                level=spatial_norm_stats["level_mean"],
            )
            self.data_std = TensorDict(
                surface=spatial_norm_stats["surface_std"],
                level=spatial_norm_stats["level_std"],
            )

        self.surface_variables = [
            "tas",
            "npp",
            "nbp",
            "gpp",
            "cVeg",
            "evspsbl",
            "mrfso",
            "mrro",
            "ps",
            "tos",
        ]
        self.level_variables = [
            a + str(p)
            for a in ["hur_", "hus_", "o3_", "ta_", "ua_", "va_", "wap_", "zg_"]
            for p in pressure_levels
        ]
        self.atmos_forcings = torch.load(f"{forcings_path}/full_atmos_normal.pt")
        self.solar_forcings = torch.load(f"{forcings_path}/full_solar_normal.pt")

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
            dtype=torch.int32,
        )  # time in seconds
        times = pd.to_datetime(out["timestamp"].cpu().numpy() * 10**9).tz_localize(None)
        current_year = (
            torch.tensor(times.month) + 1970 - 1961
        )  # plus 1970 for the timestep, -1961 to zero index
        current_month = torch.tensor(times.year) % 12

        out["forcings"] = torch.concatenate(
            [
                self.atmos_forcings[:, (current_year * 12) + current_month],
                self.solar_forcings[current_year, current_month, :],
            ]
        )
        # next obsi. has function of
        t = self.lead_time_months  # multistep

        out["next_state"] = super().__getitem__(i + t // self.timedelta)

        # Load multiple future timestamps if specified.
        if self.multistep > 1:
            future_states = []
            for k in range(1, self.multistep + 1):
                future_states.append(super().__getitem__(i + k * t // self.timedelta))
            out["future_states"] = torch.stack(future_states, dim=0)

        if self.load_prev:
            out["prev_state"] = super().__getitem__(i - self.lead_time_months // self.timedelta)
        if normalize:
            out = self.normalize(out)

        # need to replace nans with mask_value
        out = {k: replace_nans(v, self.mask_value) if "state" in k else v for k, v in out.items()}
        return out

    def normalize(self, batch):
        device = list(batch.values())[0].device
        means = self.data_mean.to(device)
        stds = self.data_std.to(device)
        out = {k: ((v - means) / stds if "state" in k else v) for k, v in batch.items()}
        return out

    def denormalize(self, batch):
        device = list(batch.values())[0].device
        means = self.data_mean.to(device)
        stds = self.data_std.to(device)
        out = {k: (v * stds + means if "state" in k else v) for k, v in batch.items()}
        return out
