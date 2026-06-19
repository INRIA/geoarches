import functools
import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import xarray as xr
from tensordict.tensordict import TensorDict
from tqdm import tqdm

from geoarches.dataloaders import nan_util
from geoarches.utils.tensordict_utils import tensordict_apply

# Appropriate xarray engine for a given file extension
engine_mapping = {
    ".nc": "netcdf4",
    ".nc4": "netcdf4",
    ".h5": "h5netcdf",
    ".hdf5": "h5netcdf",
    ".grib": "cfgrib",
    ".zarr": "zarr",
}


def optionally_rename_dimensions(xrdataset):
    """Rename dimensions if needed."""
    dimension_mapping = {
        "valid_time": "time",
        "pressure_level": "level",
        "lat": "latitude",
        "lon": "longitude",
    }
    for old_name, new_name in dimension_mapping.items():
        if old_name in xrdataset.dims:
            xrdataset = xrdataset.swap_dims({old_name: new_name})
        if old_name in xrdataset.coords:
            xrdataset = xrdataset.assign_coords(**{new_name: xrdataset.coords[old_name]})
    return xrdataset


def select_dimensions(xrdataset, dimension_indexers):
    """Select by dimensions_indexers from xarray dataset."""
    for k, v in dimension_indexers.items():
        if k == "time":
            continue
        params = dict(method="nearest", tolerance=1e-6) if not isinstance(v, slice) else {}
        v = list(v) if isinstance(v, tuple) else v
        xrdataset = xrdataset.sel({k: v}, **params)
    return xrdataset


class XarrayDataset(torch.utils.data.Dataset):
    """
    dataset to read a list of xarray files and iterate through it by timestamp.
    constraint: it should be indexed by at least one dimension named "time".

    Child classes that inherit this class, should implement convert_to_tensordict()
    which converts an xarray dataset into a tensordict (to feed into the model).
    """

    def __init__(
        self,
        path: str,
        variables: Dict[str, List[str]],
        dimension_indexers: Dict[str, Any] | None = None,
        transpose_order: tuple[Any, ...] = (..., "level", "latitude", "longitude"),
        filename_filter: Callable = lambda _: True,  # condition to keep file in dataset
        return_timestamp: bool = False,
        warning_on_nan: bool = True,
        limit_examples: int | None = None,
        force_rebuild_index: bool = False,
        interpolate_nans: nan_util.NanInterpolationMethod | None = None,
    ):
        """Initializes.

        Args:

        path: Single filepath or directory holding xarray files.
        variables: Dict holding xarray data variable lists mapped by their keys to
          be processed into tensordict.
            e.g. {surface: [data_var1, datavar2, ...], level: [...]}
            Used in convert_to_tensordict() to read data arrays in the xarray
              dataset and convert to tensordict.
        dimension_indexers: Dict of dimensions to select in xarray using
          Dataset.sel().
        filename_filter: To filter files within `path` based on filename.
        return_timestamp: Whether to return timestamp in __getitem__() along with
          tensordict.
        warning_on_nan: Whether to log warning if nan data found.
        limit_examples: Return set number of examples in dataset. Deprecated. Use
          timestamps instead.
        interpolate_nans: Whether to fill NaN values in the data. Defaults to no
          interpolation.
        force_rebuild_index: Whether to rebuild the index of timestamps for all
          files under path.
        This can be useful is the files in directory have changed.
        """
        self.filename_filter = filename_filter
        self.variables = variables
        self.path = Path(path)

        self.dimension_indexers = dict(dimension_indexers) if dimension_indexers else {}

        self.return_timestamp = return_timestamp
        self.warning_on_nan = warning_on_nan
        if interpolate_nans and interpolate_nans not in list(nan_util.NanInterpolationMethod):
            raise ValueError(
                f"Invalid interpolate_nans: {interpolate_nans}. "
                f"Valid options are: {list(nan_util.NanInterpolationMethod)}"
            )
        self.interpolate_nans = interpolate_nans

        self.transpose_order = transpose_order

        if not self.path.exists():
            raise ValueError("Path does not exist:", path)

        if self.path.is_file():
            self.xr_options = dict(engine=engine_mapping[self.path.suffix], cache=True)
            fname = self.path.name
            self.path = self.path.parent
            self.timestamps = self.get_file_timestamps(fname)
        elif not list(self.path.glob("*")):
            raise ValueError("No files found under path:", path)
        else:
            # define opening options
            file_extensions = [
                x.suffix for x in list(self.path.glob("*")) if x.suffix in engine_mapping.keys()
            ]
            engine = engine_mapping[file_extensions[0]]
            self.xr_options = dict(engine=engine, cache=True)

            # optionally build index and load
            if force_rebuild_index or not self.index_path.exists():
                self.build_index()
            self.timestamps = self.load_index()

            # for backward compatibility, we filter timestamps by filename here.
            # TODO(geco): we should filter by timestamp instead.
            self.timestamps = [x for x in self.timestamps if self.filename_filter(x[0])]

        self.id2pt = dict(enumerate(self.timestamps))

        self.cached_xrdataset = None
        self.cached_fname = None

        # for backwards comp.
        # TODO(geco): remove
        self.files = list(set([self.path / x[0] for x in self.timestamps]))

    @property
    def index_path(self) -> Path:
        return self.path.parent / (self.path.name + "_index.json")

    @functools.cached_property
    def nptime_to_loc_info(self) -> Dict[str, Dict[str, Any]]:
        return {
            nptime: dict(fname=fname, line_id=i, dataset_index=ds_idx)
            for ds_idx, (fname, i, nptime) in enumerate(self.timestamps)
        }

    def get_file_timestamps(self, fname: str) -> List[Tuple[str, int, np.datetime64]]:
        with xr.open_dataset(self.path / fname, **self.xr_options) as obs:
            time_dim_name = "time" if "time" in obs.coords else "valid_time"
            file_stamps = [
                (fname, i, t) for (i, t) in enumerate(obs.coords[time_dim_name].to_numpy())
            ]
        return file_stamps

    def load_index(self) -> List[Tuple[str, int, np.datetime64]]:
        with self.index_path.open("r") as f:
            timestamps = json.load(f)
        timestamps = [
            (fname, i, np.datetime64(str_timestamp)) for (fname, i, str_timestamp) in timestamps
        ]
        return timestamps

    def build_index(self) -> None:
        """Build index of timestamps for all files under path.

        Saves under json
        """
        # Ensure we only keep files with the expected extension.
        # In particular, this ensures we filter out temporary files which have
        # additional .tmp.xxx suffixes.
        files = (x for x in self.path.glob("*") if x.suffix in engine_mapping.keys())

        self.timestamps = []

        for fpath in tqdm(files):
            self.timestamps.extend(self.get_file_timestamps(fpath.name))

        self.timestamps = sorted(self.timestamps, key=lambda x: x[-1])  # sort by timestamp

        # check for duplicates
        timestamps = [t[-1] for t in self.timestamps]
        num_timestamps = len(timestamps)
        num_unique_timestamps = len(set(timestamps))
        if num_timestamps != num_unique_timestamps:
            raise ValueError(
                f"Timestamps are not unique. Found {num_timestamps} timestamps "
                f"but only {num_unique_timestamps} unique timestamps."
            )

        # convert to str and dump to json
        timestamps = [(fname, i, str(t)) for (fname, i, t) in self.timestamps]
        with self.index_path.open("w") as f:
            json.dump(timestamps, f)

    def set_timestamp_bounds(self, low, high, debug=False):
        """Filter timestamps loaded from dataloader between bounds.

        If low or high is None, only filter in one direction.

        Args:
            low: lower bound, inclusive. Set to None to not filter by lower bound.
            high: upper bound, exclusive. Set to None to not filter by upper bound.
        """
        original_length = len(self.timestamps)

        if low and high:
            self.timestamps = [
                x for x in self.timestamps if low <= x[-1].astype("datetime64[s]") < high
            ]
        elif low:
            self.timestamps = [x for x in self.timestamps if low <= x[-1].astype("datetime64[s]")]
        elif high:
            self.timestamps = [x for x in self.timestamps if x[-1].astype("datetime64[s]") < high]
        if debug:
            print(
                f"Filtered timestamps from {original_length} to {len(self.timestamps)} examples: "
                f"{self.timestamps[0][-1].astype('datetime64[s]')} to {self.timestamps[-1][-1].astype('datetime64[s]')}."
            )
        self.id2pt = dict(enumerate(self.timestamps))

    def __len__(self):
        return len(self.id2pt)

    def convert_to_tensordict(self, xr_dataset, debug=False):
        """Convert xarray dataset to tensordict.

        By default, it uses a mapping key from self.variables,
            e.g. {surface:[data_var1, data_var2, ...], level:[...]}
        """
        # Optionally select dimensions.
        if debug:
            print(xr_dataset)
            print(self.dimension_indexers)

        # Apply sel for indexers
        xr_dataset = select_dimensions(xr_dataset, self.dimension_indexers)

        xr_dataset = xr_dataset.transpose(*self.transpose_order)
        np_arrays = {
            key: xr_dataset[list(variables)].to_array().to_numpy()
            for key, variables in self.variables.items()
        }

        tdict = TensorDict(
            {key: torch.from_numpy(np_array).float() for key, np_array in np_arrays.items()}
        )

        return tdict

    def select_from_nptime(self, nptime: np.datetime64) -> Tuple[str, int]:
        """Get file name and line id for a given timestamp string."""
        if nptime not in self.nptime_to_loc_info:
            raise ValueError(f"Timestamp {nptime} not found in dataset.")
        idx = self.nptime_to_loc_info[nptime]["dataset_index"]
        return self[idx]

    def __getitem__(
        self,
        i,
        return_nan_mask=False,
        return_timestamp=False,
        interpolate_nans=None,
        warning_on_nan=None,
    ):
        interpolate_nans = self.interpolate_nans if interpolate_nans is None else interpolate_nans
        warning_on_nan = self.warning_on_nan if warning_on_nan is None else warning_on_nan

        fname, line_id, _ = self.id2pt[i]

        if self.cached_fname != fname:
            if self.cached_xrdataset is not None:
                self.cached_xrdataset.close()
            xrdataset = xr.open_dataset(self.path / fname, **self.xr_options)
            # postprocess some coord names
            xrdataset = optionally_rename_dimensions(xrdataset)

            self.cached_fname = fname
            self.cached_xrdataset = xrdataset

        obsi = self.cached_xrdataset.isel(time=line_id)  # pytype: disable=attribute-error

        if return_nan_mask:
            tdict_before_interpolation = self.convert_to_tensordict(obsi)
            nan_mask = tensordict_apply(torch.isnan, tdict_before_interpolation)

        obsi = nan_util.pre_norm_interpolate_nans(obsi, interpolate_nans)
        tdict = self.convert_to_tensordict(obsi)

        if warning_on_nan and interpolate_nans != nan_util.NanInterpolationMethod.NONE:
            if any([x.isnan().any().item() for x in tdict.values()]):
                warnings.warn(f"NaN values detected in {fname} {line_id}")

        if return_nan_mask:
            return tdict, nan_mask

        if return_timestamp or self.return_timestamp:
            timestamp = self.cached_xrdataset["time"][line_id].values.item()
            timestamp = torch.tensor(timestamp // 10**9, dtype=torch.int64)
            return tdict, timestamp

        return tdict
