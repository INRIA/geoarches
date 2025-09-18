import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import xarray as xr
from tensordict.tensordict import TensorDict
from tqdm import tqdm

# Appropriate xarray engine for a given file extension
engine_mapping = {
    ".nc": "netcdf4",
    ".nc4": "netcdf4",
    ".h5": "h5netcdf",
    ".hdf5": "h5netcdf",
    ".grib": "cfgrib",
    ".zarr": "zarr",
}

# Mapping to rename dimensions if needed.
dimension_mapping = {
    "valid_time": "time",
    "pressure_level": "level",
    "lat": "latitude",
    "lon": "longitude",
}


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
        interpolate_nans: bool = False,
    ):
        """Initializes.

        Args:
        path: Single filepath or directory holding xarray files.
        variables: Dict holding xarray data variable lists mapped by their keys to be processed into tensordict.
            e.g. {surface: [data_var1, datavar2, ...], level: [...]}
            Used in convert_to_tensordict() to read data arrays in the xarray dataset and convert to tensordict.
        dimension_indexers: Dict of dimensions to select in xarray using Dataset.sel().
        filename_filter: To filter files within `path` based on filename.
        return_timestamp: Whether to return timestamp in __getitem__() along with tensordict.
        warning_on_nan: Whether to log warning if nan data found.
        limit_examples: Return set number of examples in dataset
        interpolate_nans: Whether to fill NaN values in the data with the mean of the
                            data across latitude and longitude dimensions. Defaults to True.
        """
        self.filename_filter = filename_filter
        self.variables = variables

        # Separate indexers with slice and those without.
        self.dimension_indexers = dimension_indexers or {}

        self.return_timestamp = return_timestamp
        self.warning_on_nan = warning_on_nan
        self.interpolate_nans = interpolate_nans

        self.transpose_order = transpose_order

        if not Path(path).exists():
            raise ValueError("Path does not exist:", path)

        if Path(path).is_file() and "." in path.split("/")[-1]:
            print("Single file detected. Loading single file ", path)
            self.files = [path]
        else:
            files = list(Path(path).glob("*"))
            if len(files) == 0:
                raise ValueError("No files found under path:", path)

            self.files = sorted(
                [str(x) for x in files if filename_filter(x.name)],
                key=lambda x: x.replace("_6h", "_06h").replace("_0h", "_00h"),
            )
            if len(self.files) == 0:
                raise ValueError("filename_filter filtered all files under path:", path)

        file_extension = Path(self.files[0]).suffix
        engine = engine_mapping[file_extension]
        self.xr_options = dict(engine=engine, cache=True)

        self.timestamps = []

        for fid, f in tqdm(enumerate(self.files)):
            with xr.open_dataset(f, **self.xr_options) as obs:
                time_dim_name = "time" if "time" in obs.coords else "valid_time"
                file_stamps = [
                    (fid, i, t) for (i, t) in enumerate(obs.coords[time_dim_name].to_numpy())
                ]
                self.timestamps.extend(file_stamps)
            if (
                limit_examples and len(self.timestamps) > limit_examples
            ):  # get fraction of full dataset
                print("Limiting number of examples loaded to", limit_examples)
                self.timestamps = self.timestamps[:limit_examples]
                break

        self.timestamps = sorted(self.timestamps, key=lambda x: x[-1])  # sort by timestamp
        self.id2pt = dict(enumerate(self.timestamps))

        self.cached_xrdataset = None
        self.cached_fileid = None

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
        for k, v in self.dimension_indexers.items():
            if k == "time":
                continue
            params = dict(method="nearest", tolerance=1e-6) if not isinstance(v, slice) else {}
            v = list(v) if isinstance(v, tuple) else v
            xr_dataset = xr_dataset.sel({k: v}, **params)

        xr_dataset = xr_dataset.transpose(*self.transpose_order)
        np_arrays = {
            key: xr_dataset[list(variables)].to_array().to_numpy()
            for key, variables in self.variables.items()
        }

        tdict = TensorDict(
            {key: torch.from_numpy(np_array).float() for key, np_array in np_arrays.items()}
        )

        return tdict
    
    def __getitem__(self, i, return_timestamp=False, interpolate_nans=None, warning_on_nan=None):
        interpolate_nans = interpolate_nans or self.interpolate_nans
        warning_on_nan = warning_on_nan or self.warning_on_nan

        file_id, line_id, timestamp = self.id2pt[i]

        if self.cached_fileid != file_id:
            if self.cached_xrdataset is not None:
                self.cached_xrdataset.close()
            xrdataset = xr.open_dataset(self.files[file_id], **self.xr_options)
            # postprocess some coord names
            for old_name, new_name in dimension_mapping.items():
                if old_name in xrdataset.dims:
                    xrdataset = xrdataset.swap_dims({old_name: new_name})
                if old_name in xrdataset.coords:
                    xrdataset = xrdataset.assign_coords(**{new_name: xrdataset.coords[old_name]})

            self.cached_fileid = file_id
            self.cached_xrdataset = xrdataset

        obsi = self.cached_xrdataset.isel({self.time_dim_name: line_id})
        if interpolate_nans: 
            obsi = obsi.fillna(
                value=obsi.mean(
                    dim=["latitude", "longitude"],
                    skipna=True,
                )
            )

        tdict = self.convert_to_tensordict(obsi)

        if warning_on_nan:
            if any([x.isnan().any().item() for x in tdict.values()]):
                warnings.warn(f"NaN values detected in {file_id} {line_id} {self.files[file_id]}")

        if return_timestamp or self.return_timestamp:
            timestamp = self.cached_xrdataset["time"][line_id].values.item()
            timestamp = torch.tensor(timestamp // 10**9, dtype=torch.int64)
            return tdict, timestamp

        return tdict
