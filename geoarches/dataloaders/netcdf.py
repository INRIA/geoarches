import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

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


default_dimension_indexers = {
    "latitude": ("latitude", slice(None)),
    "longitude": ("longitude", slice(None)),
    "level": ("level", slice(None)),
    "time": ("time", slice(None)),
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
        dimension_indexers: Dict[str, Tuple[str, Any]] | None = None,
        filename_filter: Callable = lambda _: True,  # condition to keep file in dataset
        return_timestamp: bool = False,
        warning_on_nan: bool = True,
        limit_examples: int | None = None,
        interpolate_nans: bool = False,
    ):
        """
        Args:
        path: Single filepath or directory holding xarray files.
        variables: Dict holding xarray data variable lists mapped by their keys to be processed into tensordict.
            e.g. {surface: [data_var1, datavar2, ...], level: [...]}
            Used in convert_to_tensordict() to read data arrays in the xarray dataset and convert to tensordict.
        dimension_indexers: Dict of dimensions to select in xarray using Dataset.sel(). Also provides
            the dimension names to treat the xarray dataset as tensordict.
            Defaults to default_dimension_indexers.
            If not provided, defaults to selecting all data in all dimensions.
            First element is the dimension name in the xr dataset, second is the indexer used in Dataset.sel(indexer).
        filename_filter: To filter files within `path` based on filename.
        return_timestamp: Whether to return timestamp in __getitem__() along with tensordict.
        warning_on_nan: Whether to log warning if nan data found.
        limit_examples: Return set number of examples in dataset
        interpolate_nans: Whether to fill NaN values in the data with the mean of the
                            data across latitude and longitude dimensions. Defaults to True.
        """
        self.filename_filter = filename_filter
        self.variables = variables

        self.dimension_indexers = default_dimension_indexers.copy()
        self.dimension_indexers.update(dimension_indexers or {})

        self.time_dim_name = self.dimension_indexers["time"][0]
        self.latitude_dim_name = self.dimension_indexers["latitude"][0]
        self.longitude_dim_name = self.dimension_indexers["longitude"][0]
        self.level_dim_name = self.dimension_indexers["level"][0]

        # Separate indexers with slice and those without.
        indexers = {v[0]: v[1] for k, v in self.dimension_indexers.items() if k != "time"}
        self.slice_indexers = {k: v for k, v in indexers.items() if isinstance(v, slice)}
        self.other_indexers = {k: list(v) for k, v in indexers.items() if not isinstance(v, slice)}
        if not self.slice_indexers:
            self.slice_indexers = None
        if not self.other_indexers:
            self.other_indexers = None

        self.return_timestamp = return_timestamp
        self.warning_on_nan = warning_on_nan
        self.interpolate_nans = interpolate_nans

        # Workaround to avoid calling ds.sel() after ds.transponse() to avoid OOM.
        self.already_ran_index_selection = False

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
                file_stamps = [(fid, i, t) for (i, t) in enumerate(obs.coords[self.time_dim_name].to_numpy())]
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
        """
        Convert xarray dataset to tensordict.

        By default, it uses a mapping key from self.variables,
            e.g. {surface:[data_var1, data_var2, ...], level:[...]}
        """
        # Optionally select dimensions.
        if not self.already_ran_index_selection:
            if debug:
                print(xr_dataset)
                print(self.slice_indexers)
                print(self.other_indexers)

            # Apply sel for non-slice indexers with method and tolerance
            if self.other_indexers:
                xr_dataset = xr_dataset.sel(self.other_indexers, method="nearest", tolerance=1e-6)
            # Apply sel for slice indexers without method and tolerance
            if self.slice_indexers:
                xr_dataset = xr_dataset.sel(self.slice_indexers)

        self.already_ran_index_selection = False  # Reset for next call.

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
            self.cached_xrdataset = xr.open_dataset(self.files[file_id], **self.xr_options)
            self.cached_fileid = file_id

        obsi = self.cached_xrdataset.isel({self.time_dim_name: line_id})
        if interpolate_nans:
            obsi = obsi.fillna(
                value=obsi.mean(
                    dim=[
                        self.dimension_indexers["latitude"][0],
                        self.dimension_indexers["longitude"][0],
                    ],
                    skipna=True,
                )
            )

        tdict = self.convert_to_tensordict(obsi)

        if warning_on_nan:
            if any([x.isnan().any().item() for x in tdict.values()]):
                warnings.warn(f"NaN values detected in {file_id} {line_id} {self.files[file_id]}")

        if return_timestamp or self.return_timestamp:
            timestamp = self.cached_xrdataset.time[line_id].values.item()
            timestamp = torch.tensor(timestamp // 10**9, dtype=torch.int64)
            return tdict, timestamp

        return tdict
