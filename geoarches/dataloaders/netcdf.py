import warnings
from pathlib import Path
from typing import Callable, Dict, List

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
        dimension_indexers: Dict[str, list] | None = None,
        filename_filter: Callable = lambda _: True,  # condition to keep file in dataset
        return_timestamp: bool = False,
        warning_on_nan: bool = False,
        limit_examples: int | None = None,
    ):
        """
        Args:
            path: Single filepath or directory holding xarray files.
            variables: Dict holding xarray data variable lists mapped by their keys to be processed into tensordict.
                e.g. {surface: [data_var1, datavar2, ...], level: [...]}
                Used in convert_to_tensordict() to read data arrays in the xarray dataset and convert to tensordict.
            dimension_indexers: Dict of dimensions to select in xarray using Dataset.sel(dimension_indexers).
            filename_filter: To filter files within `path` based on filename.
            return_timestamp: Whether to return timestamp in __getitem__() along with tensordict.
            warning_on_nan: Whether to log warning if nan data found.
            limit_examples: Return set number of examples in dataset
        """
        self.filename_filter = filename_filter
        self.variables = variables
        self.dimension_indexers = dimension_indexers
        self.return_timestamp = return_timestamp
        self.warning_on_nan = warning_on_nan
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
                key=lambda x: x.replace("6h", "06h").replace("0h", "00h"),
            )
            if len(self.files) == 0:
                raise ValueError("filename_filter filtered all files.")

        file_extension = Path(self.files[0]).suffix
        engine = engine_mapping[file_extension]
        self.xr_options = dict(engine=engine, cache=True)

        self.timestamps = []

        for fid, f in tqdm(enumerate(self.files)):
            with xr.open_dataset(f, **self.xr_options) as obs:
                file_stamps = [(fid, i, t) for (i, t) in enumerate(obs.time.to_numpy())]
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

    def set_timestamp_bounds(self, low, high):
        self.timestamps = [
            x for x in self.timestamps if low <= x[-1].astype("datetime64[s]") < high
        ]
        self.id2pt = dict(enumerate(self.timestamps))

    def __len__(self):
        return len(self.id2pt)

    def convert_to_tensordict(self, xr_dataset):
        """
        Convert xarray dataset to tensordict.

        By default, it uses a mapping key from self.variables,
            e.g. {surface:[data_var1, data_var2, ...], level:[...]}
        """
        # Optionally select dimensions.
        if self.dimension_indexers and not self.already_ran_index_selection:
            xr_dataset = xr_dataset.sel(self.dimension_indexers)
        self.already_ran_index_selection = False  # Reset for next call.

        np_arrays = {
            key: xr_dataset[list(variables)].to_array().to_numpy()
            for key, variables in self.variables.items()
        }
        tdict = TensorDict(
            {key: torch.from_numpy(np_array).float() for key, np_array in np_arrays.items()}
        )
        return tdict

    def __getitem__(self, i, return_timestamp=False):
        file_id, line_id, timestamp = self.id2pt[i]

        if self.cached_fileid != file_id:
            if self.cached_xrdataset is not None:
                self.cached_xrdataset.close()
            self.cached_xrdataset = xr.open_dataset(self.files[file_id], **self.xr_options)
            self.cached_fileid = file_id

        obsi = self.cached_xrdataset.isel(time=line_id)
        tdict = self.convert_to_tensordict(obsi)

        if self.warning_on_nan:
            if any([x.isnan().any().item() for x in tdict.values()]):
                warnings.warn(f"NaN values detected in {file_id} {line_id} {self.files[file_id]}")

        if return_timestamp or self.return_timestamp:
            timestamp = self.cached_xrdataset.time[line_id].values.item()
            timestamp = torch.tensor(timestamp // 10**9, dtype=torch.int32)
            return tdict, timestamp
        return tdict
