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


class NetcdfDataset(torch.utils.data.Dataset):
    """
    dataset to read a list of netcdf files and iterate through it.
    constraint: it should be indexed by at least one dimension "time".
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
            path: Single filepath or directory holding files.
            variables: Dict holding variable lists mapped by their keys to be processed into tensordict.
                e.g. {surface:[...], level:[...]
            dimension_indexers: Dict of dimensions to select using Dataset.sel(dimension_indexers).
            filename_filter: To filter files within data directory based on filename.
            engine: xarray dataset backend.
            file_extension: Expected file extension for data file(s) (ie. '.zarr').
            return_timestamp: Whether to return timestamp in __getitem__() along with tensordict.
            limit_examples: Return set number of examples in dataset
        """
        self.filename_filter = filename_filter
        self.variables = variables
        self.dimension_indexers = dimension_indexers
        self.return_timestamp = return_timestamp
        self.warning_on_nan = warning_on_nan
        # Workaround to avoid calling ds.sel() after ds.transponse() to avoid OOM.
        self.already_ran_index_selection = False

        if Path(path).is_file():
            self.files = [path]
        else:
            files = list(Path(path).glob("*"))
            if len(files) == 0:
                assert ValueError("No files found under path:", path)

            self.files = sorted(
                [str(x) for x in files if filename_filter(x.name)],
                key=lambda x: x.replace("6h", "06h").replace("0h", "00h"),
            )
            if len(self.files) == 0:
                assert ValueError("filename_filter filtered all files.")

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
        how to convert to tensordict.
        by default it uses a mapping key: variables,
        e.g. {surface:[...], level:[...]
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
