import warnings
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import xarray as xr
from tensordict.tensordict import TensorDict
from tqdm import tqdm
from collections import OrderedDict
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
        dimension_indexers: Dict[str, list] = None,
        filename_filter: Callable = lambda _: True,  # condition to keep file in dataset
        return_timestamp: bool = False,
        warning_on_nan: bool = False,
        limit_examples: int | None = None,
        timestamp_key: Callable = lambda x: (x[-1]),
        interpolate_nans: bool = False,
        add_climate_indices: bool = False
    ):
        """
        Args:
        path: Single filepath or directory holding xarray files.
        variables: Dict holding xarray data variable lists mapped by their keys to be processed into tensordict.
            e.g. {surface: [data_var1, datavar2, ...], level: [...]}
            Used in convert_to_tensordict() to read data arrays in the xarray dataset and convert to tensordict.
        dimension_indexers: Dict of dimensions to select in xarray using Dataset.sel(dimension_indexers). Also provides
            the dimension names to treat the xarray dataset as tensordict.
            defaults to:
                dimension_indexers = {
                    'latitude': ('latitude', None),
                    'longitude': ('longitude', None),
                    'level': ('level', None),
                    'time': ('time', None)
                }
            If not provided, defaults to selecting all data in all dimensions.
            First value is the dimension name in xarray, second value is the indexer
            If None is used as the indexer, all coordinates in that dimension are used.
            To select a range, use a tuple (start, end, step) as the indexer.
        filename_filter: To filter files within `path` based on filename.
        return_timestamp: Whether to return timestamp in __getitem__() along with tensordict.
        warning_on_nan: Whether to log warning if nan data found.
        limit_examples: Return set number of examples in dataset
        interpolate_nans: Whether to fill NaN values in the data with the mean of the
                            data across latitude and longitude dimensions. Defaults to True.
        """
        self.filename_filter = filename_filter
        self.variables = variables

        self.dimension_indexers = dimension_indexers or default_dimension_indexers
        self.return_timestamp = return_timestamp
        self.warning_on_nan = warning_on_nan
        self.interpolate_nans = interpolate_nans
        self.add_climate_indices = add_climate_indices
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

        # file_extension = Path(self.files[0]).suffix
        # engine = engine_mapping[file_extension]
        # self.xr_options = dict(engine=engine, cache=True)

        self.timestamps = []

        for fid, f in tqdm(enumerate(self.files)):
            # with TensorDict.load_memmap(f) as obs:
            obs = TensorDict.load_memmap(f) 
            if(self.add_climate_indices):
                file_stamps = [(fid, i, t) for (i, t) in enumerate(obs['time']) if i > 23]
            else:
                file_stamps = [(fid, i, t) for (i, t) in enumerate(obs['time'])]

            self.timestamps.extend(file_stamps)


            if (
                limit_examples and len(self.timestamps) > limit_examples
            ):  # get fraction of full dataset
                print("Limiting number of examples loaded to", limit_examples)
                self.timestamps = self.timestamps[:limit_examples]
                break

        self.timestamps = sorted(self.timestamps, key=timestamp_key)  


        self.id2pt = dict(enumerate(self.timestamps))

        self.cached_xrdataset = None
        self.cached_fileid = None

        self._ozone_cache = {}
        self.ozone_cache_limit = 1 
        self._ozone_cache = OrderedDict()

    def _get_ozone_handle(self, exp, path):
        # 1. Check if we already have this file open
        if exp in self._ozone_cache:
            # Move to end (mark as recently used)
            self._ozone_cache.move_to_end(exp)
            return self._ozone_cache[exp]
        
        # 2. If cache is full, remove the oldest item to free memory
        if len(self._ozone_cache) >= self.ozone_cache_limit:
            oldest_exp, oldest_handle = self._ozone_cache.popitem(last=False)
            # Explicitly delete the handle to encourage garbage collection immediately
            del oldest_handle 
        
        # 3. Load the new file (mmap keeps it out of RAM until accessed)
        # Note: 15GB mmap uses Virtual Memory, not Physical RAM. It is safe to hold a few.
        handle = torch.load(path, weights_only=False, mmap=True)
        self._ozone_cache[exp] = handle
        return handle
    
    def __len__(self):
        return len(self.id2pt)
        
   
    def _ensure_cache(self):
        if getattr(self, "_file_cache", None) is None:
            self._file_cache = {}


    def __getitem__(self, i):
        file_id, line_id, timestamp = self.id2pt[i]

        file_path = self.files[file_id]

        self._ensure_cache()
        
        # --- OPTIMIZATION 1: Efficient Main Data Cache ---
        if file_path not in self._file_cache:
            self._file_cache[file_path] = TensorDict.load_memmap(file_path)
        data = self._file_cache[file_path]

        # # --- OPTIMIZATION 3: Lazy Concatenation ---
        # # Prepare the spatial forcings components
        spatial_forcings = data['spatial_forcings'][:, line_id]
        if(('ozone_0' not in self.variables['spatial_forcings'])): 
            # if(self.full_ozone):
            #     out['spatial_forcings'] = torch.concatenate([out['spatial_forcings'][-72:-66],out['spatial_forcings'][-66:]])  
            # else:
            spatial_forcings = spatial_forcings[:-6]
        if self.full_ozone:
            spatial_forcings = torch.cat([spatial_forcings,data['ozone'][line_id]])          
        if('level' in data.keys()):
            out = TensorDict({
                'lev': data['lev'][:, line_id],
                'level': data['level'][:, line_id][:len(self.variables['level'])], # Slice early
                'surface': data['surface'][:, line_id, None],
                'non_spatial_forcings': data['non_spatial_forcings'][:, line_id],
                'spatial_forcings': spatial_forcings 
            })
            out['level'] = out['level'][:len(self.variables['level'])]

        else:
            out = TensorDict({
                'lev': data['lev'][:, line_id],
                'surface': data['surface'][:, line_id, None],
                'non_spatial_forcings': data['non_spatial_forcings'][:, line_id],
                'spatial_forcings': data['spatial_forcings'][:, line_id] #override ozone logic here 
            })
        #ORDER is very important here. first check if aero is in, which we cut out of the middle if not. then we can remove the beginning which is ghg, finally we check if we remove the end, ozone
        if('load_ASNO3M' not in self.variables['spatial_forcings']):
            out['spatial_forcings'] = torch.concatenate([out['spatial_forcings'][:3],out['spatial_forcings'][9:]],dim=0)
        if('methane' not in self.variables['spatial_forcings']):
            out['spatial_forcings'] = out['spatial_forcings'][3:]
        if(len(self.variables['non_spatial_forcings']) == 0):
            out['non_spatial_forcings'] = torch.tensor([])  
        return out

