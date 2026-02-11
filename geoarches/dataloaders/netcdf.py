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
            file_stamps = [(fid, i, t) for (i, t) in enumerate(obs['time'])]
            self.timestamps.extend(file_stamps)
            if (
                limit_examples and len(self.timestamps) > limit_examples
            ):  # get fraction of full dataset
                print("Limiting number of examples loaded to", limit_examples)
                self.timestamps = self.timestamps[:limit_examples]
                break

        self.timestamps = sorted(self.timestamps, key=timestamp_key)  # sort by timestamp
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
        
   
    # def convert_to_tensordict(self, xr_dataset,file_path):
    # def convert_to_tensordict(self, file_path=None):

    #     """
    #     Convert xarray dataset to tensordict.

    #     By default, it uses a mapping key from self.variables,
    #         e.g. {surface:[data_var1, data_var2, ...], level:[...]}
    #     """
    #     # Optionally select dimensions.
    #     # if self.dimension_indexers and not self.already_ran_index_selection:
    #     #     indexers = {v[0]: v[1] for k, v in self.dimension_indexers.items() if k != "time"}

    #     #     xr_dataset = xr_dataset.sel(**indexers)
    #     # self.already_ran_index_selection = False  # Reset for next call.
    #     with Dataset(file_path, "r") as nc:
    #         # Convert coordinate selections to indices
    #         indexers = {}
    #         if self.dimension_indexers:
    #             indexers = select_by_coords(nc, dimension_indexers)
    
    #         tdict_data = {}
    #         for key, vars_ in variables.items():
    #             if vars_:
    #                 arrays = []
    #                 for var in vars_:
    #                     var_data = nc.variables[var]
    
    #                     # Build slice object for all dimensions
    #                     slices = []
    #                     for dim in var_data.dimensions:
    #                         if dim in indexers:
    #                             slices.append(indexers[dim])
    #                         elif dim == 'line':  # example for your line_id dimension
    #                             slices.append(line_id)
    #                         else:
    #                             slices.append(slice(None))
    
    #                     data_slice = var_data[tuple(slices)]
    #                     arrays.append(torch.tensor(data_slice, dtype=torch.float32))
    #                 tdict_data[key] = torch.stack(arrays)
    #             else:
    #                 tdict_data[key] = torch.empty((0,), dtype=torch.float32)

    #     return TensorDict(tdict_data)
    #     # Make np arrays for each key and make an empty array if no variables for this list. Needed for running experiments with different variable sets
    #     # np_arrays = {}
    #     # for key, variables in self.variables.items():
    #     #     if variables:  # non-empty
    #     #         np_arrays[key] = xr_dataset[list(variables)].to_array().to_numpy()
    #     #     else:  # empty list -> create an empty array
    #     #         np_arrays[key] = np.empty((0,))

    #     # tdict = TensorDict(
    #     #     {key: torch.from_numpy(np_array).float() for key, np_array in np_arrays.items()}
    #     # )

    #     from netCDF4 import Dataset
    #     with Dataset(file_path, "r") as nc:
    #         tdict_data = {}
    #         for key, variables in self.variables.items():
    #             if variables:
    #                 arrays = [torch.tensor(nc[var][line_id], dtype=torch.float32) for var in variables]
    #                 tdict_data[key] = torch.stack(arrays)
    #             else:
    #                 tdict_data[key] = torch.empty((0,), dtype=torch.float32)
    #     return TensorDict(tdict_data)
    #     #     tdict_data = TensorDict()
        
    #     # for key, variables in self.variables.items():
    #     #     if variables:  # non-empty
    #     #         xr_data = xr_dataset[list(variables)].to_array()
    #     #         data = xr_data.data  # now this is in-memory numpy array

    #     #         # convert directly to torch tensor
    #     #         tdict_data[key] = torch.as_tensor(data, dtype=torch.float32)
    #     #     else:  # empty list
    #     #         tdict_data[key] = torch.empty((0,), dtype=torch.float32)
        
    #     # tdict = TensorDict(tdict_data)


    #     # import xarray.backends.file_manager as fm
        
    #     # def cleanup_xarray():
    #     #     fm.CachingFileManager._caches.clear()
    #     #     gc.collect()

    #     # cleanup_xarray()






        
        # np_array = None
        # xr_dataset.close()
        # # import netCDF4
        # # netCDF4.Dataset._close(xr_dataset)
        # # import pdb; pdb.set_trace()
        # return tdict


    # def __getitem__2(self, i, return_timestamp=False, interpolate_nans=None, warning_on_nan=None):
    #     import gc
    #     file_id, line_id, timestamp = self.id2pt[i]

    #     file_path=self.files[file_id]
        
    #     tdict_data = TensorDict()
        
    #     with Dataset(file_path, "r") as nc:
    #         # Cache indices
    #         if not hasattr(self, '_index_cache'):
    #             self._index_cache = {}
            
    #         if file_path not in self._index_cache:
    #             cache = {}
    #             if 'plev' in self.dimension_indexers and 'plev' in nc.variables:
    #                 cache['plev'] = [list(nc['plev'][:]).index(val) for val in self.dimension_indexers['plev'][1]]
    #             if 'lev' in self.dimension_indexers and 'lev' in nc.variables:
    #                 lev_vals = [round(float(x), 3) for x in nc['lev'][:]]
    #                 cache['lev'] = [lev_vals.index(round(float(val), 3)) for val in self.dimension_indexers['lev'][1]]
    #             self._index_cache[file_path] = cache
            
    #         indices = self._index_cache[file_path]
            
    #         for key, vars_ in self.variables.items():
    #             tensor_list = []
                
    #             for var in vars_:
    #                 # CRITICAL: Load entire slice into memory first
    #                 if key == 'surface':
    #                     # Load to numpy, then immediately copy and convert
    #                     temp = np.array(nc[var][line_id], dtype=np.float32)
    #                     tensor = torch.from_numpy(temp).unsqueeze(0)
    #                 elif key == 'level':
    #                     temp = np.array(nc[var][line_id, indices['plev'], :, :], dtype=np.float32)
    #                     tensor = torch.from_numpy(temp)
    #                 elif key == 'lev':
    #                     temp = np.array(nc[var][line_id, indices['lev'], :], dtype=np.float32)
    #                     tensor = torch.from_numpy(temp)
    #                 elif key in ['spatial_forcings', 'non_spatial_forcings', 'ozone']:
    #                     temp = np.array(nc[var][line_id], dtype=np.float32)
    #                     tensor = torch.from_numpy(temp)
                    
    #                 # Detach from any computation graph
    #                 tensor = tensor.detach().clone()
    #                 tensor_list.append(tensor)
                    
    #                 del temp, tensor
                
    #             tdict_data[key] = torch.stack(tensor_list).detach().clone()
    #             del tensor_list
        
    #     # Force cleanup
    #     gc.collect()
        
    #     return tdict_data

    def _ensure_cache(self):
        if getattr(self, "_file_cache", None) is None:
            self._file_cache = {}

    # def __getitem__(self, i, return_timestamp=False, interpolate_nans=None, warning_on_nan=None):
    #     # import gc
        
    #     file_id, line_id, timestamp = self.id2pt[i]
    #     # file_id, line_id, timestamp = 0,0,0
    #     file_path = self.files[file_id]
    #     # data = TensorDict.load_memmap(file_path)
        
    #     # out = TensorDict({'lev':data['lev'][:,line_id],
    #     #           'level':data['level'][:,line_id],
    #     #           'surface':data['surface'][:,line_id,None],
    #     #           'non_spatial_forcings':data['non_spatial_forcings'][:,line_id],
    #     #           'spatial_forcings':torch.cat([data['spatial_forcings'][:,line_id],data['ozone'][0,line_id]]) if 'ozone' in data.keys() else data['spatial_forcings'][:,line_id]})
    #     # return out

    #     self._ensure_cache()
    #     if file_path not in self._file_cache:
    #         # open once per process (and per worker)
    #         self._file_cache[file_path] = TensorDict.load_memmap(file_path)
    #     data = self._file_cache[file_path]
    #     if(self.full_ozone):
    #         exp = file_path.split("/")[-1].split(".")[0].split("_")[1]
    #         if(exp == 'piControl'):
    #             ozone = torch.load(f'/scratch/gclyne/reference_data/ozone_historical.pt',weights_only=False,mmap=True)
    #             data['ozone'] =  ozone[line_id%12].clone()

    #         else:
    #             ozone = torch.load(f'/scratch/gclyne/reference_data/ozone_{exp}.pt',weights_only=False,mmap=True)
    #             data['ozone'] =  ozone[line_id].clone()

    #         # ozone[line_id] 
        
    #     out = TensorDict({
    #         'lev': data['lev'][:, line_id],
    #         'level': data['level'][:, line_id],
    #         'surface': data['surface'][:, line_id, None],
    #         'non_spatial_forcings': data['non_spatial_forcings'][:, line_id],
    #         'spatial_forcings':
    #             torch.cat([data['spatial_forcings'][:, line_id], data['ozone']])
    #             if 'ozone' in data.keys()
    #             else data['spatial_forcings'][:, line_id]
    #     })
    #     if('methane' not in self.variables['spatial_forcings']):
    #         out['spatial_forcings'] = out['spatial_forcings'][3:]
    #     if('ozone_0' not in self.variables['spatial_forcings']):
    #         out['spatial_forcings'] = out['spatial_forcings'][:-6]
    #     if('load_ASNO3M' not in self.variables['spatial_forcings']):
    #         out['spatial_forcings'] = torch.concatenate([out['spatial_forcings'][:3],out['spatial_forcings'][9:]],dim=0)
    #     if(len(self.variables['non_spatial_forcings']) == 0):
    #         out['non_spatial_forcings'] = torch.tensor([])  
    #     out['level'] = out['level'][:len(self.variables['level'])]
    #     return out

    def __getitem__(self, i):
        file_id, line_id, timestamp = self.id2pt[i]
        file_path = self.files[file_id]

        self._ensure_cache()
        
        # --- OPTIMIZATION 1: Efficient Main Data Cache ---
        if file_path not in self._file_cache:
            self._file_cache[file_path] = TensorDict.load_memmap(file_path)
        data = self._file_cache[file_path]

        # --- OPTIMIZATION 2: Efficient Ozone Caching ---
        # ozone_tensor = None
        # if self.full_ozone:
        #     # Move string parsing here (only runs once per file, not per sample if we cached by file_path)
        #     # However, since 'exp' depends on file_path, let's cache the ozone handles based on 'exp'
            
        #     # Extract experiment name
        #     # We can optimize this by pre-calculating exp per file_id in __init__, but this is okay for now
        #     exp = file_path.split("/")[-1].split(".")[0].split("_")[1] # THIS IS BAD FOR PERFORMANCE
            

            
        #     # 4. Reuse the EXISTING handle
        #     # ozone_source = self.ozone_handles[ozone_path]
        #     # if exp not in self._ozone_cache:
        #     # Load ONCE and store the handle
        #     if exp == 'piControl':
        #         path = f'{self.forcings_path}/ozone_historical.pt'
        #     else:
        #         path = f'{self.forcings_path}/ozone_{exp}.pt'
            
        #     #     # Keep the mmap open in memory
        #     #     self._ozone_cache[exp] = torch.load(path, weights_only=False, mmap=True)
                
        #     # Access the cached handle directly
        #     # ozone_source = self._ozone_cache[exp]
        #     # ozone_source = self._get_ozone_handle(exp, path)
        #     ozone_source = torch.load(path, weights_only=False, mmap=True)
        #     # Handle index mapping
        #     idx = line_id % 12 if exp == 'piControl' else line_id
        #     ozone_tensor = ozone_source[idx].clone() # No .clone() yet, delay copy
        
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

