import warnings
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import xarray as xr
from tensordict.tensordict import TensorDict
from tqdm import tqdm
import psutil
import os
from netCDF4 import Dataset
import zarr
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
# def select_by_coords(nc, dimension_indexers):
#     # dimension_indexers: dict like {'lat': 10.0, 'lon': 20.0}
#     # returns a dict {dim_name: index} to be used for slicing

#     selected_indices = {}
#     for dim_name, coord_value in dimension_indexers.items():
#         coord_var = nc.variables[dim_name]
#         coord_data = coord_var[:]
#         print('coord_var', coord_var)
#         print('coord_data',coord_data)
#         print('coord_value',coord_value)
#         # Find index where coord_data == coord_value (or closest)
#         # For floats, do approximate matching:
#         index = (abs(coord_data - coord_value)).argmin()
#         selected_indices[dim_name] = index

#     return selected_indices

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

        file_extension = Path(self.files[0]).suffix
        # engine = engine_mapping[file_extension]
        # self.xr_options = dict(engine=engine, cache=True)

        self.timestamps = []
        # for fid, f in tqdm(enumerate(self.files)):
        #     # Open Zarr store
        #     zr = zarr.open(f, mode='r')
            
        #     # Get time values from the Zarr array
        #     time_values = zr['time'][:]
        #     file_stamps = [(fid, i, t) for (i, t) in enumerate(time_values)]
        #     self.timestamps.extend(file_stamps)
            
        #     if (
        #         limit_examples and len(self.timestamps) > limit_examples
        #     ):  # get fraction of full dataset
        #         print("Limiting number of examples loaded to", limit_examples)
        #         self.timestamps = self.timestamps[:limit_examples]
        #         break


        
        # for fid, f in tqdm(enumerate(self.files)):
        #     with xr.open_dataset(f, **self.xr_options) as obs:
        #         file_stamps = [(fid, i, t) for (i, t) in enumerate(obs.time.to_numpy())]
        #         self.timestamps.extend(file_stamps)
        #     if (
        #         limit_examples and len(self.timestamps) > limit_examples
        #     ):  # get fraction of full dataset
        #         print("Limiting number of examples loaded to", limit_examples)
        #         self.timestamps = self.timestamps[:limit_examples]
        #         break
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
                # Open with consolidated metadata (much faster)
        # zr = zarr.open_consolidated(self.files[0], mode='r') if zarr.consolidate_metadata else zarr.open(file_path, mode='r')
        
        # # PRE-COMPUTE indices outside the loop (CRITICAL!)
        # lev_indices_temp = [round(float(x), 3) for x in zr['lev'][:]]
        # self.lev_indices_cache = {}
        # if 'lev' in self.dimension_indexers:
        #     self.lev_indices_cache['lev'] = [lev_indices_temp.index(round(float(val), 3)) 
        #                                   for val in self.dimension_indexers['lev'][1]]
        
        # self.plev_indices_cache = {}
        # if 'plev' in self.dimension_indexers:
        #     plev_list = list(zr['plev'][:])
        #     self.plev_indices_cache['plev'] = [plev_list.index(val) 
        #                                    for val in self.dimension_indexers['plev'][1]]
    # def set_timestamp_bounds(self, low, high, debug=False):
    #     """Filter timestamps loaded from dataloader between bounds.

    #     If low or high is None, only filter in one direction.

    #     Args:
    #         low: lower bound, inclusive. Set to None to not filter by lower bound.
    #         high: upper bound, exclusive. Set to None to not filter by upper bound.
    #     """
    #     original_length = len(self.timestamps)

    #     if low and high:
    #         self.timestamps = [
    #             x for x in self.timestamps if low <= x[-1].astype("datetime64[s]") < high
    #         ]
    #     elif low:
    #         self.timestamps = [x for x in self.timestamps if low <= x[-1].astype("datetime64[s]")]
    #     elif high:
    #         self.timestamps = [x for x in self.timestamps if x[-1].astype("datetime64[s]") < high]
    #     if debug:
    #         print(
    #             f"Filtered timestamps from {original_length} to {len(self.timestamps)} examples: "
    #             f"{self.timestamps[0][-1].astype('datetime64[s]')} to {self.timestamps[-1][-1].astype('datetime64[s]')}."
    #         )
    #     self.id2pt = dict(enumerate(self.timestamps))

    def __len__(self):
        return len(self.id2pt)
        
   
    # def convert_to_tensordict(self, xr_dataset,file_path):
    def convert_to_tensordict(self, file_path=None):

        """
        Convert xarray dataset to tensordict.

        By default, it uses a mapping key from self.variables,
            e.g. {surface:[data_var1, data_var2, ...], level:[...]}
        """
        # Optionally select dimensions.
        # if self.dimension_indexers and not self.already_ran_index_selection:
        #     indexers = {v[0]: v[1] for k, v in self.dimension_indexers.items() if k != "time"}

        #     xr_dataset = xr_dataset.sel(**indexers)
        # self.already_ran_index_selection = False  # Reset for next call.
        with Dataset(file_path, "r") as nc:
            # Convert coordinate selections to indices
            indexers = {}
            if self.dimension_indexers:
                indexers = select_by_coords(nc, dimension_indexers)
    
            tdict_data = {}
            for key, vars_ in variables.items():
                if vars_:
                    arrays = []
                    for var in vars_:
                        var_data = nc.variables[var]
    
                        # Build slice object for all dimensions
                        slices = []
                        for dim in var_data.dimensions:
                            if dim in indexers:
                                slices.append(indexers[dim])
                            elif dim == 'line':  # example for your line_id dimension
                                slices.append(line_id)
                            else:
                                slices.append(slice(None))
    
                        data_slice = var_data[tuple(slices)]
                        arrays.append(torch.tensor(data_slice, dtype=torch.float32))
                    tdict_data[key] = torch.stack(arrays)
                else:
                    tdict_data[key] = torch.empty((0,), dtype=torch.float32)

        return TensorDict(tdict_data)
        # Make np arrays for each key and make an empty array if no variables for this list. Needed for running experiments with different variable sets
        # np_arrays = {}
        # for key, variables in self.variables.items():
        #     if variables:  # non-empty
        #         np_arrays[key] = xr_dataset[list(variables)].to_array().to_numpy()
        #     else:  # empty list -> create an empty array
        #         np_arrays[key] = np.empty((0,))

        # tdict = TensorDict(
        #     {key: torch.from_numpy(np_array).float() for key, np_array in np_arrays.items()}
        # )

        from netCDF4 import Dataset
        with Dataset(file_path, "r") as nc:
            tdict_data = {}
            for key, variables in self.variables.items():
                if variables:
                    arrays = [torch.tensor(nc[var][line_id], dtype=torch.float32) for var in variables]
                    tdict_data[key] = torch.stack(arrays)
                else:
                    tdict_data[key] = torch.empty((0,), dtype=torch.float32)
        return TensorDict(tdict_data)
        #     tdict_data = TensorDict()
        
        # for key, variables in self.variables.items():
        #     if variables:  # non-empty
        #         xr_data = xr_dataset[list(variables)].to_array()
        #         data = xr_data.data  # now this is in-memory numpy array

        #         # convert directly to torch tensor
        #         tdict_data[key] = torch.as_tensor(data, dtype=torch.float32)
        #     else:  # empty list
        #         tdict_data[key] = torch.empty((0,), dtype=torch.float32)
        
        # tdict = TensorDict(tdict_data)


        # import xarray.backends.file_manager as fm
        
        # def cleanup_xarray():
        #     fm.CachingFileManager._caches.clear()
        #     gc.collect()

        # cleanup_xarray()






        
        # np_array = None
        # xr_dataset.close()
        # # import netCDF4
        # # netCDF4.Dataset._close(xr_dataset)
        # # import pdb; pdb.set_trace()
        # return tdict


    def __getitem__2(self, i, return_timestamp=False, interpolate_nans=None, warning_on_nan=None):
        import gc
        file_id, line_id, timestamp = self.id2pt[i]

        file_path=self.files[file_id]
        
        tdict_data = TensorDict()
        
        with Dataset(file_path, "r") as nc:
            # Cache indices
            if not hasattr(self, '_index_cache'):
                self._index_cache = {}
            
            if file_path not in self._index_cache:
                cache = {}
                if 'plev' in self.dimension_indexers and 'plev' in nc.variables:
                    cache['plev'] = [list(nc['plev'][:]).index(val) for val in self.dimension_indexers['plev'][1]]
                if 'lev' in self.dimension_indexers and 'lev' in nc.variables:
                    lev_vals = [round(float(x), 3) for x in nc['lev'][:]]
                    cache['lev'] = [lev_vals.index(round(float(val), 3)) for val in self.dimension_indexers['lev'][1]]
                self._index_cache[file_path] = cache
            
            indices = self._index_cache[file_path]
            
            for key, vars_ in self.variables.items():
                tensor_list = []
                
                for var in vars_:
                    # CRITICAL: Load entire slice into memory first
                    if key == 'surface':
                        # Load to numpy, then immediately copy and convert
                        temp = np.array(nc[var][line_id], dtype=np.float32)
                        tensor = torch.from_numpy(temp).unsqueeze(0)
                    elif key == 'level':
                        temp = np.array(nc[var][line_id, indices['plev'], :, :], dtype=np.float32)
                        tensor = torch.from_numpy(temp)
                    elif key == 'lev':
                        temp = np.array(nc[var][line_id, indices['lev'], :], dtype=np.float32)
                        tensor = torch.from_numpy(temp)
                    elif key in ['spatial_forcings', 'non_spatial_forcings', 'ozone']:
                        temp = np.array(nc[var][line_id], dtype=np.float32)
                        tensor = torch.from_numpy(temp)
                    
                    # Detach from any computation graph
                    tensor = tensor.detach().clone()
                    tensor_list.append(tensor)
                    
                    del temp, tensor
                
                tdict_data[key] = torch.stack(tensor_list).detach().clone()
                del tensor_list
        
        # Force cleanup
        gc.collect()
        
        return tdict_data

    def _ensure_cache(self):
        if getattr(self, "_file_cache", None) is None:
            self._file_cache = {}

    def __getitem__(self, i, return_timestamp=False, interpolate_nans=None, warning_on_nan=None):
        # import gc
        
        file_id, line_id, timestamp = self.id2pt[i]
        # file_id, line_id, timestamp = 0,0,0
        file_path = self.files[file_id]
        # data = TensorDict.load_memmap(file_path)
        # # print(data)
        
        # out = TensorDict({'lev':data['lev'][:,line_id],
        #           'level':data['level'][:,line_id],
        #           'surface':data['surface'][:,line_id,None],
        #           'non_spatial_forcings':data['non_spatial_forcings'][:,line_id],
        #           'spatial_forcings':torch.cat([data['spatial_forcings'][:,line_id],data['ozone'][0,line_id]]) if 'ozone' in data.keys() else data['spatial_forcings'][:,line_id]})
        # # print(out)
        # return out

        self._ensure_cache()
        if file_path not in self._file_cache:
            # open once per process (and per worker)
            self._file_cache[file_path] = TensorDict.load_memmap(file_path)

        data = self._file_cache[file_path]

        out = TensorDict({
            'lev': data['lev'][:, line_id],
            'level': data['level'][:, line_id],
            'surface': data['surface'][:, line_id, None],
            'non_spatial_forcings': data['non_spatial_forcings'][:, line_id],
            'spatial_forcings':
                torch.cat([data['spatial_forcings'][:, line_id], data['ozone'][0, line_id]])
                if 'ozone' in data.keys()
                else data['spatial_forcings'][:, line_id]
        })
        # print(self.variables)
        if(len(self.variables['non_spatial_forcings']) == 0):
            out['non_spatial_forcings'] = []
        return out



        
        # # Load all data into pure Python objects FIRST, outside the with block
        # data_dict = TensorDict()
        
        # with Dataset(file_path, "r") as nc:
        #     # Get indices once
        #     if 'plev' in self.dimension_indexers:
        #         plev_list = [float(x) for x in nc['plev'][:]]
        #         plev_indices = [plev_list.index(val) for val in self.dimension_indexers['plev'][1]]
        #     else:
        #         plev_indices = None
            
        #     if 'lev' in self.dimension_indexers:
        #         lev_list = [round(float(x), 3) for x in nc['lev'][:]]
        #         lev_indices = [lev_list.index(round(float(val), 3)) for val in self.dimension_indexers['lev'][1]]
        #     else:
        #         lev_indices = None
            
        #     # Read ALL data as Python lists (completely detached from netCDF)
        #     for key, vars_ in self.variables.items():
        #         # print(key)
        #         # data_dict[key] = []
        #         data_list = []
        #         for var in vars_:
        #             if key == 'surface':
        #                 data = nc[var][line_id][:]  # Convert to Python list
        #             elif key == 'level':
        #                 data = nc[var][line_id, plev_indices, :, :][:]
        #             elif key == 'lev':
        #                 data = nc[var][line_id, lev_indices,:][:]
        #             elif key in ['spatial_forcings','ozone']:
    
        #                 data = nc[var][line_id][:]
        #             elif key in ['non_spatial_forcings']:
        #                 data = nc[var][line_id]
        #                 # print(data)
        #                 if(type(data) is float):
        #                     data = [data]
        #             # print(var)
        #             # print(len(data))
        #             # print(data[0])
        #             data_list.append(data)
        #         data_dict[key] = data_list
        
        # # netCDF file is now CLOSED - no references remain
        # gc.collect()
        
        # # Now convert Python lists to tensors
        # tdict_data = TensorDict()
        # for key, data_list in data_dict.items():
        #     arrays = []
        #     # print('key',key)
        #     # print(len(data_list))
        #     for data in data_list:

        #         tensor = torch.tensor(data, dtype=torch.float32)
        #         if key == 'surface':
        #             tensor = tensor.unsqueeze(0)
        #         arrays.append(tensor)
        
        #     tdict_data[key] = torch.stack(arrays)
        #     del arrays
        
        # del data_dict
        # gc.collect()
        
        # return tdict_data
        
        #with zarr
        # tdict_data = TensorDict()

        # zr = zarr.open(file_path, mode='r')
        # for key, vars_ in self.variables.items():
        #     arrays = []
        #     for var in vars_:
        #         if key == 'surface':
        #             var_data = zr[var].oindex[line_id]
        #             var_data = torch.tensor(np.array(var_data, copy=True))[None]
        #         elif key == 'lev':
        #             lev_indices_temp = [round(float(x), 3) for x in list(zr['lev'][:])]
        #             lev_indices = [lev_indices_temp.index(round(float(val), 3)) for val in self.dimension_indexers['lev'][1]]
        #             # Use oindex for fancy indexing
        #             var_data = torch.tensor(np.array(zr[var].oindex[line_id, lev_indices, :], copy=True))
        #         elif key == 'lev':
        #             lev_indices_temp = [round(float(x), 3) for x in list(zr['lev'][:])]
        #             lev_indices = [lev_indices_temp.index(round(float(val), 3)) for val in self.dimension_indexers['lev'][1]]
        #             var_data = torch.tensor(np.array(zr[var].oindex[line_id, lev_indices, :], copy=True))
        #         elif key in ['spatial_forcings', 'non_spatial_forcings', 'ozone']:
        #             var_data = torch.tensor(np.array(zr[var].oindex[line_id], copy=True))
        #         arrays.append(var_data)
        #     tdict_data[key] = torch.stack(arrays)
        #     del arrays  # Explicit cleanup
        
        # tdict_data = TensorDict()

        # Open with consolidated metadata (much faster)
        # zr = zarr.open_consolidated(file_path, mode='r') if zarr.consolidate_metadata else zarr.open(file_path, mode='r')
        
        # PRE-COMPUTE indices outside the loop (CRITICAL!)
        # lev_indices_temp = [round(float(x), 3) for x in zr['lev'][:]]
        # lev_indices_cache = {}
        # if 'lev' in self.dimension_indexers:
        #     lev_indices_cache['lev'] = [lev_indices_temp.index(round(float(val), 3)) 
        #                                   for val in self.dimension_indexers['lev'][1]]
        
        # plev_indices_cache = {}
        # if 'plev' in self.dimension_indexers:
        #     plev_list = list(zr['plev'][:])
        #     plev_indices_cache['plev'] = [plev_list.index(val) 
        #                                    for val in self.dimension_indexers['plev'][1]]
        
        # Now loop through variables
        # for key, vars_ in self.variables.items():
        #     arrays = []
        #     for var in vars_:
        #         if key == 'surface':
        #             # Use direct slicing instead of oindex when possible
        #             var_data = torch.from_numpy(zr[var][line_id:line_id+1].copy())
                    
        #         elif key == 'level':
        #             plev_indices = self.plev_indices_cache['plev']
        #             # Use oindex for fancy indexing
        #             var_data = torch.from_numpy(zr[var].oindex[line_id, plev_indices, :, :].copy())
                    
        #         elif key == 'lev':
        #             lev_indices = self.lev_indices_cache['lev']
        #             # Use oindex for fancy indexing
        #             var_data = torch.from_numpy(zr[var].oindex[line_id, lev_indices, :].copy())
                    
        #         elif key in ['spatial_forcings']:
        #             var_data = torch.from_numpy(zr[var][line_id].copy())
        #         elif key in ['non_spatial_forcings' ]:
        #             var_data = torch.tensor(zr[var][line_id])
        #         # elif key in ['ozone']:
        #         #     key = 'spatial_forcings'
        #         #     var_data = torch.from_numpy(zr[var][line_id].copy())

        #         arrays.append(var_data)
        #     if(key == 'spatial_forcings'):
        #         for i in range(66):
        #             arrays.append(torch.from_numpy(zr['ozone'][line_id,i].copy()))
        #     if(key == 'ozone'):
        #         continue
        #     tdict_data[key] = torch.stack(arrays)
        #     del arrays

            # print(tdict_data)







        

        
        # with Dataset(file_path, "r") as nc:
        #     for key, vars_ in self.variables.items():
        #         arrays = []
        #         for var in vars_:
        #             if key == 'surface':
        #                 var_data = nc[var][line_id]
        #                 var_data = torch.tensor(np.array(var_data, copy=True))[None]
        #             elif key == 'level':
        #                 plev_indices = [list(nc['plev'][:]).index(val) for val in self.dimension_indexers['plev'][1]]
        #                 var_data = torch.tensor(np.array(nc[var][line_id, plev_indices, :, :], copy=True))
        #             elif key == 'lev':
        #                 lev_indices_temp = [round(float(x), 3) for x in list(nc['lev'][:])]
        #                 lev_indices = [lev_indices_temp.index(round(float(val), 3)) for val in self.dimension_indexers['lev'][1]]
        #                 var_data = torch.tensor(np.array(nc[var][line_id, lev_indices, :], copy=True))
        #             elif key in ['spatial_forcings', 'non_spatial_forcings', 'ozone']:
        #                 var_data = torch.tensor(np.array(nc[var][line_id], copy=True))
        #             arrays.append(var_data)
        #         if(key == 'spatial_forcings'):
        #             for i in range(66):
        #                 arrays.append(torch.from_numpy(nc['ozone'][line_id,i].copy()))
        #         if(key == 'ozone'):
        #             continue
        #         tdict_data[key] = torch.stack(arrays)
        #         del arrays  # Explicit cleanup
        # with Dataset(file_path, "r") as nc:
        #     # Convert coordinate selections to indices
        #     indexers = {}
        #     print(self.dimension_indexers)
        #     if self.dimension_indexers:
        #         indexers = select_by_coords(nc, self.dimension_indexers)
        #     print(indexers)
        #     tdict_data = {}
        #     for key, vars_ in variables.items():
        #         if vars_:
        #             arrays = []
        #             for var in vars_:
        #                 var_data = nc.variables[var]
    
        #                 # Build slice object for all dimensions
        #                 slices = []
        #                 for dim in var_data.dimensions:
        #                     if dim in indexers:
        #                         slices.append(indexers[dim])
        #                     # elif dim == 'line':  # example for your line_id dimension
        #                     #     slices.append(line_id)
        #                     else:
        #                         slices.append(slice(None))
    
        #                 data_slice = var_data[tuple(slices)]
        #                 arrays.append(torch.tensor(data_slice, dtype=torch.float32))
        #             tdict_data[key] = torch.stack(arrays)
        #         else:
        #             tdict_data[key] = torch.empty((0,), dtype=torch.float32)

        # return tdict_data


        
        # with Dataset(file_path, "r") as nc:
        #     tdict_data = {}
        #     for key, variables in self.variables.items():
        #         if variables:
        #             arrays = [torch.tensor(nc[var][line_id], dtype=torch.float32) for var in variables]
        #             tdict_data[key] = torch.stack(arrays)
        #         else:
        #             tdict_data[key] = torch.empty((0,), dtype=torch.float32)
        # return TensorDict(tdict_data)
        # tdict = self.convert_to_tensordict(file_path=self.files[file_id])

        # cached_xrdataset.close()

        # obsi.close()

        # if warning_on_nan:
        #     if any([x.isnan().any().item() for x in tdict.values()]):
        #         warnings.warn(f"NaN values detected in {file_id} {line_id} {self.files[file_id]}")

        # if return_timestamp or self.return_timestamp:
        #     timestamp = self.cached_xrdataset.time[line_id].values.item()
        #     timestamp = torch.tensor(timestamp // 10**9, dtype=torch.int32)
        #     return tdict, timestamp

        # return tdict
