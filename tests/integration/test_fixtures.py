import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def create_dummy_era5_data(data_dir: Path, num_timestamps: int = 6, lat: int = 120, lon: int = 240):
    """Create minimal dummy ERA5 data for integration tests."""
    LAT, LON = lat, lon  # Default to match constant masks dimensions (120x240)
    LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  # Match pangu defaults
    
    # Surface variables to include (full names expected by Era5Forecast)
    surface_vars = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure"]
    # Level variables to include (full names expected by Era5Forecast)
    level_vars = ["geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity", "vertical_velocity"]
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy files with expected filename patterns for train/val/test
    for i in range(num_timestamps // 2):  # 2 timestamps per file
        # Use years that match the filename filters: train (2018), val (2019), test (2020)
        year = 2018 + (i % 3)  # Cycle through 2018, 2019, 2020
        # Add time suffix for test_z0012 filter compatibility
        time_suffix = "0h" if i % 2 == 0 else "12h"
        file_path = data_dir / f"era5_{year}_0{i:02d}_{time_suffix}.nc"
        
        # Create timestamps that match the year in the filename
        start_date = f"{year}-01-01"
        times = pd.date_range(start_date, periods=2, freq="6h")
        time = times
        
        # Create dummy data with proper shapes
        level_var_data = np.random.randn(len(time), len(LEVELS), LON, LAT)
        surface_var_data = np.random.randn(len(time), LAT, LON)
        
        data_vars = {}
        # Add level variables
        for var in level_vars:
            data_vars[var] = (["time", "level", "longitude", "latitude"], level_var_data)
        
        # Add surface variables  
        for var in surface_vars:
            data_vars[var] = (["time", "latitude", "longitude"], surface_var_data)
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": time,
                "latitude": np.linspace(-90, 90, LAT),
                "longitude": np.linspace(0, 360, LON),
                "level": LEVELS,
            },
        )
        
        ds.to_netcdf(file_path)
    
    return data_dir


def create_dummy_stats_file(stats_dir: Path):
    """Create minimal statistics file for normalization."""
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy quantiles file
    LAT, LON = 4, 8
    LEVELS = [50, 100, 200, 500, 850, 1000]
    
    surface_vars = ["u10", "v10", "t2m", "msl"]
    level_vars = ["u", "v", "t", "z"]
    
    data_vars = {}
    
    # Create dummy quantile data for each variable
    for var in surface_vars:
        # Mean and std for normalization
        data_vars[f"{var}_mean"] = (["latitude", "longitude"], np.zeros((LAT, LON)))
        data_vars[f"{var}_std"] = (["latitude", "longitude"], np.ones((LAT, LON)))
    
    for var in level_vars:
        data_vars[f"{var}_mean"] = (["level", "latitude", "longitude"], np.zeros((len(LEVELS), LAT, LON)))
        data_vars[f"{var}_std"] = (["level", "latitude", "longitude"], np.ones((len(LEVELS), LAT, LON)))
    
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "latitude": np.linspace(-90, 90, LAT),
            "longitude": np.linspace(0, 360, LON),
            "level": LEVELS,
        },
    )
    
    stats_file = stats_dir / "era5-quantiles-2016_2022.nc"
    ds.to_netcdf(stats_file)
    
    return stats_file