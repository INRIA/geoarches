import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def create_dummy_era5_data(data_dir: Path, num_timestamps: int = 6, lat: int = 121, lon: int = 240):
    """Create minimal dummy ERA5 data for integration tests."""
    # Use full ERA5 grid dimensions
    # The dataloader expects these exact coordinate values
    LAT_COORDS = 121  # Full ERA5 latitude points
    LON_COORDS = 240  # Full ERA5 longitude points
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
        
        # Create dummy data with full ERA5 grid dimensions
        # This ensures compatibility with the dataloader's coordinate selection
        level_var_data = np.random.randn(len(time), len(LEVELS), LAT_COORDS, LON_COORDS)
        surface_var_data = np.random.randn(len(time), LAT_COORDS, LON_COORDS)
        
        data_vars = {}
        # Add level variables
        for var in level_vars:
            data_vars[var] = (["time", "level", "latitude", "longitude"], level_var_data)
        
        # Add surface variables  
        for var in surface_vars:
            data_vars[var] = (["time", "latitude", "longitude"], surface_var_data)
        
        # Generate coordinates that match exactly what Era5Forecast expects
        # Era5Forecast will try to select these specific values with method="nearest"
        expected_lats = np.arange(90, -90 - 1e-6, -180 / 120)  # Full ERA5 latitudes (121 points)
        expected_lons = np.arange(0, 360, 360 / 240)  # Full ERA5 longitudes (240 points)
        
        # Use the full expected coordinate arrays
        latitude = expected_lats[:LAT_COORDS]
        longitude = expected_lons[:LON_COORDS]
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": time,
                "latitude": latitude,
                "longitude": longitude,
                "level": LEVELS,
            },
        )
        
        ds.to_netcdf(file_path)
    
    return data_dir
