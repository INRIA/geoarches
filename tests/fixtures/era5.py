import hydra
import numpy as np
import omegaconf
import pandas as pd
import pytest
import xarray as xr

from geoarches.dataloaders import era5_constants

with hydra.initialize(version_base=None, config_path="../../geoarches/configs", job_name="test"):
    cfg = hydra.compose(config_name="config")
    omegaconf.OmegaConf.resolve(cfg)

# Dimension sizes.
LAT, LON = 2, 4
# Need real levels to load correct normalization stats.
all_levels = omegaconf.OmegaConf.to_object(cfg.stats.module.levels)
LEVEL = len(all_levels)


class TestBase:
    """Base class for ERA5 tests, providing a setup with fake ERA5 data files.

    This class sets up a temporary directory and creates three fake NetCDF files
    (fake_era5_0.nc, fake_era5_1.nc, fake_era5_2.nc) containing dummy level and
    surface variable data. NaNs are introduced in files 1 and 2 to test handling
    of missing data. The files cover a time range from 2024-01-01 to 2024-01-02
    with 6-hourly frequency.
    """

    @classmethod
    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_path_factory):
        # Use tmp_path_factory to create a class-level temporary directory.
        self.test_dir = tmp_path_factory.mktemp("data")
        times = pd.date_range("2024-01-01", periods=6, freq="6h")  # datetime64[ns]

        # 3 files with 2 timestamps each.
        for i in range(3):
            file_path = self.test_dir / f"fake_era5_{i}.nc"
            time = times[i * 2 : i * 2 + 2]

            # Create some dummy data
            level_var_data = np.zeros((len(time), LEVEL, LON, LAT))  # Lon first.
            surface_var_data = np.zeros((len(time), LAT, LON))  # Lat first.

            # Introduce NaNs in the second file (index 1) for all timestamps.
            if i == 1:
                level_var_data[:, 0, 0, 0] = np.nan
                surface_var_data[:, 0, 0] = np.nan
            # Introduce NaNs in the third file (index 2) for first timestep.
            if i == 2:
                level_var_data[0, 0, 0, 0] = np.nan
                surface_var_data[0, 0, 0] = np.nan

            ds = xr.Dataset(
                data_vars=dict(
                    **{
                        var_name: (["time", "level", "longitude", "latitude"], level_var_data)
                        for var_name in era5_constants.arches_default_level_variables
                    },
                    **{
                        var_name: (["time", "latitude", "longitude"], surface_var_data)
                        for var_name in era5_constants.arches_default_surface_variables
                    },
                ),
                coords={
                    "time": time,
                    "latitude": np.arange(0, LAT),
                    "longitude": np.arange(0, LON),
                    "level": all_levels,
                },
            )
            ds.to_netcdf(file_path)
