import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hydra import compose, initialize
from omegaconf import OmegaConf

from geoarches.dataloaders import era5, era5_constants

with initialize(version_base=None, config_path="../../geoarches/configs", job_name="test"):
    cfg = compose(config_name="config")
    OmegaConf.resolve(cfg)

# Dimension sizes.
LAT, LON = 2, 4
# Need real levels to load correct normalization stats.
all_levels = OmegaConf.to_object(cfg.stats.module.levels)
LEVEL = len(all_levels)


class TestBase:
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


class TestEra5Dataset(TestBase):
    def test_load_current_state(self):
        ds = era5.Era5Dataset(
            path=str(self.test_dir),
            domain="all",
            # Select all values in each dimension.
            dimension_indexers={
                "level": ("level", all_levels),
                "latitude": ("latitude", slice(None)),
                "longitude": ("longitude", slice(None)),
            },
        )
        example = ds[0]

        assert len(ds) == 6
        assert example["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["level"].shape == (6, LEVEL, LAT, LON)  #  (var, lev, lat, lon)

    def test_load_current_state_with_timestamp(self):
        ds = era5.Era5Dataset(
            path=str(self.test_dir),
            domain="all",
            return_timestamp=True,
            # Select all values in each dimension.
            dimension_indexers={
                "level": ("level", all_levels),
                "latitude": ("latitude", slice(None)),
                "longitude": ("longitude", slice(None)),
            },
        )
        example, timestamp = ds[0]

        assert len(ds) == 6
        # Current state
        assert example["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["level"].shape == (6, LEVEL, LAT, LON)  #  (var, lev, lat, lon)
        assert timestamp == 1704067200  # 2024-01-01-00-00

    @pytest.mark.parametrize(
        "indexers, expected_lat, expected_lon",
        [
            # Filter by level only.
            (
                {
                    "level": ("level", all_levels[3:]),
                    "latitude": ("latitude", slice(None)),
                    "longitude": ("longitude", slice(None)),
                },
                LAT,
                LON,
            ),
            # Filter by level and latitude.
            (
                {
                    "level": ("level", all_levels[3:]),
                    "latitude": ("latitude", np.arange(0, LAT - 1)),
                    "longitude": ("longitude", slice(None)),
                },
                LAT - 1,
                LON,
            ),
            # Filter by level and longitude.
            (
                {
                    "level": ("level", all_levels[3:]),
                    "latitude": ("latitude", slice(None)),
                    "longitude": ("longitude", np.arange(0, LON - 1)),
                },
                LAT,
                LON - 1,
            ),
        ],
    )
    def test_dimension_indexers(self, indexers, expected_lat, expected_lon):
        ds = era5.Era5Dataset(
            path=str(self.test_dir),
            domain="all",
            dimension_indexers=indexers,
        )
        example = ds[0]

        assert len(ds) == 6
        assert example["surface"].shape == (
            4,
            1,
            expected_lat,
            expected_lon,
        )  #  (var, 1, lat, lon)
        assert example["level"].shape == (
            6,
            len(indexers["level"][1]),
            expected_lat,
            expected_lon,
        )  #  (var, lev, lat, lon)


class TestEra5Forecast(TestBase):
    @pytest.mark.parametrize(
        "lead_time_hours, expected_len, expected_next_timestamp",
        [(6, 5, 1704088800), (12, 4, 1704110400), (24, 2, 1704153600)],
    )
    def test_load_current_and_next_state(
        self, lead_time_hours, expected_len, expected_next_timestamp
    ):
        ds = era5.Era5Forecast(
            stats_cfg=None,
            path=str(self.test_dir),
            domain="all",
            lead_time_hours=lead_time_hours,
            load_prev=False,
            load_clim=False,
            # Select all values in each dimension.
            dimension_indexers={
                "level": ("level", all_levels),
                "latitude": ("latitude", slice(None)),
                "longitude": ("longitude", slice(None)),
            },
        )
        example = ds[0]

        assert len(ds) == expected_len
        # Current state
        assert example["timestamp"] == 1704067200  # 2024-01-01-00-00
        assert example["state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Next state
        assert example["next_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["next_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # No multistep
        assert "future_states" not in example
        # No prev state
        assert "prev_state" not in example

    @pytest.mark.parametrize("multistep, expected_len", [(2, 4), (3, 3), (4, 2)])
    def test_multistep(self, multistep, expected_len):
        ds = era5.Era5Forecast(
            stats_cfg=None,
            path=str(self.test_dir),
            domain="all",
            lead_time_hours=6,
            multistep=multistep,
            load_prev=False,
            load_clim=False,
            # Select all values in each dimension.
            dimension_indexers={
                "level": ("level", all_levels),
                "latitude": ("latitude", slice(None)),
                "longitude": ("longitude", slice(None)),
            },
        )
        example = ds[0]

        assert len(ds) == expected_len
        # Current state
        assert example["timestamp"] == 1704067200  # 2024-01-01-00-00
        assert example["state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Next state
        assert example["next_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["next_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Future states
        assert example["future_states"]["surface"].shape[0] == multistep
        assert example["future_states"]["level"].shape[0] == multistep
        # No prev state
        assert "prev_state" not in example

    @pytest.mark.parametrize("multistep, expected_len", [(2, 3), (3, 2), (4, 1)])
    def test_multistep_and_load_prev(self, multistep, expected_len):
        ds = era5.Era5Forecast(
            stats_cfg=None,
            path=str(self.test_dir),
            domain="all",
            lead_time_hours=6,
            multistep=multistep,
            load_prev=True,
            load_clim=False,
            # Select all values in each dimension.
            dimension_indexers={
                "level": ("level", all_levels),
                "latitude": ("latitude", slice(None)),
                "longitude": ("longitude", slice(None)),
            },
        )
        example = ds[0]

        assert len(ds) == expected_len
        # Current state
        assert example["timestamp"] == 1704088800  # 2024-01-01-06-00
        assert example["state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Next state
        assert example["next_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["next_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Future states
        assert example["future_states"]["surface"].shape[0] == multistep
        assert example["future_states"]["level"].shape[0] == multistep
        # Prev state
        assert example["prev_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["prev_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)


class TestEra5ForecastWithGraphcastNormalization(TestBase):
    @pytest.mark.parametrize(
        "lead_time_hours, expected_len, expected_next_timestamp",
        [(6, 5, 1704088800), (12, 4, 1704110400), (24, 2, 1704153600)],
    )
    def test_load_current_and_next_state(
        self, lead_time_hours, expected_len, expected_next_timestamp
    ):
        cfg.stats.module.norm_scheme = "graphcast"
        ds = era5.Era5Forecast(
            stats_cfg=cfg.stats,
            path=str(self.test_dir),
            domain="all",
            lead_time_hours=lead_time_hours,
            load_prev=False,
            load_clim=False,
            # Select all lat/lon.
            dimension_indexers={
                "latitude": ("latitude", slice(None)),
                "longitude": ("longitude", slice(None)),
            },
        )
        example = ds[0]

        assert len(ds) == expected_len
        # Current state
        assert example["timestamp"] == 1704067200  # 2024-01-01-00-00
        assert example["state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Next state
        assert example["next_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["next_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # No multistep
        assert "future_states" not in example
        # No prev state
        assert "prev_state" not in example

    @pytest.mark.parametrize("multistep, expected_len", [(2, 4), (3, 3), (4, 2)])
    def test_multistep(self, multistep, expected_len):
        cfg.stats.module.norm_scheme = "graphcast"
        ds = era5.Era5Forecast(
            stats_cfg=cfg.stats,
            path=str(self.test_dir),
            domain="all",
            lead_time_hours=6,
            multistep=multistep,
            load_prev=False,
            load_clim=False,
            # Select all lat/lon.
            dimension_indexers={
                "latitude": ("latitude", slice(None)),
                "longitude": ("longitude", slice(None)),
            },
        )
        example = ds[0]

        assert len(ds) == expected_len
        # Current state
        assert example["timestamp"] == 1704067200  # 2024-01-01-00-00
        assert example["state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Next state
        assert example["next_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["next_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Future states
        assert example["future_states"]["surface"].shape[0] == multistep
        assert example["future_states"]["level"].shape[0] == multistep
        # No prev state
        assert "prev_state" not in example

    @pytest.mark.parametrize("multistep, expected_len", [(2, 3), (3, 2), (4, 1)])
    def test_multistep_and_load_prev(self, multistep, expected_len):
        cfg.stats.module.norm_scheme = "graphcast"
        ds = era5.Era5Forecast(
            stats_cfg=cfg.stats,
            path=str(self.test_dir),
            domain="all",
            lead_time_hours=6,
            multistep=multistep,
            load_prev=True,
            load_clim=False,
            # Select all lat/lon.
            dimension_indexers={
                "latitude": ("latitude", slice(None)),
                "longitude": ("longitude", slice(None)),
            },
        )
        example = ds[0]

        assert len(ds) == expected_len
        # Current state
        assert example["timestamp"] == 1704088800  # 2024-01-01-06-00
        assert example["state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Next state
        assert example["next_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["next_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Future states
        assert example["future_states"]["surface"].shape[0] == multistep
        assert example["future_states"]["level"].shape[0] == multistep
        # Prev state
        assert example["prev_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["prev_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)


class TestEra5ForecastWithPanguNormalization(TestBase):
    @pytest.mark.parametrize(
        "lead_time_hours, expected_len, expected_next_timestamp",
        [(6, 5, 1704088800), (12, 4, 1704110400), (24, 2, 1704153600)],
    )
    def test_load_current_and_next_state(
        self, lead_time_hours, expected_len, expected_next_timestamp
    ):
        cfg.stats.module.norm_scheme = "pangu"
        ds = era5.Era5Forecast(
            stats_cfg=cfg.stats,
            path=str(self.test_dir),
            domain="all",
            lead_time_hours=lead_time_hours,
            load_prev=False,
            load_clim=False,
            # Select all lat/lon.
            dimension_indexers={
                "latitude": ("latitude", slice(None)),
                "longitude": ("longitude", slice(None)),
            },
        )

        example = ds[0]

        assert len(ds) == expected_len
        # Current state
        assert example["timestamp"] == 1704067200  # 2024-01-01-00-00
        assert example["state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Next state
        assert example["next_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["next_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # No multistep
        assert "future_states" not in example
        # No prev state
        assert "prev_state" not in example
