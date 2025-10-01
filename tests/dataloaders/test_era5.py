import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geoarches.dataloaders import era5, era5_constants
from tests.fixtures.era5 import LAT, LEVEL, LON, TestBase, all_levels, cfg


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


@pytest.fixture(scope="session")
def write_val_test_data(tmp_path_factory):
    """Write dummy data for validation and testing years."""
    # Use tmp_path_factory to create a class-level temporary directory.
    test_dir = tmp_path_factory.mktemp("val_test_data")

    def _write_data_per_year(start_date, end_date, year):
        file_path = test_dir / f"fake_era5_{year}.nc"

        time = pd.date_range(start=start_date, end=end_date, freq="12h")  # datetime64[ns]

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

    # Write 1 file per year.
    for i, year in enumerate(range(2018, 2022)):
        start_date = f"{year}-12-01" if i == 0 else f"{year}-01-01"
        end_date = f"{year}-01-31" if i == 3 else f"{year}-12-31 12:00:00"
        _write_data_per_year(start_date, end_date, year)
    for i, year in enumerate(range(2013, 2016)):
        start_date = f"{year}-12-01" if i == 0 else f"{year}-01-01"
        end_date = f"{year}-01-31" if i == 2 else f"{year}-12-31 12:00:00"
        _write_data_per_year(start_date, end_date, year)

    return test_dir


class TestEra5Forecast(TestBase):
    @pytest.mark.parametrize(
        "lead_time_hours, expected_len",
        [(6, 5), (12, 4), (24, 2)],
    )
    def test_load_current_and_next_state(self, lead_time_hours, expected_len):
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
        assert example["hour_of_day"] == 0
        assert example["month"] == 1
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
        assert example["hour_of_day"] == 0
        assert example["month"] == 1
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
        assert example["hour_of_day"] == 6
        assert example["month"] == 1
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

    @pytest.mark.parametrize(
        (
            "domain, load_prev, multistep, lead_time_hours, expected_start_time,"
            " expected_end_time, expected_len"
        ),
        [
            (
                "val",
                True,
                2,
                24,
                np.datetime64("2018-12-31T00:00:00"),
                np.datetime64("2020-01-02T12:00:00"),
                365 * 2,  # 12h steps for non-leap year
            ),  # 2019
            (
                "val",
                False,
                2,
                24,
                np.datetime64("2019-01-01T00:00:00"),
                np.datetime64("2020-01-02T12:00:00"),
                365 * 2,  # 12h steps for non-leap year
            ),  # 2019
            (
                "test",
                True,
                1,
                24,
                np.datetime64("2019-12-31T00:00:00"),
                np.datetime64("2021-01-01T12:00:00"),
                366 * 2,  # 12h steps for leap year
            ),  # 2020
            (
                "test",
                True,
                1,  # multistep
                48,  # 2 days lead time
                np.datetime64("2019-12-30T00:00:00"),
                np.datetime64("2021-01-02T12:00:00"),
                366 * 2,  # 12h steps for leap year
            ),  # 2020
            (
                "aimip_val",
                True,
                10,
                24,
                np.datetime64("2013-12-31T00:00:00"),
                np.datetime64("2015-01-10T12:00:00"),
                365 * 2,  # 12h steps for non-leap year
            ),  # 2014
        ],
    )
    def test_reselect_timestamps(
        self,
        write_val_test_data,
        domain,
        load_prev,
        multistep,
        lead_time_hours,
        expected_start_time,
        expected_end_time,
        expected_len,
    ):
        ds = era5.Era5Forecast(
            stats_cfg=None,
            path=str(write_val_test_data),
            domain=domain,
            lead_time_hours=lead_time_hours,  # lead time to load.
            timedelta_hours=12,  # step between timestamps in data.
            multistep=multistep,
            load_prev=load_prev,
            load_clim=False,
            # Select all values in each dimension.
            dimension_indexers={
                "level": ("level", all_levels),
                "latitude": ("latitude", slice(None)),
                "longitude": ("longitude", slice(None)),
            },
        )

        assert ds.timestamps[0][-1] == expected_start_time
        assert ds.timestamps[-1][-1] == expected_end_time
        assert len(ds) == expected_len

    def test_interpolate_nans(self):
        # NaNs are introduced at the first timestamp of the second file (t2).
        ds = era5.Era5Forecast(
            stats_cfg=None,
            path=str(self.test_dir),
            domain="all",
            lead_time_hours=6,
            multistep=1,
            load_prev=True,
            load_clim=False,
            interpolate_nans=True,  # Interpolate NaNs in input (prev_state and state).
            dimension_indexers={
                "level": ("level", all_levels),
                "latitude": ("latitude", slice(None)),
                "longitude": ("longitude", slice(None)),
            },
        )

        # example = ds[0]: prev=t0, state=t1, next=t2.
        # t2 has NaNs. Since interpolate_nans is True, prev_state and state are interpolated,
        # but next_state is not. So, next_state should contain NaNs.
        example0 = ds[0]
        assert not np.isnan(example0["prev_state"]["level"].numpy()).any()
        assert not np.isnan(example0["prev_state"]["surface"].numpy()).any()
        assert not np.isnan(example0["state"]["level"].numpy()).any()
        assert not np.isnan(example0["state"]["surface"].numpy()).any()
        # next_state should have NaNs.
        assert np.isnan(example0["next_state"]["level"].numpy()).any()
        assert np.isnan(example0["next_state"]["surface"].numpy()).any()

        # example = ds[1]: prev=t1, state=t2, next=t3.
        # t2 and t3 has NaNs. Since interpolate_nans is True, state should be interpolated.
        example1 = ds[1]
        assert not np.isnan(example1["prev_state"]["level"].numpy()).any()
        assert not np.isnan(example1["prev_state"]["surface"].numpy()).any()
        # state has NaNs originally, but should be interpolated.
        assert not np.isnan(example1["state"]["level"].numpy()).any()
        assert not np.isnan(example1["state"]["surface"].numpy()).any()
        # next_state should have NaNs.
        assert np.isnan(example1["next_state"]["level"].numpy()).any()
        assert np.isnan(example1["next_state"]["surface"].numpy()).any()

        # example = ds[2]: prev=t2, state=t3, next=t4.
        # t2, t3 and t4 has NaNs. Since interpolate_nans is True,
        # # prev_state and state should be interpolated.
        example1 = ds[2]
        # prev_state has NaNs originally, but should be interpolated.
        assert not np.isnan(example1["prev_state"]["level"].numpy()).any()
        assert not np.isnan(example1["prev_state"]["surface"].numpy()).any()
        # state has NaNs originally, but should be interpolated.
        assert not np.isnan(example1["state"]["level"].numpy()).any()
        assert not np.isnan(example1["state"]["surface"].numpy()).any()
        # next_state should have NaNs.
        assert np.isnan(example1["next_state"]["level"].numpy()).any()
        assert np.isnan(example1["next_state"]["surface"].numpy()).any()


class TestEra5ForecastWithGraphcastNormalization(TestBase):
    @pytest.mark.parametrize(
        "lead_time_hours, expected_len",
        [(6, 5), (12, 4), (24, 2)],
    )
    def test_load_current_and_next_state(self, lead_time_hours, expected_len):
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

        assert ds.norm_scheme == "graphcast"

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

        # Check if normalization is applied.
        example_normalized = example["state"]["surface"]
        example_denormalized = ds.denormalize(example)["state"]["surface"]
        assert not np.allclose(example_normalized, example_denormalized)

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

        assert ds.norm_scheme == "graphcast"

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

        assert ds.norm_scheme == "graphcast"

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
        "lead_time_hours, expected_len",
        [(6, 5), (12, 4), (24, 2)],
    )
    def test_load_current_and_next_state(self, lead_time_hours, expected_len):
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

        assert ds.norm_scheme == "pangu"

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

        # Check if normalization is applied.
        example_normalized = example["state"]["surface"]
        example_denormalized = ds.denormalize(example)["state"]["surface"]
        assert not np.allclose(example_normalized, example_denormalized)


class TestEra5ForecastWithForcings:
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

        # Forcings dataset.
        self.test_forcings_dir = tmp_path_factory.mktemp("forcings")
        # Create the time coordinate with a monthly frequency
        time = pd.date_range("2024-01-01", periods=4, freq="MS")

        file_path = self.test_forcings_dir / "fake_forcings.nc"

        ds = xr.Dataset(
            data_vars={
                "sea_ice_cover": (
                    ("time", "latitude", "longitude"),
                    np.random.rand(len(time), LAT, LON),
                ),
                "sea_surface_temperature": (
                    ("time", "latitude", "longitude"),
                    np.random.rand(len(time), LAT, LON),
                ),
            },
            coords={
                "time": time,
                "latitude": np.arange(0, LAT),
                "longitude": np.arange(0, LON),
            },
        )
        ds.to_netcdf(file_path)

    @pytest.mark.parametrize(
        "lead_time_hours, expected_len",
        [(6, 5), (12, 4), (24, 2)],
    )
    def test_load_current_and_next_state(self, lead_time_hours, expected_len):
        ds = era5.Era5Forecast(
            stats_cfg=None,
            path=str(self.test_dir),
            forcings_path=str(self.test_forcings_dir / "fake_forcings.nc"),
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
            forcing_vars=["sea_ice_cover", "sea_surface_temperature"],
        )
        example = ds[0]

        assert len(ds) == expected_len
        # Current state
        assert example["timestamp"] == 1704067200  # 2024-01-01-00-00
        assert example["state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        assert example["forcings"].shape == (2, LAT, LON)  #  (var, lat, lon)
        # Next state
        assert example["next_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["next_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # No multistep
        assert "future_states" not in example
        assert "future_forcings" not in example
        # No prev state
        assert "prev_state" not in example

    @pytest.mark.parametrize("multistep, expected_len", [(2, 4), (3, 3), (4, 2)])
    def test_multistep(self, multistep, expected_len):
        ds = era5.Era5Forecast(
            stats_cfg=None,
            path=str(self.test_dir),
            forcings_path=str(self.test_forcings_dir / "fake_forcings.nc"),
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
            forcing_vars=["sea_ice_cover", "sea_surface_temperature"],
        )
        example = ds[0]

        assert len(ds) == expected_len
        # Current state
        assert example["timestamp"] == 1704067200  # 2024-01-01-00-00
        assert example["state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        assert example["forcings"].shape == (2, LAT, LON)  #  (var, lat, lon)
        # Next state
        assert example["next_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["next_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Future states
        assert example["future_states"]["surface"].shape[0] == multistep
        assert example["future_states"]["level"].shape[0] == multistep
        assert example["future_forcings"].shape[0] == multistep
        # No prev state
        assert "prev_state" not in example

    @pytest.mark.parametrize("multistep, expected_len", [(2, 3), (3, 2), (4, 1)])
    def test_multistep_and_load_prev(self, multistep, expected_len):
        ds = era5.Era5Forecast(
            stats_cfg=None,
            path=str(self.test_dir),
            forcings_path=str(self.test_forcings_dir / "fake_forcings.nc"),
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
            forcing_vars=["sea_ice_cover", "sea_surface_temperature"],
        )
        example = ds[0]

        assert len(ds) == expected_len
        # Current state
        assert example["timestamp"] == 1704088800  # 2024-01-01-06-00
        assert example["state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        assert example["forcings"].shape == (2, LAT, LON)  #  (var, lat, lon)
        # Next state
        assert example["next_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["next_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
        # Future states
        assert example["future_states"]["surface"].shape[0] == multistep
        assert example["future_states"]["level"].shape[0] == multistep
        assert example["future_forcings"].shape[0] == multistep
        # Prev state
        assert example["prev_state"]["surface"].shape == (4, 1, LAT, LON)  #  (var, 1, lat, lon)
        assert example["prev_state"]["level"].shape == (6, 13, LAT, LON)  #  (var, lev, lat, lon)
