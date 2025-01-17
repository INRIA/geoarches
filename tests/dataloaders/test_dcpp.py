import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from geoarches.dataloaders import dcpp

# Dimension sizes.
LAT, LON = 143, 144
PLEV = 3


class TestDCPPForecast:
    @classmethod
    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_path_factory):
        # Use tmp_path_factory to create a class-level temporary directory.
        self.test_dir = tmp_path_factory.mktemp("data")
        times = pd.date_range("2024-01-01", periods=6, freq="1ME")  # datetime64[ns]
        for i in range(2):
            file_path = self.test_dir / f"fake_dcpp_{i}_tos_included.nc"
            time = times[i * 2 : i * 2 + 2]

            # Create some dummy data
            level_var_data = np.zeros((len(time), PLEV, LAT, LON))
            surface_var_data = np.zeros((len(time), LAT, LON))
            ds = xr.Dataset(
                data_vars=dict(
                    **{
                        var_name: (["time", "plev", "lat", "lon"], level_var_data)
                        for var_name in dcpp.level_variables
                    },
                    **{
                        var_name: (["time", "lat", "lon"], surface_var_data)
                        for var_name in dcpp.surface_variables
                    },
                ),
                coords={
                    "time": time,
                    "lat": np.arange(0, LAT),
                    "lon": np.arange(0, LON),
                    "plev": [85000, 70000, 50000],
                },
            )
            ds.to_netcdf(file_path)

        # make fake atmos forcings
        full_atmos_normal = torch.rand((4, 540))
        torch.save(full_atmos_normal, f"{self.test_dir}/full_atmos_normal.pt")
        full_solar_normal = torch.rand((340, 12, 6))
        torch.save(full_solar_normal, f"{self.test_dir}/full_solar_normal.pt")

    def test_load_current_state(self):
        dcpp_model = dcpp.DCPPForecast(
            path=str(self.test_dir),
            forcings_path=str(self.test_dir),
            domain="train",
            lead_time_months=1,
            load_prev=False,
            multistep=0,
            load_clim=False,
        )
        example = next(iter(dcpp_model))

        assert len(dcpp_model) == 2
        # Current state
        assert example["timestamp"] == 1711843200  # 2024-01-01-00-00
        assert example["state"]["surface"].shape == (10, 1, LAT, LON)  #  (var, lat, lon)
        assert example["state"]["level"].shape == (8, 3, LAT, LON)  #  (var, lev, lat, lon)
        assert example["forcings"].shape == torch.Size([10])  #  (var)

    @pytest.mark.parametrize(
        "lead_time_months, expected_len, expected_next_timestamp",
        [(1, 1, 1704088800), (1, 1, 1704110400)],
    )
    def test_load_current_and_next_state(
        self, lead_time_months, expected_len, expected_next_timestamp
    ):
        ds = dcpp.DCPPForecast(
            path=str(self.test_dir),
            forcings_path=str(self.test_dir),
            domain="train",
            lead_time_months=lead_time_months,
            load_prev=False,
            load_clim=False,
        )
        example = ds[0]

        assert len(ds) == expected_len
        # Current state
        assert example["timestamp"] == 1711843200  # 2024-01-01-00-00
        assert example["state"]["surface"].shape == (10, 1, LAT, LON)  #  (var, lat, lon)
        assert example["state"]["level"].shape == (8, 3, LAT, LON)  #  (var, lev, lat, lon)
        # Next state
        assert example["next_state"]["surface"].shape == (10, 1, LAT, LON)  #  (var, lat, lon)
        assert example["next_state"]["level"].shape == (8, 3, LAT, LON)  #  (var, lev, lat, lon)
        # No multistep
        assert "future_states" not in example
        # No prev state
        assert "prev_state" not in example

    # def test_norm_denorm(
    #     self, lead_time_months, expected_len, expected_next_timestamp
    # ):
    #     ds = dcpp.DCPPForecast(
    #         path=str(self.test_dir),
    #         domain="train",
    #         lead_time_months=lead_time_months,
    #         load_prev=False,
    #         load_clim=False,
    #     )
    #     example = ds[0]
    #    # assert torch.equal(ds.denormalize(ds.normalize(example))['state']['surface'],ds[0]['state']['surface'])
    #     # print(ds.denormalize(ds.normalize(example))['state']['surface'])
    #     # naned = replace_nans(ds.denormalize(ds.normalize(example)))
    #     # print(naned)
    #     # print(example)
    #     print(ds.denormalize(ds.normalize(example)))
    #     denormed = {k: replace_nans(v,self.mask_value) if 'state' in k else v for k, v in ds.denormalize(ds.normalize(example))}

    #     assert torch.allclose(denormed['state']['surface'],example['state']['surface'], rtol=1e-05, atol=1e-05, equal_nan=False)
