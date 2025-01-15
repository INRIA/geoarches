import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from geoarches.dataloaders import dcpp

# Dimension sizes.
LAT, LON = 143, 144
PLEV = 4


class TestDCPPForecast:
    @classmethod
    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_path_factory):
        # Use tmp_path_factory to create a class-level temporary directory.
        self.test_dir = tmp_path_factory.mktemp("data")
        times = pd.date_range("2024-01-01", periods=6, freq="1ME")  # datetime64[ns]
        for i in range(2):
            file_path = self.test_dir / f"1961_{i}.nc"
            time = times[i * 2 : i * 2 + 2]

            # Create some dummy data
            level_var_data = np.zeros((len(time), PLEV, LAT, LON))
            surface_var_data = np.zeros((len(time), LAT, LON))
            level_variables = ['va','ua','zg','wap']
            surface_variables = ['psl','tos']
            ds = xr.Dataset(
                data_vars=dict(
                    **{
                        var_name: (["time", "plev", "lat", "lon"], level_var_data)
                        for var_name in level_variables
                    },
                    **{
                        var_name: (["time", "lat", "lon"], surface_var_data)
                        for var_name in surface_variables
                    },
                ),
                coords={
                    "time": time,
                    "lat": np.arange(0, LAT),
                    "lon": np.arange(0, LON),
                    "plev": [85000, 70000, 50000,25000],
                },
            )
            ds.to_netcdf(file_path)

        # make fake atmos forcings
        full_atmos_normal = torch.rand((540,4))
        np.save(f"{self.test_dir}/ghg_forcings_normed.npy",full_atmos_normal.numpy())
        full_solar_normal = torch.rand((804, 6))
        np.save( f"{self.test_dir}/solar_forcings_normed.npy",full_solar_normal.numpy())

    def test_load_current_state(self):
        dcpp_model = dcpp.DCPPForecast(
            path=str(self.test_dir),
            forcings_path=str(self.test_dir),
            domain="train",
            lead_time_months=1,
            load_prev=False,
            multistep=0,
            load_clim=False,
            surface_variable_indices=[0,1],
            level_variable_indices=[0,1,2],
            surface_variables=['psl','tos'],
            level_variables=['va','ua','zg']   
        )
        example = next(iter(dcpp_model))

        assert len(dcpp_model) == 4
        # Current state
        assert example["timestamp"] == 1706659200  # 2024-01-01-00-00
        assert example["state"]["surface"].shape == (2, 1, LAT, LON)  #  (var, lat, lon)
        assert example["state"]["level"].shape == (3, 4, LAT, LON)  #  (var, lev, lat, lon)
        assert example["forcings"].shape == torch.Size([10])  #  (var)

    @pytest.mark.parametrize(
        "lead_time_months, expected_len, expected_next_timestamp",
        [(1, 3, 1704088800), (1, 3, 1704110400)],
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
            surface_variable_indices=[0,1],
            level_variable_indices=[0,1,2],
            surface_variables=['psl','tos'],
            level_variables=['va','ua','zg']        
        )
        example = ds[0]

        assert len(ds) == expected_len
        # Current state
        assert example["timestamp"] == 1706659200  # 2024-01-01-00-00
        assert example["state"]["surface"].shape == (2, 1, LAT, LON)  #  (var, lat, lon)
        assert example["state"]["level"].shape == (3, 4, LAT, LON)  #  (var, lev, lat, lon)
        # Next state
        assert example["next_state"]["surface"].shape == (2, 1, LAT, LON)  #  (var, lat, lon)
        assert example["next_state"]["level"].shape == (3, 4, LAT, LON)  #  (var, lev, lat, lon)
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
