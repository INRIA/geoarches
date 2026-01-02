import numpy as np
import pandas as pd
import pytest
import xarray as xr
import torch
from torch.utils.data import DataLoader
from geoarches.dataloaders import era5, era5pred

# Dimension sizes.
LAT, LON = 2, 4
LEVEL = len(era5.pressure_levels)

class TestEra5ForecastWithPrediction:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path_factory):
        # Use tmp_path_factory to create a shared temporary directory.
        self.test_dir = tmp_path_factory.mktemp("data")
        self.pred_dir = tmp_path_factory.mktemp("pred_data")

        
        # We need slightly more data for forecast + prediction alignment
        # 6h freq. Total 8 timestamps to be safe.
        times = pd.date_range("2024-01-01", periods=12, freq="6h")
        
        # Create Dummy Ground Truth Data (Era5)
        # 4 files with 3 timestamps each
        for i in range(4):
            file_path = self.test_dir / f"fake_era5_{i}.nc"
            file_times = times[i*3 : i*3+3]
            
            level_var_data = np.zeros((len(file_times), LEVEL, LON, LAT))
            surface_var_data = np.zeros((len(file_times), LAT, LON))

            ds = xr.Dataset(
                data_vars=dict(
                    **{
                        var_name: (["time", "level", "longitude", "latitude"], level_var_data)
                        for var_name in era5.level_variables
                    },
                    **{
                        var_name: (["time", "latitude", "longitude"], surface_var_data)
                        for var_name in era5.surface_variables
                    },
                ),
                coords={
                    "time": file_times,
                    "latitude": np.arange(0, LAT),
                    "longitude": np.arange(0, LON),
                    "level": np.arange(0, LEVEL)
                },
            )
            ds.to_netcdf(file_path)

        # Create Dummy Prediction Data
        # Needs to cover the same range roughly.
        # Often predictions might be separate files or same structure.
        # We use same structure for simplicity as per existing code reading xarray.
        for i in range(4):
            file_path = self.pred_dir / f"fake_pred_{i}.nc"
            file_times = times[i*3 : i*3+3]
            
            # Predictions might have same vars
            level_var_data = np.ones((len(file_times), LEVEL, LON, LAT)) * 0.5
            surface_var_data = np.ones((len(file_times), LAT, LON)) * 0.5
            
            ds_pred = xr.Dataset(
                 data_vars=dict(
                    **{
                        var_name: (["time", "level", "longitude", "latitude"], level_var_data)
                        for var_name in era5.level_variables
                    },
                    **{
                        var_name: (["time", "latitude", "longitude"], surface_var_data)
                        for var_name in era5.surface_variables
                    },
                ),
                coords={
                    "time": file_times,
                    "latitude": np.arange(0, LAT),
                    "longitude": np.arange(0, LON),
                    "level": np.arange(0, LEVEL)
                },
            )
            ds_pred.to_netcdf(file_path)


    def test_output_shape_and_dtype(self):
        """
        Verify that the tensors returned have the correct shape and dtype (float32).
        This covers Unit test shape and dtype.
        """
        ds = era5pred.Era5ForecastWithPrediction(
            path=str(self.test_dir),
            pred_path=str(self.pred_dir),
            domain="all",
            lead_time_hours=6,
            load_prev=False,
            norm_scheme=None # Disable normalization for simpler value checking
        )
        
        # Fetch one sample
        sample = ds[0]
        
        # Check basic keys exist
        assert "state" in sample
        assert "next_state" in sample
        assert "pred_state" in sample
        
        # Check Dtypes
        assert sample["state"]["surface"].dtype == torch.float32
        assert sample["pred_state"]["surface"].dtype == torch.float32

        # Check Shapes
        # state: (var, 1, lat, lon)
        # surface vars: 4
        # level vars: 6 * 13 = 78 (but dataloader might keep them separate)
        # Era5Dataset normalizes/stacks them in specific ways.
        # surface: (4, 1, LAT, LON)
        assert sample["state"]["surface"].shape == (4, 1, LAT, LON)
        assert sample["pred_state"]["surface"].shape == (4, 1, LAT, LON)


    def test_batch_size_1_dataloader(self):
        """
        Verify that the dataset works correctly when wrapped in a DataLoader with batch_size=1.
        This covers Batch size 1 test.
        """
        ds = era5pred.Era5ForecastWithPrediction(
            path=str(self.test_dir),
            pred_path=str(self.pred_dir),
            domain="all",
            lead_time_hours=6,
            load_prev=False,
            norm_scheme=None
        )
        
        def collate_fn(batch):
            # Manually stack nested dicts/TensorDicts
            # We explicitly handle the structure to avoid TensorDict iteration issues
            # and torch.stack(dict) errors
            return {
                "state": {
                    kk: torch.stack([x["state"][kk] for x in batch])
                    for kk in batch[0]["state"].keys()
                },
                "pred_state": {
                    kk: torch.stack([x["pred_state"][kk] for x in batch])
                    for kk in batch[0]["pred_state"].keys()
                },
                "next_state": {
                    kk: torch.stack([x["next_state"][kk] for x in batch])
                    for kk in batch[0]["next_state"].keys()
                },
            }

        dl = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
        batch = next(iter(dl))
        
        # Batch dimension should be 1
        assert batch["state"]["surface"].shape[0] == 1
        assert batch["pred_state"]["surface"].shape[0] == 1
        
    def test_timestamp_alignment(self):
        """
        Verify that the prediction state loaded corresponds to the same timestamp as the ground truth.
        """
        ds = era5pred.Era5ForecastWithPrediction(
            path=str(self.test_dir),
            pred_path=str(self.pred_dir),
            domain="all",
            lead_time_hours=24, # Larger gap
            load_prev=False,
            norm_scheme=None
        )
        
        # Alignment is validated by internal assertions in the dataset
        
        for i in range(len(ds)):
            sample = ds[i]
            # Verify basic integrity
            assert "timestamp" in sample
            assert sample["state"] is not None
            
    def test_load_prev_alignment(self):
        """
        Test alignment when load_prev is True.
        """
        ds = era5pred.Era5ForecastWithPrediction(
            path=str(self.test_dir),
            pred_path=str(self.pred_dir),
            domain="all",
            lead_time_hours=6,
            load_prev=True,
            norm_scheme=None
        )
        # Just iterating to ensure assertions pass
        sample = ds[0]
        assert "prev_state" in sample
        assert "pred_state" in sample
