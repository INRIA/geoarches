"""Integration tests for the geoarches train/test workflow.
Minimal overrides - use defaults where possible

These tests validate:
1. Config composition works correctly
2. Real dataloader can be instantiated with train/test data
3. Training workflow runs without crashing
4. Test/inference workflow runs without crashing
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from geoarches.main_hydra import main as hydra_main


def create_dummy_era5_data(data_dir: Path, num_timestamps: int = 6):
    """Create minimal dummy ERA5 data for integration tests."""
    # Use full ERA5 grid dimensions
    # The dataloader expects these exact coordinate values
    LAT_COORDS = 121  # Full ERA5 latitude points
    LON_COORDS = 240  # Full ERA5 longitude points
    LEVELS = [
        50,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        850,
        925,
        1000,
    ]  # Match pangu defaults

    # Surface variables to include (full names expected by Era5Forecast)
    surface_vars = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
    ]
    # Level variables to include (full names expected by Era5Forecast)
    level_vars = [
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "specific_humidity",
        "vertical_velocity",
    ]

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
        rng = np.random.default_rng(42)
        level_var_data = rng.standard_normal((len(time), len(LEVELS), LAT_COORDS, LON_COORDS))
        surface_var_data = rng.standard_normal((len(time), LAT_COORDS, LON_COORDS))

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


def test_config_composition():
    """Test that Hydra config composition works correctly."""
    GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path="../../geoarches/configs"):
        cfg = compose(config_name="config", overrides=["max_steps=5", "log=false"])

        # Validate config structure
        assert hasattr(cfg, "mode")
        assert hasattr(cfg, "dataloader")
        assert hasattr(cfg, "module")
        assert hasattr(cfg, "stats")
        assert hasattr(cfg, "cluster")
        assert cfg.max_steps == 5
        assert cfg.log is False


def test_real_dataloader_instantiation(tmp_path):
    """Test that Era5Forecast dataloader can be instantiated with test data."""

    data_dir = tmp_path / "data"
    create_dummy_era5_data(data_dir, num_timestamps=10)

    GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path="../../geoarches/configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"dataloader.dataset.path={data_dir}",
                "dataloader.dataset.multistep=1",
                "dataloader.dataset.lead_time_hours=6",
                "max_steps=1",
                "batch_size=1",
                "log=false",
            ],
        )

        # Test dataloader instantiation (as done in main_hydra.py)
        from hydra.utils import instantiate

        dataset = instantiate(cfg.dataloader.dataset, cfg.stats)
        assert dataset is not None
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")
        assert len(dataset) > 0

        # Test getting an actual sample
        sample = dataset[0]
        assert "state" in sample
        assert "next_state" in sample
        assert "timestamp" in sample
        assert "lead_time_hours" in sample


def test_workflow_with_real_dataloader(tmp_path):
    """Test complete train/test workflow with real Era5Forecast dataloader."""

    os.chdir(tmp_path)
    GlobalHydra.instance().clear()

    data_dir = tmp_path / "data"
    create_dummy_era5_data(data_dir, num_timestamps=8)

    with initialize(version_base=None, config_path="../../geoarches/configs"):
        # First, run training to create a checkpoint
        train_cfg = compose(
            config_name="config",
            overrides=[
                f"exp_dir={tmp_path}/test_model",
                "name=test_model",
                "mode=train",
                "max_steps=1",  # Minimal steps for quick test
                "batch_size=1",  # Small batch size
                "log=false",
                "save_step_frequency=1",
                # Minimal dataloader overrides - use defaults where possible
                f"dataloader.dataset.path={data_dir}",
                "dataloader.dataset.multistep=1",
                "dataloader.dataset.lead_time_hours=6",
                # Disable multiprocessing to avoid pickling issues with lambda functions
                "cluster.cpus=0",
                # Disable GPU to avoid memory issues in CI
                "cluster.gpus=0",
                # Reduce model complexity to save memory
                "module.backbone.emb_dim=8",  # Minimal embedding dimension
                "module.backbone.num_heads=[2,2,2,2]",  # Minimal attention heads
                "module.backbone.depth_multiplier=1",
                "module.backbone.mlp_ratio=1.0",
                "module.embedder.emb_dim=8",
                "module.embedder.out_emb_dim=16",
                # Minimal test settings
                "limit_val_batches=1",
                "seed=42",
            ],
        )

        # Run training
        hydra_main(train_cfg)

        # Verify checkpoint creation
        ckpt_dir = Path(train_cfg.exp_dir) / "checkpoints"
        assert ckpt_dir.exists(), "Checkpoint directory not created"
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        assert len(ckpts) > 0, "No checkpoints saved"

        # Save config for test mode
        config_path = Path(train_cfg.exp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(train_cfg, resolve=True))

        # Now run test mode with the same configuration
        test_cfg = train_cfg.copy()
        test_cfg.mode = "test"

        # Run test/inference
        hydra_main(test_cfg)
