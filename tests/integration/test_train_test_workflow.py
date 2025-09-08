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
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from geoarches.main_hydra import main as hydra_main



def test_config_composition():
    """Test that Hydra config composition works correctly."""
    GlobalHydra.instance().clear()
    
    with initialize(version_base=None, config_path="../../geoarches/configs"):
        cfg = compose(
            config_name="config",
            overrides=["max_steps=5", "log=false"]
        )
        
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
    from tests.integration.test_fixtures import create_dummy_era5_data
    
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
                "log=false"
            ]
        )
        
        # Test dataloader instantiation (as done in main_hydra.py)
        from hydra.utils import instantiate
        
        dataset = instantiate(cfg.dataloader.dataset, cfg.stats)
        assert dataset is not None
        assert hasattr(dataset, '__len__')
        assert hasattr(dataset, '__getitem__')
        assert len(dataset) > 0
        
        # Test getting an actual sample
        sample = dataset[0]
        assert "state" in sample
        assert "next_state" in sample
        assert "timestamp" in sample
        assert "lead_time_hours" in sample


def test_workflow_with_real_dataloader(tmp_path):
    """Test complete train/test workflow with real Era5Forecast dataloader."""
    from tests.integration.test_fixtures import create_dummy_era5_data
    
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
                
                
                # Reduce model complexity to save memory
                "module.backbone.depth_multiplier=1",
                "module.backbone.mlp_ratio=1.0",
                
                # Minimal test settings
                "limit_val_batches=1",
                "seed=42",
            ]
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