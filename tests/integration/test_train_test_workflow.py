"""Integration tests for the geoarches train/test workflow.

These tests validate:
1. Config composition works correctly
2. Real dataloader can be instantiated with test data
3. Training workflow runs without crashing (with mock data)
4. Test/inference workflow runs without crashing (with mock data)
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


class MockDataset:
    """Mock dataset that mimics Era5Forecast interface for testing."""
    
    def __init__(self, *args, **kwargs):
        self.data_size = 5
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        # Return data structure matching what the model expects
        return {
            "state": torch.randn(4, 4, 8),  # [channels, height, width] 
            "timestamp": torch.tensor(1234567890, dtype=torch.int64),
            "lead_time_hours": torch.tensor(6, dtype=torch.int32),
        }


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
    create_dummy_era5_data(data_dir, num_timestamps=6)
    
    GlobalHydra.instance().clear()
    
    with initialize(version_base=None, config_path="../../geoarches/configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"dataloader.dataset.path={data_dir}",
                "dataloader.dataset.norm_scheme=null",  # Disable normalization for testing
                "dataloader.dataset.load_prev=false", 
                "dataloader.dataset.multistep=1",
                "dataloader.dataset.lead_time_hours=6",
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


@pytest.mark.parametrize("mode", ["train", "test"]) 
def test_workflow_with_real_dataloader(tmp_path, mode):
    """Test complete train/test workflow with real Era5Forecast dataloader."""
    from tests.integration.test_fixtures import create_dummy_era5_data
    
    os.chdir(tmp_path)
    GlobalHydra.instance().clear()
    
    # Create test data
    data_dir = tmp_path / "data"
    create_dummy_era5_data(data_dir, num_timestamps=8)  # More timestamps for multistep
    
    with initialize(version_base=None, config_path="../../geoarches/configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"exp_dir={tmp_path}/test_model",
                "name=test_model",
                f"mode={mode}",
                "max_steps=1", 
                "batch_size=1",
                "log=false",
                "save_step_frequency=1",
                
                # Real dataloader configuration
                f"dataloader.dataset.path={data_dir}",
                "dataloader.dataset.norm_scheme=null",  # Disable normalization for testing
                "dataloader.dataset.load_prev=false",
                "dataloader.dataset.multistep=1", 
                "dataloader.dataset.lead_time_hours=6",
                
                # Minimal model for fast testing
                "module.module.cond_dim=8",
                "module.module.num_warmup_steps=1", 
                "module.backbone.tensor_size=[13,120,240]",  # [pressure_levels, lat, lon] to match our data
                "module.backbone.emb_dim=8",
                "module.backbone.cond_dim=8",
                "module.embedder.img_size=[82,120,240]",  # Total channels: 4 surface + 6*13 level = 82, spatial dims match constant masks
                "module.embedder.surface_ch=4",  # 4 surface variables
                "module.embedder.level_ch=6",   # 6 level variables  
                "module.embedder.n_concatenated_states=0",  # No previous states since load_prev=false
                "module.embedder.patch_size=[1,1,1]",  # Use 1x1x1 patches to avoid dimension issues
                "module.embedder.emb_dim=8", 
                "module.embedder.out_emb_dim=8",
                
                # Test settings
                "cluster.cpus=0",
                "cluster.batch_size=1", 
                "cluster.wandb_mode=disabled",
                "limit_val_batches=1",
                "seed=42",
            ]
        )
        
        # Create dummy checkpoint and config for test mode
        if mode == "test":
            model_dir = Path(cfg.exp_dir) / "checkpoints"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {}}, model_dir / "dummy.ckpt")
            
            # Create config.yaml file that test mode expects
            config_path = Path(cfg.exp_dir) / "config.yaml"
            with open(config_path, "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))
        
        # Run workflow
        hydra_main(cfg)
        
        # For train mode, verify checkpoint creation
        if mode == "train":
            ckpt_dir = Path(cfg.exp_dir) / "checkpoints" 
            assert ckpt_dir.exists(), "Checkpoint directory not created"
            ckpts = list(ckpt_dir.glob("*.ckpt"))
            assert len(ckpts) > 0, "No checkpoints saved"