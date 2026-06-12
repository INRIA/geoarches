import lightning as L  # noqa: N812
import torch

from geoarches.main_hydra import GeoArchesCheckpointIO


def test_geoarches_checkpoint_io_adds_missing_lightning_version(tmp_path):
    ckpt_path = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": {"weight": torch.ones(1)}}, ckpt_path)

    checkpoint = GeoArchesCheckpointIO().load_checkpoint(ckpt_path)

    assert checkpoint["pytorch-lightning_version"] == L.__version__


def test_geoarches_checkpoint_io_keeps_existing_lightning_version(tmp_path):
    ckpt_path = tmp_path / "modern.ckpt"
    torch.save(
        {"state_dict": {"weight": torch.ones(1)}, "pytorch-lightning_version": "2.0.0"},
        ckpt_path,
    )

    checkpoint = GeoArchesCheckpointIO().load_checkpoint(ckpt_path)

    assert checkpoint["pytorch-lightning_version"] == "2.0.0"
