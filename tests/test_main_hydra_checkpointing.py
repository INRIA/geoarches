import os

import pytest

from geoarches.main_hydra import select_checkpoint


def test_select_checkpoint_uses_latest_checkpoint_in_experiment_dir(tmp_path):
    ckpt_dir = tmp_path / "run" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    first = ckpt_dir / "checkpoint_global_step=100.ckpt"
    second = ckpt_dir / "checkpoint_global_step=200.ckpt"
    first.touch()
    second.touch()
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))

    assert select_checkpoint(tmp_path / "run") == second


def test_select_checkpoint_filters_by_filename_match(tmp_path):
    ckpt_dir = tmp_path / "run" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    latest = ckpt_dir / "checkpoint_global_step=200.ckpt"
    matched = ckpt_dir / "checkpoint_global_step=100.ckpt"
    latest.touch()
    matched.touch()
    os.utime(latest, (2, 2))
    os.utime(matched, (1, 1))

    assert select_checkpoint(tmp_path / "run", "100") == matched


def test_select_checkpoint_accepts_checkpoint_file(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.touch()

    assert select_checkpoint(ckpt) == ckpt


def test_select_checkpoint_errors_when_match_is_missing(tmp_path):
    ckpt_dir = tmp_path / "run" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "checkpoint_global_step=100.ckpt").touch()

    with pytest.raises(FileNotFoundError, match="matching '200'"):
        select_checkpoint(tmp_path / "run", "200")
