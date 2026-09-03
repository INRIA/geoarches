import io
import zipfile
from pathlib import Path

import torch

from geoarches.download import dl_aimip_models, dl_aw_models


def test_download_models_from_source_config(tmp_path, monkeypatch):
    """Test downloading ArchesWeather models from HuggingFace in a source checkout."""
    model = "archesweather-m-seed0"
    calls = []

    # Mock HuggingFace download to return a minimal dummy checkpoint
    def fake_download(**kwargs):
        calls.append(kwargs)
        checkpoint = Path(kwargs["local_dir"]) / kwargs["filename"]
        torch.save({"state_dict": {"weight": torch.tensor([1.0])}}, checkpoint)
        return str(checkpoint)

    monkeypatch.setattr(dl_aw_models, "hf_hub_download", fake_download)

    output_directory = tmp_path / "models"
    dl_aw_models.download_models(output_directory)

    # Verify checkpoint contents, Lightning metadata, and paper config
    model_directory = output_directory / model
    checkpoint = torch.load(
        model_directory / "checkpoints" / "checkpoint.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["state_dict"]["weight"].item() == 1.0
    assert checkpoint["pytorch-lightning_version"] == "2.5.0.post0"
    assert (model_directory / "config.yaml").read_text() == (
        dl_aw_models._SOURCE_CONFIG_DIRECTORY / f"{model}.yaml"
    ).read_text()
    assert calls[0]["revision"] == dl_aw_models.MODEL_REVISION

    # Verify existing checkpoints are reused without extra downloads
    dl_aw_models.download_models(output_directory)
    assert len(calls) == len(dl_aw_models.MODEL_NAMES)


def test_patch_raw_state_dictionary(tmp_path):
    """Test patching raw checkpoints to include state_dict and Lightning version."""
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    torch.save({"weight": torch.tensor([2.0])}, checkpoint_path)

    dl_aw_models._patch_checkpoint(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["state_dict"]["weight"].item() == 2.0
    assert checkpoint["pytorch-lightning_version"] == "2.5.0.post0"


def test_install_config_from_matching_release_tag(tmp_path, monkeypatch):
    """Test remote config download via GitHub release tag (wheel installations)."""
    destination = tmp_path / "config.yaml"
    # Simulate wheel installation where local paper/configs does not exist
    monkeypatch.setattr(dl_aw_models, "_SOURCE_CONFIG_DIRECTORY", tmp_path / "missing")
    monkeypatch.setattr(dl_aw_models, "version", lambda _: "1.2.3")

    def fake_urlretrieve(url, path):
        assert "/v1.2.3/paper/configs/archesweathergen.yaml" in url
        Path(path).write_text("config")

    monkeypatch.setattr(dl_aw_models, "urlretrieve", fake_urlretrieve)

    dl_aw_models._install_config("archesweathergen", destination)

    assert destination.read_text() == "config"


def test_download_aimip_models_from_source_config(tmp_path, monkeypatch):
    """Test downloading AIMIP models from Zenodo in a source checkout."""
    calls = []

    # Mock downloading the Zenodo zip file with synthetic checkpoints
    def fake_download(url, destination, **kwargs):
        calls.append(url)
        with zipfile.ZipFile(destination, "w") as zf:
            for model in dl_aimip_models.MODEL_NAMES:
                ckpt_bytes = io.BytesIO()
                torch.save({"state_dict": {"weight": torch.tensor([1.0])}}, ckpt_bytes)
                zf.writestr(
                    f"model_checkpoints/{model}/checkpoints/checkpoint_global_step=300000.ckpt",
                    ckpt_bytes.getvalue(),
                )

    monkeypatch.setattr(dl_aimip_models, "urlretrieve", fake_download)

    output_directory = tmp_path / "models"
    dl_aimip_models.download_models(output_directory)

    # Verify extracted checkpoint, Lightning metadata, and copied paper config
    model = "aimip-archesweather-m-seed0"
    model_directory = output_directory / model
    checkpoint = torch.load(
        model_directory / "checkpoints" / "checkpoint_global_step=300000.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["state_dict"]["weight"].item() == 1.0
    assert checkpoint["pytorch-lightning_version"] == "2.5.0.post0"
    assert (model_directory / "config.yaml").read_text() == (
        dl_aimip_models._SOURCE_CONFIG_DIRECTORY / f"{model}.yaml"
    ).read_text()
    assert len(calls) == 1

    # Verify existing checkpoints are reused without redownloading the zip
    dl_aimip_models.download_models(output_directory)
    assert len(calls) == 1


def test_install_aimip_config_from_matching_release_tag(tmp_path, monkeypatch):
    """Test remote AIMIP config download via GitHub release tag (wheel installations)."""
    destination = tmp_path / "config.yaml"
    # Simulate wheel installation where local paper/configs does not exist
    monkeypatch.setattr(dl_aimip_models, "_SOURCE_CONFIG_DIRECTORY", tmp_path / "missing")
    monkeypatch.setattr(dl_aimip_models, "version", lambda _: "1.2.3")

    def fake_urlretrieve(url, path):
        assert "/v1.2.3/paper/configs/aimip-archesweathergen.yaml" in url
        Path(path).write_text("config")

    monkeypatch.setattr(dl_aimip_models, "urlretrieve", fake_urlretrieve)

    dl_aimip_models._install_config("aimip-archesweathergen", destination)

    assert destination.read_text() == "config"
