"""Download pretrained ArchesWeather models and their matching configurations."""

import argparse
import shutil
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlretrieve

import torch
from huggingface_hub import hf_hub_download

MODEL_REPOSITORY = "gcouairon/ArchesWeather"
MODEL_REVISION = "b93acfab1061cbd6792bc02533434c1125065893"
MODEL_NAMES = (
    "archesweather-m-seed0",
    "archesweather-m-seed1",
    "archesweather-m-skip-seed0",
    "archesweather-m-skip-seed1",
    "archesweathergen",
)
_SOURCE_CONFIG_DIRECTORY = Path(__file__).resolve().parents[2] / "paper" / "configs"
_LIGHTNING_CHECKPOINT_VERSION = "2.5.0.post0"


def _install_config(model: str, destination: Path) -> None:
    """Install the Hydra config matching the current geoarches version.

    A source checkout already contains the configs under ``paper/configs``. PyPI wheels do
    not include that directory, so wheel installations download the same config from the Git
    tag corresponding to the installed package version.
    """
    # Case 1: geoarches is installed from a source checkout.
    source_config = _SOURCE_CONFIG_DIRECTORY / f"{model}.yaml"
    if source_config.is_file():
        shutil.copyfile(source_config, destination)
        return

    # Case 2: geoarches is installed from a PyPI wheel.
    release_tag = f"v{version('geoarches')}"
    config_url = (
        f"https://raw.githubusercontent.com/INRIA/geoarches/{release_tag}/"
        f"paper/configs/{model}.yaml"
    )
    urlretrieve(config_url, destination)


def _patch_checkpoint(checkpoint_path: Path) -> None:
    """Ensure a downloaded checkpoint can be restored by Lightning."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        checkpoint = {"state_dict": checkpoint}

    if "pytorch-lightning_version" not in checkpoint:
        checkpoint["pytorch-lightning_version"] = _LIGHTNING_CHECKPOINT_VERSION

    torch.save(checkpoint, checkpoint_path)


def download_models(
    output_directory: str | Path = "modelstore",
) -> None:
    """Download model checkpoints and version-matched Hydra configurations."""
    output_directory = Path(output_directory)

    for model in MODEL_NAMES:
        model_directory = output_directory / model
        checkpoint_directory = model_directory / "checkpoints"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_directory / "checkpoint.ckpt"
        config_path = model_directory / "config.yaml"

        if not checkpoint_path.is_file():
            downloaded_checkpoint = hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                filename=f"{model}_checkpoint.ckpt",
                revision=MODEL_REVISION,
                local_dir=model_directory,
            )
            Path(downloaded_checkpoint).replace(checkpoint_path)
            _patch_checkpoint(checkpoint_path)

        if not config_path.is_file():
            _install_config(model, config_path)

        print(f"Downloaded {model} to {model_directory}")


def main() -> None:
    """Run the pretrained-model downloader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        default="modelstore",
        help="Directory in which model folders are created (default: modelstore).",
    )
    args = parser.parse_args()
    download_models(args.output_directory)


if __name__ == "__main__":
    main()
