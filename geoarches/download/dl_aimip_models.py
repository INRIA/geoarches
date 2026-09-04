"""Download pretrained AIMIP ArchesWeather models and their matching configurations."""

import argparse
import shutil
import zipfile
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlretrieve

import torch

from geoarches.download.constants import ZENODO_RECORD_ID

ZENODO_ZIP_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/model_checkpoints.zip?download=1"
)
MODEL_NAMES = (
    "aimip-archesweather-m-seed0",
    "aimip-archesweather-m-seed1",
    "aimip-archesweather-m-seed2",
    "aimip-archesweather-m-seed3",
    "aimip-archesweathergen",
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


def _has_checkpoint(model_directory: Path) -> bool:
    """Check whether a checkpoint file exists for the given model directory."""
    checkpoint_directory = model_directory / "checkpoints"
    return checkpoint_directory.is_dir() and any(checkpoint_directory.glob("*.ckpt"))


def _download_file(url: str, destination: Path) -> None:
    """Download a file with progress reporting."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(
                f"\rDownloading {destination.name}: {mb:.1f} / {total_mb:.1f} MB ({percent:.1f}%)",
                end="",
                flush=True,
            )

    try:
        urlretrieve(url, tmp_path, reporthook=_reporthook)
        print()
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _extract_checkpoints_from_zip(
    zip_path: Path,
    output_directory: Path,
    models: tuple[str, ...] | list[str] = MODEL_NAMES,
) -> None:
    """Extract model checkpoints from the downloaded zip file and patch them."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if (
                member.is_dir()
                or member.filename.startswith("__MACOSX")
                or member.filename.endswith(".DS_Store")
            ):
                continue

            path_parts = Path(member.filename).parts
            for model in models:
                if model in path_parts and member.filename.endswith(".ckpt"):
                    checkpoint_directory = output_directory / model / "checkpoints"
                    checkpoint_directory.mkdir(parents=True, exist_ok=True)
                    dest_checkpoint = checkpoint_directory / Path(member.filename).name
                    with zf.open(member) as src, open(dest_checkpoint, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    _patch_checkpoint(dest_checkpoint)


def download_models(
    output_directory: str | Path = "modelstore",
    models: tuple[str, ...] | list[str] = MODEL_NAMES,
) -> None:
    """Download AIMIP model checkpoints and version-matched Hydra configurations."""
    output_directory = Path(output_directory)

    missing_models = [model for model in models if not _has_checkpoint(output_directory / model)]

    if missing_models:
        zip_path = output_directory / "model_checkpoints.zip"
        output_directory.mkdir(parents=True, exist_ok=True)
        try:
            _download_file(ZENODO_ZIP_URL, zip_path)
            _extract_checkpoints_from_zip(zip_path, output_directory, models=missing_models)
        finally:
            if zip_path.exists():
                zip_path.unlink()

    for model in models:
        model_directory = output_directory / model
        config_path = model_directory / "config.yaml"

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
