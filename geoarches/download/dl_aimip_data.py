"""Download pretrained AIMIP ERA5 dataset slice, monthly forcings, and normalization stats."""

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from geoarches.download.constants import ZENODO_RECORD_ID

ZENODO_DATA_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"

DATA_FILES = {
    "era5_1978_00h_sliced": {
        "url": f"{ZENODO_DATA_BASE_URL}/era5_1978_00h_sliced.nc?download=1",
        "dest": Path("data") / "era5_1x1" / "daily_averaged" / "era5_1978_00h_sliced.nc",
        "description": "ERA5 1978 00h daily averaged slice",
    },
    "monthly_forcing": {
        "url": f"{ZENODO_DATA_BASE_URL}/ERA5-0.25deg-monthly-mean-forcing-1978-2024_regridded_conservative_2025_11_03.nc?download=1",
        "dest": Path("data")
        / "era5_1x1"
        / "ERA5-0.25deg-monthly-mean-forcing-1978-2024_regridded_conservative_2025_11_03.nc",
        "description": "Monthly mean forcing (1978-2024)",
    },
    "forcing_stats": {
        "url": f"{ZENODO_DATA_BASE_URL}/ERA5-0.25deg-monthly-mean-forcing-1978-2013_regridded_conservative_norm_stats.nc?download=1",
        "dest": Path("stats")
        / "ERA5-0.25deg-monthly-mean-forcing-1978-2013_regridded_conservative_norm_stats.nc",
        "description": "Monthly mean forcing normalization stats",
    },
    "daily_averaged_norm_stats": {
        "url": f"{ZENODO_DATA_BASE_URL}/daily_averaged_aimip_norm_stats.nc?download=1",
        "dest": Path("stats") / "daily_averaged_aimip_norm_stats.nc",
        "description": "Daily averaged AIMIP normalization stats",
    },
    "residual_stats": {
        "url": f"{ZENODO_DATA_BASE_URL}/residual_std_aimip_213852624.nc?download=1",
        "dest": Path("stats") / "residual_std_aimip_213852624.nc",
        "description": "Residual std stats for ArchesWeatherGen",
    },
}


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


def download_data(output_directory: str | Path = ".", force: bool = False) -> None:
    """Download AIMIP ERA5 data, forcings, and normalization stats from Zenodo."""
    output_directory = Path(output_directory)

    for key, info in DATA_FILES.items():
        destination = output_directory / info["dest"]
        if destination.is_file() and not force:
            print(f"Skipping {info['description']} (already exists at {destination})")
            continue

        print(f"Downloading {info['description']}...")
        _download_file(info["url"], destination)
        print(f"Saved to {destination}")


def main() -> None:
    """Run the AIMIP dataset downloader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        default=".",
        help="Root directory in which data/ and stats/ directories are created (default: .).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download and overwrite existing local files.",
    )
    args = parser.parse_args()
    download_data(args.output_directory, force=args.force)


if __name__ == "__main__":
    main()
