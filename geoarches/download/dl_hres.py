"""Download HRES initial conditions from Weather Bench and store in .zarr files.

Used as groundtruth to evaluate IFS forecasts.

Format follows ERA5 (to use same ERA5 dataloader):
  1. Chunked on time dimension to enable efficient data access.
  2. Stored by hour because assumes 24 hour lead times when reading.
"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import xarray as xr

from geoarches.dataloaders import era5


def download_year(
    ds,
    year,
    hour,
    time_chunk_size,
    folder,
    force=False,
):
    """Downloads one year into a zarr filepath.

    This allows parallelizing the download. Storing year in separate files
    allows for easy restarting if job is pre-empted.
    """
    filepath = Path(folder) / f"{year}_{hour}h.zarr"
    if filepath.exists():
        if force:
            shutil.rmtree(filepath)
    if not filepath.exists():
        ds.sel(time=(ds.time.dt.year.isin([year])) & (ds.time.dt.hour.isin([hour]))).chunk(
            dict(
                time=time_chunk_size,
                level=3,
                longitude=240,
                latitude=121,
            )
        ).to_zarr(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Where to store downloads.")
    parser.add_argument(
        "--years",
        nargs="+",  # Accepts 1 or more arguments as a list.
        type=int,
        default=list(range(2016, 2023)),
        help="Year(s) to download. By default downloads all 2016-2022.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force redownload if file already exists."
    )
    parser.add_argument(
        "--time_chunk_size",
        default=100,
        type=int,
        help="Size of chunking along time dimension. Avoids 1 file per time index.",
    )
    parser.add_argument(
        "--max_threads",
        default=7,
        type=int,
        help="Number of threads to start up at once. Each thread downloads a year. Reduce if memory constraints.",
    )
    args = parser.parse_args()

    Path(args.folder).mkdir(parents=True, exist_ok=True)

    file = "2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
    ds = xr.open_zarr("gs://weatherbench2/datasets/hres_t0/" + file, chunks="auto")
    # Drop zarr input chunk encoding so that we can rechunk when we write.
    ds = ds.drop_encoding()

    vars = era5.level_variables + era5.surface_variables
    vars += ["10m_wind_speed", "wind_speed"]
    ds = ds[vars]
    ds = ds.sel(level=[500, 700, 850])

    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        for year in args.years:
            for hour in [0, 12]:
                print(year)
                executor.submit(
                    download_year,
                    ds,
                    year=year,
                    hour=hour,
                    time_chunk_size=args.time_chunk_size,
                    folder=args.folder,
                    force=args.force,
                )
