"""Download IFS ENS forecasts from Weather Bench and store in .zarr files chunked by time.

Downloads by year (default 2020).
Optionally download fraction of ensemble members (using args.split and args.nplits).
"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import xarray as xr
from geoarches.dataloaders import era5
from tqdm import tqdm

PRED_TIME_DELTAS = range(0, 61, 4)


def download_time_slice(
    ds,
    start_index,
    end_index,
    time_chunk_size,
    folder,
    year,
    force=False,
):
    """Downloads one time slice of dataset into a zarr filepath.

    This allows parallelizing the download. Storing time chunks in separate files
    allows for easy restarting if job is pre-empted.
    """
    filepath = Path(folder) / f"{year}-{start_index:03}.zarr"
    if filepath.exists():
        if force:
            shutil.rmtree(filepath)
    if not filepath.exists():
        # Store by chunks along the time dimension to match input chunking.
        ds.isel(time=range(start_index, end_index)).chunk(
            dict(
                time=time_chunk_size,
                number=50,
                prediction_timedelta=len(PRED_TIME_DELTAS),
                level=3,
                longitude=240,
                latitude=121,
            )
        ).to_zarr(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Where to save downloads.")
    parser.add_argument("--year", default=2020, type=int, help="Year to download.")
    parser.add_argument(
        "--force", action="store_true", help="Force redownload if file already exists."
    )
    parser.add_argument(
        "--max_threads",
        default=10,
        type=int,
        help="Number of threads to start up at once. Each thread downloads a time slice. Reduce if memory constraints.",
    )
    parser.add_argument("--start", default=0, type=int, help="Start time index to download.")
    parser.add_argument("--end", default=None, type=int, help="End time index to download.")
    parser.add_argument(
        "--time_chunk_size", default=5, type=int, help="Size of chunking along time dimension."
    )
    args = parser.parse_args()

    Path(args.folder).mkdir(parents=True, exist_ok=True)

    file = "2018-2022-240x121_equiangular_with_poles_conservative.zarr"
    ds = xr.open_zarr("gs://weatherbench2/datasets/ifs_ens/" + file, chunks="auto")
    ds = ds.where(ds.time.dt.year == args.year, drop=True)

    vars = era5.level_variables + era5.surface_variables
    vars.remove("vertical_velocity")
    ds = ds[vars]

    ds = ds.sel(time=ds.time.dt.hour.isin([0, 12]))
    ds = ds.isel(prediction_timedelta=PRED_TIME_DELTAS)  # every 1 day

    total = ds.time.shape[0]
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        for i in range(args.start, args.end or total, args.time_chunk_size):
            executor.submit(
                download_time_slice,
                ds,
                start_index=i,
                end_index=min(total, i + args.time_chunk_size),
                time_chunk_size=args.time_chunk_size,
                folder=args.folder,
                year=args.year,
                force=args.force,
            )
