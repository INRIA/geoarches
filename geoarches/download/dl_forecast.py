"""Download forecasts (ie. ENS, NeuralGCM) from Weather Bench and store in .zarr files chunked by time.

Downloads by year (default 2020).

Implementation:
    - Chunks by time (increase --time_chunk_size increase time chunk stored per .zarr file):
        1. to enable easy restart after job preempt or timeout (by using --start and --force args).
        2. to enable efficient access by time slice in dataloader.
    - Parallelizes download (by using --max_threads arg).
    - Efficient (use --variables to only download vars you need).
"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import xarray as xr

PRED_TIME_DELTAS = [timedelta(days=i) for i in range(0, 16)]


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
                prediction_timedelta=len(PRED_TIME_DELTAS),
                level=-1,
                longitude=240,
                latitude=121,
            )
        ).to_zarr(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Where to save downloads.")
    parser.add_argument(
        "--input_address",
        type=str,
        default="gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr",
        help="Where to read data from.",
    )
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
    parser.add_argument(
        "--variables",
        default=[
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
            "temperature",
            "specific_humidity",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
        ],
        nargs="+",
        help="Variables to download. By default level and surface vars.",
    )
    args = parser.parse_args()

    Path(args.folder).mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(args.input_address, chunks="auto")
    ds = ds.where(ds.time.dt.year == args.year, drop=True)

    ds = ds[args.variables]

    ds = ds.sel(time=ds.time.dt.hour.isin([0, 12]))
    ds = ds.sel(prediction_timedelta=PRED_TIME_DELTAS)  # every 1 day
    ds = ds.sel(level=[500, 700, 850])

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
