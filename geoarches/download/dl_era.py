"""
Download ERA5 from Weather Bench and store in netcdf files.
Used as groundtruth to train and evaluate ML-based models.

Format:
 - Netcdf bypasses limitation of number of files on cluster.
 - Stored by year and hour:
     1. Chunked on time dimension to enable efficient data access.
     2. Stored by hour because assumes 24 hour lead times when reading.
"""

import argparse
import os
from pathlib import Path

import xarray as xr

obs_path = "gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
climatology_path = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr"

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default="data/era5_240/full/", help="where to store outputs")
parser.add_argument(
    "--years",
    nargs="+",  # Accepts 1 or more arguments as a list.
    type=int,
    default=list(range(1979, 2022)),
    help="Year(s) to download. By default downloads all 1979-2021.",
)
parser.add_argument("--years", default="", type=str, help="year to download")
parser.add_argument("--clim", action="store_true", help="whether to download climatology")

args = parser.parse_args()

Path(args.folder).mkdir(parents=True, exist_ok=True)

clim_tgt = Path(args.folder).parent.joinpath("era5_240_clim.nc")
if not clim_tgt.exists() and (args.clim or not args.years):
    obs_xarr = xr.open_zarr(climatology_path)
    obs_xarr.to_netcdf(clim_tgt)
    exit()

for year in args.years:
    for hour in (0, 6, 12, 18):
        fname = Path(args.folder) / f"era5_240_{year}_{hour}h.nc"
        if Path(fname).exists() and os.stat(fname).st_size < 4580000000:
            os.remove(fname)  # file is corrupted
        if not Path(fname).exists():
            obs_xarr = xr.open_zarr(obs_path)

            ds = obs_xarr.sel(time=obs_xarr.time.dt.year.isin([year]))
            ds2 = ds.sel(time=ds.time.dt.hour.isin([hour]))

            ds2.to_netcdf(fname)
