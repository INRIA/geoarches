"""Compute percentiles per latitude/longitude across a given time range.

Computes both high (99.99th, 99.9th and 99th) and low (0.01st, 0.1st and 1st) climatological percentiles.
Used for brier score computation.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from geoarches.dataloaders import era5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data_path",
        type=str,
        default="data/era5_240/full",
        help="Path holding data to compute quantiles on.",
    )
    parser.add_argument(
        "--input_data_file_extension",
        type=str,
        default=".nc",
        choices={".nc", ".zarr"},
        help="File extension for data to compute quantiles on",
    )
    parser.add_argument(
        "--save_path",
        default="src/geoarches/stats/era5-quantiles-2016_2022.nc",
        type=str,
        help="Full filepath to save quantiles netcdf to.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2016", "2017", "2018", "2019", "2020", "2021", "2022"],
        help="Year of groundtruth to calculate quantiles on.",
    )
    parser.add_argument(
        "--vars",
        nargs="*",  # Accepts 0 or more arguments as a list.
        default=era5.surface_variables + era5.level_variables + ["10m_wind_speed", "wind_speed"],
        help="Variables in dataset to compute quantiles over. Set to empty to compute over all variables.",
    )
    args = parser.parse_args()

    # 6-hourly data within year range.
    filename_filter = lambda x: any(substring in x for substring in args.years)
    files = sorted(
        [
            str(x)
            for x in Path(args.input_data_path).glob(f"*{args.input_data_file_extension}")
            if filename_filter(x.name)
        ]
    )
    print(f"Computing quantiles over {len(files)} files.")

    def _preprocess(ds):
        out = ds
        if args.vars:
            out = out[args.vars]
        return out

    engine = "netcdf4" if args.input_data_file_extension == ".nc" else "zarr"
    ds = xr.open_mfdataset(
        files, concat_dim="time", combine="nested", preprocess=_preprocess, engine=engine
    )
    # Rechunk time into a single chunk (Required for quantile computation).
    ds = ds.chunk({"time": -1})

    q = np.array([0.01, 0.1, 1, 99, 99.9, 99.99]) / 100
    quantile_ds = ds.quantile(q, dim="time")

    quantile_ds.to_netcdf(args.save_path)
