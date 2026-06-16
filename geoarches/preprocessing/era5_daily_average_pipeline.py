r"""Pipeline to compute daily averaged ERA5 data.

Example of how to run this:
To run locally (good for testing with small subsets):
blaze run --config=dmtf //third_party/py/geoarches/preprocessing:era5_daily_average_pipeline -- \
  --flume_exec_mode=LOCAL_PROCESSES \
  --target_years="2022" \
  --alsologtostderr  \
  --debug

To run on Borg:
blaze run -c opt //third_party/py/geoarches/preprocessing:era5_daily_average_pipeline.par -- \
  --flume_exec_mode=BORG \
  --flume_borg_cells=lu \
  --flume_borg_accounting_charged_user_name=deepmind-resources-borg \
  --flume_use_batch_scheduler \
  --flume_batch_scheduler_strategy=RUN_SOON \
  --alsologtostderr
"""

import os

import apache_beam as beam
import xarray as xr
from absl import app, flags, logging
from google3.pipeline.flume.py import runner
from google3.pyglib import atomic_file, gfile

# Define base paths
INPUT_BASE_PATH = "/cns/prod/home/slalom/datasets/era5_1deg_rechunked/full/"
OUTPUT_BASE_PATH = "/cns/prod/home/slalom/datasets/era5_1deg_rechunked/daily_averaged"

ROLLING_WINDOW = flags.DEFINE_integer(
    "rolling_window",
    4,
    "Size of the rolling window to use for averaging. This is the number of"
    " hours to average over. The rolling window is centered on the target"
    " hour, so a rolling window of 4 will use 3 hours before and"
    " the target hour. Note that the first few timesteps"
    " will not have enough preceding datapoints to compute a"
    " rolling average. The first valid datapoint will be at index"
    " rolling_window - 1.",
)
# Define target years to process
# The pipeline will process data for each year in this list.
# For each target year, it will read data from target_year and target_year - 1.
_TARGET_YEARS = flags.DEFINE_list(
    "target_years",
    [str(y) for y in range(1979, 2025)],
    "Comma-separated list of years for which to compute daily averages.",
)
_DEBUG = flags.DEFINE_boolean("debug", False, "If true, run in debug mode with a subset of data.")


def atomic_netcdf_write(ds: xr.Dataset, path: str):
    """Writes an xarray dataset to a netcdf file atomically."""
    with atomic_file.open_for_writing(path, mode="b") as output_gfile:
        output_gfile.write(ds.to_netcdf())


def _get_input_files_for_year(year: int) -> list[str]:
    """Returns a list of input files for a given target year."""
    # To compute rolling averages at year boundaries, we need data from the
    # previous year. The rolling average uses preceding data.
    input_years = [year]
    if year > 1978:
        # For 1978, data for 1977 is not available.
        input_years.insert(0, year - 1)
    logging.info(
        "Using input years: %s for target year: %s",
        input_years,
        year,
    )

    file_patterns = []
    for input_year in input_years:
        # Based on sample filenames, hour format can be '0h' or '06h'.
        # TODO(yhasson): Use gfile.Glob to find all files for a given year to be
        # robust to changes in the file naming scheme.
        for hour in ["0", "06", "12", "18"]:
            file_patterns.append(f"{INPUT_BASE_PATH}era5_{input_year}_{hour}h.nc")
    return file_patterns


class LoadCombineAndAverageDataDoFn(beam.DoFn):
    """Loads, concatenates, computes rolling average, and preprocesses data."""

    def process(self, year_and_files):
        target_year, files_for_year = year_and_files
        open_files = []
        has_previous_year_data = False
        try:
            # Use Dask for out-of-core computation by opening files in chunked mode
            # and deferring computation.
            open_files = [gfile.Open(fp, "rb") for fp in files_for_year]
            datasets = []
            for f, file_path in zip(open_files, files_for_year):
                logging.info("Loading file: %s", file_path)
                is_previous_year_file = f"era5_{target_year - 1}_" in file_path
                ds = xr.open_dataset(f, engine="h5netcdf", chunks="auto")

                if is_previous_year_file:
                    has_previous_year_data = True
                    if "time" in ds.dims:
                        num_time_steps = ds.dims["time"]
                        ds = ds.isel(time=slice(max(0, num_time_steps - 10), None))

                if _DEBUG.value:
                    vars_to_keep = [v for v in ["2m_temperature"] if v in ds.data_vars]
                    if vars_to_keep:
                        ds = ds[vars_to_keep]
                    ds = ds.isel({"time": slice(0, 10)})

                ds = ds.drop_vars(["number", "expver"], errors="ignore")
                ds = ds.swap_dims({"time": "valid_time"})
                if "valid_time" in ds.data_vars:
                    ds = ds.set_coords("valid_time")
                datasets.append(ds)

            if not datasets:
                raise ValueError(f"No datasets found for year: {target_year}")

            # The following operations are lazy because datasets are Dask-backed.
            hourly_ds = xr.concat(datasets, dim="valid_time")
            hourly_ds = hourly_ds.sortby("valid_time")
            averaged_ds = hourly_ds.rolling(valid_time=ROLLING_WINDOW.value).mean()
            start_date = f"{target_year}-01-01T00:00:00"
            end_date = f"{target_year}-12-31T23:59:59"
            averaged_ds = averaged_ds.sel(valid_time=slice(start_date, end_date))
            if not has_previous_year_data:
                # First 3 timesteps will be nan since window size < rolling window size.
                averaged_ds = averaged_ds.isel(valid_time=slice(ROLLING_WINDOW.value - 1, None))

            if averaged_ds.valid_time.size > 0:
                # Load time coordinates to determine hours for partitioning.
                hours = averaged_ds.valid_time.dt.hour.values
                expected_hours = [0, 6, 12, 18]
                for hour in expected_hours:
                    ds_hour = averaged_ds.isel(valid_time=(hours == hour))
                    if ds_hour.valid_time.size > 0:
                        # Load data for this partition into memory before yielding.
                        yield (target_year, hour, ds_hour.load())
            beam.metrics.Metrics.counter("LoadCombineAndAverageData", "Success").inc()
        finally:
            for f in open_files:
                f.close()


class WriteDataDoFn(beam.DoFn):
    """Writes an xarray Dataset to a NetCDF file in CNS."""

    def __init__(self, output_base_path):
        self.output_base_path = output_base_path

    def setup(self):
        # Create the output directory if it doesn't exist.
        # gfile.MakeDirs is recursive.
        if not gfile.Exists(self.output_base_path):
            logging.info("Creating output directory: %s", self.output_base_path)
            gfile.MakeDirs(self.output_base_path)

    def process(self, year_hour_and_ds):
        target_year, hour, ds_hour = year_hour_and_ds

        if ds_hour is None or ds_hour.valid_time.size == 0:
            logging.warning("Received empty dataset for year %s, hour %s.", target_year, hour)
            beam.metrics.Metrics.counter(
                "WriteData", f"EmptyDataset_{target_year}_{hour:02d}h"
            ).inc()
            return

        ds_hour = ds_hour.chunk({"valid_time": 1, "latitude": -1, "longitude": -1})
        logging.info("Writing year %s, hour %02dh.", target_year, hour)

        output_filename = f"era5_{target_year}_{hour:02d}h.nc"
        output_path = os.path.join(self.output_base_path, output_filename)

        # Write the xarray Dataset to a NetCDF file in CNS.
        atomic_netcdf_write(ds_hour, output_path)
        logging.info("Successfully wrote: %s", output_path)
        beam.metrics.Metrics.counter("WriteData", f"Wrote_{target_year}_{hour:02d}h").inc()


def build_pipeline(p: beam.Pipeline, target_years: list[int]):
    """Builds the FlumePython pipeline for the given target years."""
    # 1. Create a PCollection where each element is a target year to process.
    years_pcol = p | "CreateTargetYears" >> beam.Create(target_years)

    # 2. For each year, determine the input files and create a (year, files)
    # tuple.
    files_to_process = years_pcol | "GetInputFiles" >> beam.Map(
        lambda year: (year, _get_input_files_for_year(year))
    )

    # 3. Load, combine, and average data for each year.
    averaged_pcol = files_to_process | (
        "LoadCombineAndAverageData" >> beam.ParDo(LoadCombineAndAverageDataDoFn())
    )

    # 4. Write partitioned data to CNS.
    if _DEBUG.value:
        output_base_path = OUTPUT_BASE_PATH + "_debug/"
    else:
        output_base_path = OUTPUT_BASE_PATH
    # TODO(yhasson): Write and read data using xarray-beam.
    _ = averaged_pcol | "WriteData" >> beam.ParDo(WriteDataDoFn(output_base_path))


def main(unused_argv):
    # Command line arguments are parsed by FlumeRunner.
    # Ensure Flume is initialized.
    runner.program_started()

    target_years = [int(y) for y in _TARGET_YEARS.value]
    logging.info("Starting Flume pipeline for target years: %s", target_years)

    def pipeline_fn(p):
        build_pipeline(p, target_years)

    # Use FlumeRunner to get a UI link and manage execution.
    runner.FlumeRunner().run(pipeline_fn).wait_until_finish()

    logging.info("Flume pipeline finished for target years: %s", target_years)


if __name__ == "__main__":
    app.run(main)
