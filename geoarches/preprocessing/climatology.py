r"""Compute climatology: means over time period for each gridpoint, level.

Useful when dataset is split over multiple files.

Example of how to run this:
To run locally (good for testing with small subsets):
blaze run //third_party/py/geoarches/preprocessing:climatology -- \
  --flume_exec_mode=LOCAL_PROCESSES \
  --alsologtostderr  \
  --debug

To run on Borg:
blaze run -c opt //third_party/py/geoarches/preprocessing:climatology.par -- \
  --flume_exec_mode=BORG \
  --flume_borg_cells=lu \
  --flume_borg_accounting_charged_user_name=deepmind-resources-borg \
  --flume_use_batch_scheduler \
  --flume_batch_scheduler_strategy=RUN_SOON \
  --alsologtostderr \
  --start_year 1979 \
  --end_year_inclusive 2024
"""

import apache_beam as beam
import xarray as xr
from absl import app, flags, logging
from etils import epath
from google3.pipeline.flume.py import runner

ZARR_PATH = flags.DEFINE_string(
    "path",
    "/cns/prod/home/slalom/datasets/era5_1x1/full",
    "Base path for zarr files.",
)
OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    "/cns/lu-d/home/kastor/datasets/era5_1x1/clim/",
    "Output path for climatology files.",
)
START_YEAR = flags.DEFINE_integer(
    "start_year",
    1979,
    "Start year for climatology.",
)
END_YEAR_INCLUSIVE = flags.DEFINE_integer(
    "end_year_inclusive",
    2024,
    "End year inclusive for climatology.",
)
PER_MONTH = flags.DEFINE_boolean(
    "per_month",
    False,
    "If true, compute climatology for each month separately. Otherwise,"
    " computes mean over the entire time period.",
)
_DEBUG = flags.DEFINE_boolean("debug", False, "If true, run in debug mode with a subset of data.")
_OVERWRITE = flags.DEFINE_boolean("overwrite", False, "If true, overwrite existing zarr file.")


def rename_dims_coords(xrdataset: xr.Dataset) -> xr.Dataset:
    """Renames dimensions and coordinates to be consistent."""
    dimension_mapping = {
        "valid_time": "time",
        "pressure_level": "level",
        "lat": "latitude",
        "lon": "longitude",
    }

    for old_name, new_name in dimension_mapping.items():
        if old_name in xrdataset.dims and new_name not in xrdataset.dims:
            xrdataset = xrdataset.rename({old_name: new_name})
        elif old_name in xrdataset.coords and new_name not in xrdataset.coords:
            xrdataset = xrdataset.rename({old_name: new_name})

    coords_to_drop = [c for c in xrdataset.coords if c not in xrdataset.dims]
    xrdataset = xrdataset.drop_vars(coords_to_drop)
    # Drop duplicates in time.
    xrdataset = xrdataset.drop_duplicates("time")
    return xrdataset


class ReadTimeSlice(beam.DoFn):
    """Reads a single ERA5 file and filters for a given time slice."""

    def __init__(self, start_year: int, end_year_inclusive: int):
        self.start_year = start_year
        self.end_year_inclusive = end_year_inclusive

        self.processed_files_counter = beam.metrics.Metrics.counter(
            "ReadTimeSlice", "processed_files"
        )

    def process(self, file_path):
        self.processed_files_counter.inc()
        ds = xr.open_zarr(file_path, consolidated=True)
        ds = ds.squeeze(drop=True)
        ds = rename_dims_coords(ds)
        ds = ds.sel(time=slice(f"{self.start_year}-01-01", f"{self.end_year_inclusive}-12-31"))
        if ds.time.size == 0:
            return None
        if _DEBUG.value:
            ds = ds.isel(time=slice(2))
        yield ds


class AvgDsCombineFn(beam.CombineFn):
    """Combiner to average over time dim of xarray Datasets.

    Average per gridpoint, per level.
    """

    def __init__(self):
        self.datasets_processed = beam.metrics.Metrics.counter(
            "AvgDsCombineFn", "datasets_processed"
        )
        self.timestamps_processed = beam.metrics.Metrics.counter(
            "AvgDsCombineFn", "timestamps_processed"
        )

    def create_accumulator(self):
        return None, 0

    def add_input(
        self,
        accumulator: tuple[xr.Dataset | None, int | xr.Dataset],
        input_ds: xr.Dataset,
    ) -> tuple[xr.Dataset | None, int | xr.Dataset]:
        self.datasets_processed.inc()
        if input_ds is None:
            return accumulator

        sum_ds, count = accumulator

        if PER_MONTH.value:
            current_sum = input_ds.groupby("time.month").sum(dim="time", skipna=False)
            current_count = input_ds.groupby("time.month").count(dim="time")
        else:
            current_sum = input_ds.sum(dim="time", skipna=False)  # keep NaNs per gridpt
            current_count = input_ds.time.size
        self.timestamps_processed.inc(input_ds.time.size)

        if sum_ds is None:
            return current_sum, current_count
        return sum_ds + current_sum, count + current_count

    def merge_accumulators(self, accumulators):
        sum_ds, count = None, None
        for partial_sum_ds, partial_count in accumulators:
            if partial_sum_ds is None:
                continue
            if sum_ds is None:
                sum_ds = partial_sum_ds
                count = partial_count
            else:
                sum_ds += partial_sum_ds
                count += partial_count
        return sum_ds, count

    def extract_output(self, accumulator):
        sum_ds, count = accumulator
        if sum_ds is None:
            return None
        return sum_ds / count


def write_to_zarr(dataset, output_path):
    if dataset is None:
        raise ValueError("No data to write.")

    dataset.to_zarr(output_path, mode="w", consolidated=True)
    logging.info("Successfully wrote to %s", output_path)


def main(argv):
    del argv  # Unused.
    runner.program_started()

    files = list(epath.Path(ZARR_PATH.value).glob("*.zarr"))
    if not files:
        raise ValueError(f"No files found in {ZARR_PATH.value}")
    if _DEBUG.value:
        logging.warning("Running in DEBUG mode with only 1 files.")
        files = [f for f in files if str(START_YEAR.value) in f.name][:1]

    output_path = epath.Path(OUTPUT_PATH.value)
    if _DEBUG.value:
        output_path = output_path / "debug"
    prefix = "monthly_" if PER_MONTH.value else ""
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_path / f"{prefix}climatology_{START_YEAR.value}_{END_YEAR_INCLUSIVE.value}.zarr"
    )
    logging.info("Will write results to %s", output_path)
    if output_path.exists() and not _OVERWRITE.value:
        raise ValueError(f"Output file already exists, skipping: {output_path}")

    def pipeline(root):
        _ = (
            root
            | "ListFiles" >> beam.Create(files)
            | "ReshuffleFiles" >> beam.Reshuffle()
            | "ReadTimeSlice"
            >> beam.ParDo(ReadTimeSlice(START_YEAR.value, END_YEAR_INCLUSIVE.value))
            | "ComputeAverages" >> beam.CombineGlobally(AvgDsCombineFn())
            | "WriteToZarr" >> beam.Map(write_to_zarr, output_path)
        )

    runner.FlumeRunner().run(pipeline)
    logging.info("Pipeline finished.")


if __name__ == "__main__":
    app.run(main)
