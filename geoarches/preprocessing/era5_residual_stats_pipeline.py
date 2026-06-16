r"""Beam pipeline to compute residual statistics between predictions and ERA5 daily averaged data.

Example of how to run this:
To run locally (good for testing with small subsets):
blaze run //third_party/py/geoarches/preprocessing:era5_residual_stats_pipeline -- \
  --flume_exec_mode=LOCAL_PROCESSES \
  --alsologtostderr  \
  --debug

To run on Borg:
blaze run -c opt //third_party/py/geoarches/preprocessing:era5_residual_stats_pipeline.par -- \
  --flume_exec_mode=BORG \
  --flume_borg_cells=lu \
  --flume_borg_accounting_charged_user_name=deepmind-resources-borg \
  --flume_use_batch_scheduler \
  --flume_batch_scheduler_strategy=RUN_SOON \
  --alsologtostderr
"""

import datetime
import os

import apache_beam as beam
import numpy as np
import pandas as pd
import xarray as xr
import xarray_beam as xbeam
from absl import app, flags, logging
from etils import epath
from google3.pipeline.flume.py import runner
from google3.pyglib import gfile

XID_WID = flags.DEFINE_string(
    "xid_wid",
    "209004791/avg",
    "XID/WID of the experiment for prediction paths.",
)
PREDS_PATH = flags.DEFINE_string(
    "preds_path",
    "/cns/prod/home/slalom/singhren/{xid_wid}/",
    "Base path for prediction files. Can contain {xid_wid}.",
)
GROUNDTRUTH_PATH = flags.DEFINE_string(
    "groundtruth_path",
    "/cns/prod/home/slalom/datasets/era5_1deg_rechunked/daily_averaged/",
    "Path to ground truth files.",
)
OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "/cns/prod/home/slalom/{username}/residual_stats/{xid_wid}/",
    "Output directory for residual stats. Can contain {xid_wid} and {username}.",
)
_DEBUG = flags.DEFINE_boolean("debug", False, "If true, run in debug mode with a subset of data.")


class SumDsCombineFn(beam.CombineFn):
    """Combiner to sum xarray Datasets."""

    def create_accumulator(self):
        return None

    def add_input(self, accumulator, input_ds):
        if accumulator is None:
            return input_ds
        return accumulator + input_ds

    def merge_accumulators(self, accumulators):
        total = None
        for acc in accumulators:
            if acc is not None:
                if total is None:
                    total = acc
                else:
                    total += acc
        return total

    def extract_output(self, accumulator):
        return accumulator


def update_time_to_next_state_timestamp(pred_ds):
    """Updates the time coordinate to the next state timestamp to align with GT era5 files."""
    next_state_time = pred_ds.time + pred_ds.prediction_timedelta.values[0]
    return pred_ds.assign_coords({"time": next_state_time})


def rename_dims_coords(xrdataset):
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


def glob_files(pattern, start_year=None, end_year_inclusive=None):
    """Globs files matching pattern, with optional year filtering."""
    files = gfile.Glob(pattern)
    if start_year and end_year_inclusive:
        return [
            f for f in files if any(str(y) in f for y in range(start_year, end_year_inclusive + 1))
        ]
    elif start_year:
        return [f for f in files if str(start_year) in f]
    else:
        return files


class ComputeDiff(beam.DoFn):
    """Computes difference between GT and Preds for intersecting times."""

    def __init__(self, gt_path):
        self.gt_path = gt_path
        self.processed_files_counter = beam.metrics.Metrics.counter(
            "ComputeDiff", "processed_files"
        )

    def get_gt_files_for_pred(self, pred_ds: xr.Dataset) -> list[str]:
        """Returns list of GT files covering the time range of pred_ds."""
        time_stamps = pd.to_datetime(pred_ds.time.values)
        years = time_stamps.year.unique()
        hours = time_stamps.hour.unique()

        gt_files = []
        for year in years:
            for hour in hours:
                # We expect gt files to contain only 0, 6, 12, 18h data.
                if hour not in [0, 6, 12, 18]:
                    raise ValueError(f"GT files only contain 0, 6, 12, 18h data, got {hour}h.")
                else:
                    gt_files.append(os.path.join(self.gt_path, f"era5_{year}_{hour:02d}h.nc"))

        # In debug mode, we might want to limit files loaded.
        if _DEBUG.value:
            gt_files = gt_files[:1]
        return gt_files

    def process(self, pred_path):
        self.processed_files_counter.inc()
        with gfile.Open(pred_path, "rb") as f:
            # Load predicted data, dropping duplicates in time.
            pred_ds = rename_dims_coords(xr.open_dataset(f, engine="h5netcdf"))
            pred_ds = update_time_to_next_state_timestamp(pred_ds)
            pred_times = pred_ds.time.values
            gt_files = self.get_gt_files_for_pred(pred_ds)
            logging.info(
                "Found %d GT files for pred file %s, first: %s, last: %s",
                len(gt_files),
                pred_path,
                gt_files[0],
                gt_files[-1],
            )

        gt_times_all_list = []
        for f_path in gt_files:
            with gfile.Open(f_path, "rb") as f:
                gt_times_all_list.append(
                    rename_dims_coords(xr.open_dataset(f, engine="h5netcdf")).time.values
                )
        gt_times_all = np.unique(np.concatenate(gt_times_all_list))

        common_times = np.intersect1d(pred_times, gt_times_all)
        logging.info("Found %d common times for pred file %s", len(common_times), pred_path)
        if common_times.size == 0:
            raise ValueError(f"No overlapping GT data found for pred file {pred_path}")
        if not _DEBUG.value and len(common_times) != len(pred_times):
            missing_times = set(pred_times) - set(common_times)
            raise ValueError(
                f"Missing times {missing_times} in pred file {pred_path} which are in"
                " common with GT data."
            )

        gt_ds_list = []
        for f_path in gt_files:
            with gfile.Open(f_path, "rb") as f:
                ds = rename_dims_coords(xr.open_dataset(f, engine="h5netcdf"))
                file_common_times = np.intersect1d(ds.time.values, common_times)
                if file_common_times.size > 0:
                    try:
                        gt_ds_list.append(ds.sel(time=pd.DatetimeIndex(file_common_times)).load())
                    except KeyError as exc:
                        raise ValueError(
                            f"Couldn't get {file_common_times} for file {f_path}"
                        ) from exc
        gt_ds = xr.concat(gt_ds_list, dim="time").sortby("time")

        with gfile.Open(pred_path, "rb") as f:
            pred_chunk = (
                update_time_to_next_state_timestamp(
                    rename_dims_coords(xr.open_dataset(f, engine="h5netcdf"))
                )
                .sel(time=pd.DatetimeIndex(common_times))
                .squeeze(drop=True)  # Drop prediction_timedelta dim.
                .load()
            )

        # Ensure alignment and subtract
        pred_chunk, gt_chunk = xr.align(pred_chunk, gt_ds, join="inner")
        diff = gt_chunk - pred_chunk

        for var in diff.data_vars:
            residual = diff[var]
            agg_dims = [d for d in ("latitude", "longitude", "time") if d in residual.dims]
            key = xbeam.Key(offsets={}, vars={var})
            count = residual.notnull().sum(dim=agg_dims)
            variable_sum = residual.sum(dim=agg_dims, skipna=True)
            variable_sq_sum = (residual**2).sum(dim=agg_dims, skipna=True)
            sums_ds = xr.Dataset(
                {
                    "sum": variable_sum,
                    "sq_sum": variable_sq_sum,
                    "count": count,
                }
            )
            beam.metrics.Metrics.counter("ComputeDiff", f"partials_yielded_{var}").inc()
            beam.metrics.Metrics.counter("ComputeDiff", f"timestamps_processed_{var}").inc(
                residual.time.size
            )
            yield key, sums_ds


class ComputeStdDevDoFn(beam.DoFn):
    """Computes std dev from partial sums and counts."""

    def process(self, element):
        key, sums_chunk = element
        var = list(key.vars)[0]
        beam.metrics.Metrics.counter("ComputeStdDev", f"std_computed_{var}").inc()
        mean = sums_chunk["sum"] / sums_chunk["count"]
        logging.info("Computed %s mean for %s data points", var, sums_chunk["count"])
        mean_sq = sums_chunk["sq_sum"] / sums_chunk["count"]
        variance = mean_sq - mean**2
        std = np.sqrt(variance)
        std_ds = xr.Dataset({var: std})
        std_ds = std_ds.expand_dims(statistic=["diff_std"])  # To match geoarches stats format.
        yield key, std_ds


def main(argv):
    del argv  # Unused.
    runner.program_started()

    preds_path = PREDS_PATH.value.format(xid_wid=XID_WID.value)
    gt_path = GROUNDTRUTH_PATH.value
    out_dir = OUTPUT_DIR.value.format(
        xid_wid=XID_WID.value, username=os.environ.get("USER", "yhasson")
    )
    pred_files = glob_files(os.path.join(preds_path, "*.nc"))
    if not pred_files:
        raise ValueError(f"No prediction files found in {preds_path}")

    if _DEBUG.value:
        logging.warning("Running in DEBUG mode with only one file.")
        now = datetime.datetime.now()
        out_dir = os.path.join(out_dir, f"debug_{now.strftime('%Y_%m_%d_%H_%M_%S')}")
        pred_files = pred_files[:1]

    out_path = os.path.join(out_dir, "residual_stddev.zarr")
    epath.Path(out_dir).mkdir(parents=True, exist_ok=True)
    logging.info("Will write results to %s", out_path)

    def pipeline(root):
        partials = (
            root
            | "ListFiles" >> beam.Create(pred_files)
            | "ReshuffleFiles" >> beam.Reshuffle()
            | "ComputeDiffs" >> beam.ParDo(ComputeDiff(gt_path))
            | "ReshuffleDiffs" >> beam.Reshuffle()
        )

        total_sums = partials | "CombineSums" >> beam.CombinePerKey(SumDsCombineFn())

        std_devs = total_sums | "ComputeStdDev" >> beam.ParDo(ComputeStdDevDoFn())

        final_result = std_devs | "ConsolidateVars" >> xbeam.ConsolidateVariables()

        template = beam.pvalue.AsSingleton(
            final_result | "CreateTemplate" >> beam.MapTuple(lambda k, ds: xbeam.make_template(ds))
        )
        _ = final_result | "WriteToZarr" >> xbeam.ChunksToZarr(out_path, template=template)

    runner.FlumeRunner().run(pipeline)
    logging.info("Pipeline finished.")


if __name__ == "__main__":
    app.run(main)
