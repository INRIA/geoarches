"""Script to compute and store metrics defined in metric_registry.py.

Optionally caches intermediate metrics in case of process preemption (set --cache_metrics_every_nbatches).

How to use:
 1. Run model inference and store outputs in xarray format (ie. zarr or netcdf).
 2. Define metric and its arguments using register_class() in metric_registry.py (see file for examples).
 3. Run this script, passing in the metric name to --metrics.

Example commandline:
    python -m geoarches.evaluation.eval_multistep \
        --pred_path data/ifs_ens/ \
        --output_dir evalstore/ens/ \
        --groundtruth_path data/hres/ \
        --level_vars geopotential u_component_of_wind v_component_of_wind temperature specific_humidity \
        --metrics hres_brier_skill_score
"""

import argparse
import copy
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from tensordict.tensordict import TensorDict
from torch.utils.data import default_collate
from tqdm import tqdm

from geoarches.dataloaders import era5
from geoarches.dataloaders.netcdf import default_dimension_indexers
from geoarches.metrics.label_wrapper import convert_metric_dict_to_xarray

from . import metric_registry


def _custom_collate_fn(batch):
    """
    Custom collate function to handle batches of containers with TensorDict elements.

    Args:
        batch (list of dict): A batch of data samples, where each sample is a container (dict or tuple).

    Returns:
        A container with the same nested structure, where each leaf contains a batch of data.
    """
    elem = batch[0]
    # Handle values of dictionary with custom_collate_fn to catch TensorDict values.
    if isinstance(elem, dict):
        return {key: _custom_collate_fn([d[key] for d in batch]) for key in elem}
    # Handle tuple elements with custom_collate_fn to catch TensorDict elements.
    if isinstance(elem, tuple):
        return [_custom_collate_fn(samples) for samples in list(zip(*batch))]
    # Handle batching of TensorDict.
    if isinstance(elem, TensorDict):
        return TensorDict(
            {
                key: _custom_collate_fn([d[key] for d in batch]) for key in elem.keys()
            }  # Cannot be handled by default_collate
        )

    # For all other types (lists, tensors, etc.), use PyTorch's default_collate
    return default_collate(batch)


def cache_metrics(output_dir, timestamp, nbatches, metrics):
    """
    Saves the training state to disk.
    :param filepath: Path to save the checkpoint file.
    :param timestamp: The timestamp of the current training iteration.
    :param nbatches: Number of batches already processed.
    :param metrics: A dictionary of metrics to save.
    """
    metrics = copy.deepcopy(metrics)  # Avoid modifying the original metrics dictionary.
    output_dir = Path(output_dir).joinpath("tmp").joinpath("_".join(metrics.keys()))
    output_dir.mkdir(parents=True, exist_ok=True)

    if "era5_rank_histogram_50_members" in metrics:
        # Hack: Can't pickle lambda functions.
        metrics["era5_rank_histogram_50_members"].metrics["surface"].metric.preprocess = None

    # Need to save seed for rank_hist reproducibility.
    torch.save(
        {"metrics": metrics, "np_random_state": np.random.get_state(), "nbatches": nbatches},
        output_dir.joinpath(f"{timestamp}.pt"),
    )
    print(
        f"metrics saved until and including timestamp: {np.datetime64(timestamp, 's')} ({timestamp})"
    )


def load_metrics(output_dir, metric_names):
    """
    Loads the training state from disk.
    :param dir: Directory to load the checkpoint files from.
    :return: A dictionary of metrics loaded from the checkpoint files.
    """
    output_dir = Path(output_dir).joinpath("tmp").joinpath("_".join(metric_names))
    if Path(output_dir).exists():
        files = sorted(Path(output_dir).glob("*.pt"))
        if len(files) == 0:
            print(f"No intermediate metrics found in {output_dir}.")
            return None, None, None
        file = files[-1]
        cached_dict = torch.load(file, weights_only=False)
        nbatches = cached_dict["nbatches"]
        metrics = cached_dict["metrics"]
        np.random.set_state(cached_dict["np_random_state"])

        for metric_name in metric_names:
            if "rank_histogram" in metric_name:
                # Hack: Add back lambda function.
                metrics[metric_name].metrics["surface"].metric.preprocess = lambda x: x.squeeze(-3)

        timestamp = np.datetime64(int(file.stem), "s")
        print(
            f"Loaded intermediate metrics from file: {file}, timestamp: {timestamp}, nbatches: {nbatches}"
        )
        return metrics, timestamp, nbatches

    print(f"No intermediate metrics found in {output_dir}.")
    return None, None, None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save the evaluation metrics. "
        "This script stores separate files per metric requested in --metrics. "
        "Recommended to make one output directory per model being evaluated.",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Directory or file path to find model predictions.",
    )
    parser.add_argument(
        "--pred_filename_filter",
        nargs="*",  # Accepts 0 or more arguments as a list.
        type=str,
        help="Substring(s) in filenames under --pred_path to keep files to run inference on.",
    )
    parser.add_argument(
        "--groundtruth_path",
        type=str,
        required=True,
        help="Directory or file path to read groundtruth.",
    )
    parser.add_argument(
        "--multistep",
        default=10,
        type=int,
        help="Number of future timesteps model is rolled out for evaluation. In days "
        "(This script assumes lead time is 24 hours).",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=1,
        type=int,
        help="Batch size to load preds and targets for eval.",
    )
    parser.add_argument(
        "--cache_metrics_every_nbatches",
        type=int,
        help="Set to cache accumulated metrics to disk every n batches. By default, does not cached metrics"
        "Caches metrics to {output_dir}/tmp/{metric_name}/{timestamp_until_which_metrics_computed}.pt",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Num workers to load data with PyTorch dataloader.",
    )
    parser.add_argument(
        "--level_vars",
        nargs="*",  # Accepts 0 or more arguments as a list.
        default=era5.level_variables,
        help="Level vars to load from preds. Order is respected when read into tensors. Can be empty.",
    )
    parser.add_argument(
        "--surface_vars",
        nargs="*",  # Accepts 0 or more arguments as a list.
        default=era5.surface_variables,
        help="Surface vars to load from preds. Order is respected when read into tensors. Can be empty.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",  # Accepts 1 or more arguments as a list.
        help="Metrics from metrics_registry.py to compute.",
    )
    parser.add_argument(
        "--eval_clim",
        action="store_true",
        help="Whether to evaluate climatology.",
    )

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Output directory to save evaluation.
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Saving evaluation metrics to:", output_dir)

    # Variables to load and evaluate. Assumes same variable naming in groundtruth and predictions.
    variables = {}
    if args.level_vars:
        variables["level"] = args.level_vars
    if args.surface_vars:
        variables["surface"] = args.surface_vars
    if not variables:
        raise ValueError(
            "Need to provide surface and/or level variables to load using --surface_vars and --level_vars."
        )

    # Init metrics.
    metrics, reloaded_timestamp, nbatches = load_metrics(output_dir, args.metrics)
    if metrics is None:
        nbatches = 0
        metrics = {}
        for metric_name in args.metrics:
            metrics[metric_name] = metric_registry.instantiate_metric(
                metric_name,
                surface_variables=args.surface_vars,
                level_variables=args.level_vars,
                pressure_levels=[500, 700, 850],
                lead_time_hours=24 if args.multistep else None,
                rollout_iterations=args.multistep,
            ).to(device)
    print(f"Computing: {metrics.keys()}")

    # Groundtruth.
    dimension_indexers = default_dimension_indexers.copy()
    dimension_indexers["level"] = ("level", [500, 700, 850])  # Use only these pressure levels.
    ds_test = era5.Era5Forecast(
        path=args.groundtruth_path,
        # filename_filter=lambda x: ("2020" in x) and ("0h" in x or "12h" in x),
        domain="test_z0012",
        lead_time_hours=24,
        multistep=args.multistep,
        load_prev=False,
        norm_scheme=None,
        variables=variables,
        dimension_indexers=dimension_indexers,
        load_clim=True if args.eval_clim else False,  # Set if evaluating climatology.
    )

    print(f"Reading {len(ds_test.files)} files from groundtruth path: {args.groundtruth_path}.")

    # Predictions.
    def _pred_filename_filter(filename):
        if "metric" in filename:
            return False
        if args.pred_filename_filter is None:
            return True
        for substring in args.pred_filename_filter:
            if substring not in filename:
                return False
        return True

    if not args.eval_clim:
        dimension_indexers["prediction_timedelta"] = (
            "prediction_timedelta",
            [timedelta(days=i) for i in range(1, args.multistep + 1)],
        )

        # Load predictions.
        ds_pred = era5.Era5Dataset(
            path=args.pred_path,
            filename_filter=_pred_filename_filter,  # Update filename_filter to filter within pred_path.
            variables=variables,
            return_timestamp=True,
            dimension_indexers=dimension_indexers,
        )
        print(f"Reading {len(ds_pred.files)} files from pred_path: {args.pred_path}.")

        if reloaded_timestamp is not None:
            # Don't include the reloaded timestamp.
            low_bound = reloaded_timestamp + np.timedelta64(1, "s")
            # If reloading metrics, filter dataset to timestamps that were not evaluated.
            ds_pred.set_timestamp_bounds(low=low_bound, high=None, debug=True)

        # check if prediction timestamps are in ds
        class SelectTimestampsDataset(torch.utils.data.Dataset):
            def __init__(self, ds, select_timestamps):
                self.ds = ds
                self.select_timestamps = select_timestamps
                self.ds_timestamp_to_idx = {k[-1]: i for i, k in enumerate(ds.timestamps)}

            def __len__(self):
                return len(self.select_timestamps)

            def __getitem__(self, idx):
                new_idx = self.ds_timestamp_to_idx[self.select_timestamps[idx][-1]]
                return self.ds[new_idx]

        ds_test = SelectTimestampsDataset(ds_test, ds_pred.timestamps)

    # init dataloaders:
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=_custom_collate_fn,
    )
    if not args.eval_clim:
        dl_pred = torch.utils.data.DataLoader(
            ds_pred,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=_custom_collate_fn,
        )

    # iterable = tqdm(dl_test) if args.eval_clim else tqdm(zip(dl_test, dl_pred))
    for next_batch in tqdm(dl_test) if args.eval_clim else tqdm(zip(dl_test, dl_pred)):
        nbatches += 1

        if args.eval_clim:
            target = next_batch
            pred = next_batch["clim_state"].apply(lambda x: x.unsqueeze(1))  # Add mem dimension.
        else:
            target, (pred, pred_timestamps) = next_batch
            # Check same timestep loaded from groundtruth and prediction.
            torch.testing.assert_close(
                target["timestamp"],
                pred_timestamps,
            )
            # Switch var dimension.
            pred = pred.apply(
                lambda tensor: rearrange(
                    tensor,
                    "batch var mem ... lev lat lon -> batch mem ... var lev lat lon",
                )
            )
        timestamps = target["timestamp"]
        if args.multistep == 0 or args.eval_clim:  # No timedelta dimension for climatology.
            target = target["state"]
        elif args.multistep == 1:
            target = target["next_state"]
        else:
            target = target["future_states"]

        # Update metrics.
        for metric in metrics.values():
            metric.update(target.to(device), pred.to(device))

        if args.cache_metrics_every_nbatches and nbatches % args.cache_metrics_every_nbatches == 0:
            print(f"Processed {nbatches} batches.")
            cache_metrics(
                output_dir=output_dir,
                timestamp=int(timestamps[-1]),
                nbatches=nbatches,
                metrics=metrics,
            )

    timestamp = int(timestamps[-1])
    print(
        f"Finished computation. Computed until {np.datetime64(timestamp, 's')} ({timestamps[-1]})"
    )

    for metric_name, metric in metrics.items():
        labelled_metric_output = metric.compute()

        if Path(args.pred_path).is_file():
            output_filename = f"{Path(args.pred_path).stem}-{metric_name}"
        else:
            output_filename = f"test-multistep={args.multistep}-{metric_name}"

        # Get xr dataset.
        if isinstance(labelled_metric_output, dict):
            labelled_dict = {
                k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in labelled_metric_output.items()
            }
            extra_dimensions = ["prediction_timedelta"]
            if "brier" in metric_name:
                extra_dimensions = ["quantile", "prediction_timedelta"]
            if "rankhist" in metric_name or "rank_hist" in metric_name:
                extra_dimensions = ["bins", "prediction_timedelta"]
            ds = convert_metric_dict_to_xarray(labelled_dict, extra_dimensions)

            # Write labeled dict.
            labelled_dict["metadata"] = dict(
                groundtruth_path=args.groundtruth_path, predictions_path=args.pred_path
            )
            torch.save(labelled_dict, Path(output_dir).joinpath(f"{output_filename}.pt"))
        else:
            ds = labelled_metric_output
        # Write xr dataset.
        ds.to_netcdf(Path(output_dir).joinpath(f"{output_filename}.nc"))


if __name__ == "__main__":
    main()
