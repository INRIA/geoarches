"""Given paths to stored metrics, plot metrics.

Plot each metric in a separate .png file.
Plot metric for each variable separately, for all models and all lead times.
"""

import argparse
import math
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from geoarches.metrics.label_wrapper import convert_metric_dict_to_xarray

plot_metric_kwargs = {
    "rmse": dict(y_label="Ensemble mean RMSE"),
    "crps": dict(y_label="CRPS"),
    "fcrps": dict(y_label="Fair CRPS"),
    "spskr": dict(y_label="Spread/skill", horizontal_reference=1),
    "brierskillscore": dict(y_label="Brier skill score", horizontal_reference=0),
    "rankhist": dict(y_label="Frequency", horizontal_reference=1),
}

HIGH_QUANTILES = ["99.0%", "99.9%", "99.99%"]
LOW_QUANTILES = ["1.0%", "0.1%", "0.01%"]


def plot_metric(
    data_dict: dict[str, xr.Dataset],
    vars: list[str],
    metric_name: str,
    y_label: str | None = None,
    horizontal_reference: float | None = None,
    figsize=(10, 4),
    debug: bool = False,
):
    """For all models, plot lead time vs. metric per variable.
    Does not handle brier score or rank histogram (see below methods).

    Args:
        data_dict: Mapping from model name to xarray dataset holding metrics.
            Dataset variables are metric
            Dataset dimensions include `variable` and `prediction_timedelta`.
        vars: list of variables to read from dimension `variable` in xarray dataset.
        metric_name: Name of metric to read from dataset variables.
        y_label: (Optional) to label y axis of the figure.
        horizontal_reference: (Optional) y value for horizontal dashed line on the plots.
        figsize: Figure size.
        debug: Whether to print debug statements.
    """
    # Create a grid of subplots with 2 rows.
    num_cols = math.ceil(len(vars) / 2)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, num_cols * 2, figure=fig)

    if y_label:
        fig.supylabel(y_label)

    for i, var in enumerate(vars):
        if debug:
            print(var)

        row, col = i // num_cols, (i % num_cols) * 2
        if len(vars) % 2 == 1:
            col += row  # offset bottom row.
        ax = fig.add_subplot(gs[row, col : col + 2])
        ax.set_title(var)
        ax.set_xlabel("Lead time (days)")

        for model, ds in data_dict.items():
            if metric_name == "rmse":
                scores = ds["mse"].sel(variable=var)
                scores = np.sqrt(scores)
            else:
                scores = ds[metric_name].sel(variable=var)

            if debug:
                print(model, scores)

            days = ds.prediction_timedelta.dt.days
            ax.plot(days, scores, label=model)
            ax.grid(True)

            if horizontal_reference is not None:
                ax.axhline(y=horizontal_reference, color="gray", linestyle="--", linewidth=1)

    plt.tight_layout()  # ensure titles and other plot elements don't overlap.
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5))


def plot_brier_metric(
    data_dict: dict[str, xr.Dataset],
    vars: list[str],
    quantile_levels: list[str],  # high or low
    metric_name: str,
    y_label: str | None = None,
    horizontal_reference: float | None = None,
    figsize=(10, 5),
    debug: bool = False,
):
    """For all models, plot lead time vs. brier score per variable and quantile.

    Args:
        data_dict: Mapping from model name to xarray dataset holding metrics.
            Dataset variables are metric
            Dataset dimensions include `variable` and `prediction_timedelta`.
        vars: list of variables to read from dimension `variable` in xarray dataset.
        quantile_levels: Whether to plot high quantiles (ie. 99%, 99.9%, 99.99% levels) or low for each variable.
            Same length as `vars` list arg.
        metric_name: Name of metric to read from dataset variables.
        y_label: (Optional) to label y axis of the figure.
        horizontal_reference: (Optional) y value for horizontal dashed line on the plots.
        figsize: Figure size.
        debug: Whether to print debug statements.
    """
    quantiles_per_var = [HIGH_QUANTILES if q == "high" else LOW_QUANTILES for q in quantile_levels]

    # Create a grid of subplots: quantile vs. variable.
    fig, axs = plt.subplots(len(HIGH_QUANTILES), len(vars), figsize=figsize)
    # y labels.
    if y_label:
        fig.supylabel(y_label)
    for q in range(3):
        axs[q, 0].set_ylabel(f"{10 ** -q}% extremes")

    for model, ds in data_dict.items():
        for i, (var, quantiles) in enumerate(zip(vars, quantiles_per_var)):
            if debug:
                print(var, quantiles)

            axs[0, i].set_title(f"{var}\nextreme {quantile_levels[i]}")
            axs[2, i].set_xlabel("Lead time (days)")

            for q, quantile in enumerate(quantiles):
                scores = ds[metric_name].sel(variable=var, quantile=quantile)
                if debug:
                    print(model, scores)

                days = ds.prediction_timedelta.dt.days
                axs[q, i].plot(days, scores, label=model)
                axs[q, i].grid(True)

                if horizontal_reference is not None:
                    axs[q, i].axhline(
                        y=horizontal_reference, color="gray", linestyle="--", linewidth=1
                    )

    plt.tight_layout()  # ensure titles and other plot elements don't overlap.
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5))


def plot_rankhist(
    data_dict: dict[str, xr.Dataset],
    vars: list[str],
    prediction_timedeltas_days: list[int],
    metric_name: str = "rankhist",
    y_label: str | None = None,
    horizontal_reference: float | None = None,
    figsize=(10, 5),
    debug: bool = False,
):
    """For all models, rank bin vs. frequency per variable and lead time.

    Args:
        data_dict: Mapping from model name to xarray dataset holding metrics.
            Dataset variables are metric
            Dataset dimensions include `variable` and `prediction_timedelta`.
        vars: list of variables to read from dimension `variable` in xarray dataset.
        prediction_timedeltas_days: List of lead times (in days) to plot.
        metric_name: Name of metric to read from dataset variables.
        y_label: (Optional) to label y axis of the figure.
        horizontal_reference: (Optional) y value for horizontal dashed line on the plots.
        figsize: Figure size.
        debug: Whether to print debug statements.
    """
    # Create a grid of subplots: vars vs. lead time.
    fig, axs = plt.subplots(len(prediction_timedeltas_days), len(vars), figsize=figsize)
    # y labels.
    if y_label:
        fig.supylabel(y_label)
    for i, days in enumerate(prediction_timedeltas_days):
        axs[i, 0].set_ylabel(f"{days} days")

    for model, ds in data_dict.items():
        for col, var in enumerate(vars):
            axs[0, col].set_title(var)
            for row, days in enumerate(prediction_timedeltas_days):
                scores = ds[metric_name].sel(
                    variable=var, prediction_timedelta=timedelta(days=days)
                )
                axs[row, col].plot(scores, label=model)
                axs[row, col].grid(True)
                if horizontal_reference is not None:
                    axs[row, col].axhline(
                        y=horizontal_reference, color="gray", linestyle="--", linewidth=1
                    )

    plt.tight_layout()  # ensure titles and other plot elements don't overlap.
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder to save .png plots to.",
    )
    parser.add_argument(
        "--metric_paths",
        nargs="+",  # Accepts 1 or more arguments as a list.
        type=str,
        required=True,
        help="Paths holding metrics. Expected output from LabelWrapper class:"
        "labelled dict stored as .pt file or xarray dataset. "
        "Expects `prediction_timedelta` dimension."
        "Can have multiple metric_paths for the same model.",
    )
    parser.add_argument(
        "--model_names_for_legend",
        nargs="+",  # Accepts 1 or more arguments as a list.
        type=str,
        required=True,
        help="Model name corresponding to each metric_path to use in plot legends."
        "Expected same length as --metric_paths. Can have multiple entried for the same model_name.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",  # Accepts 1 or more arguments as a list.
        type=str,
        required=True,
        default=["rmse", "crps", "spskr"],
        help="Names of metrics to plot. These match the keys that exist in the labeled dict as <metric>_<var>_..."
        " or match the name of the `metric` dimension in the xarray dataset.",
    )
    parser.add_argument(
        "--vars",
        nargs="+",  # Accepts 1 or more arguments as a list.
        type=str,
        default=["Z500", "Q700", "T850", "U850", "V850", "T2m", "U10m", "V10m", "SP"],
        # default=["T2m", "T2m", "U10m", "V10m", "SP"],  # For brierskillscore.
        help="Variables to plot. If None, plots all available ariables in metric paths.",
    )
    parser.add_argument(
        "--figsize",
        nargs="+",  # Accepts 1 or more arguments as a list.
        type=int,
        default=[10, 5],
        help="Matplotlib figsize param for size of graph.",
    )
    parser.add_argument(
        "--brier_quantile_levels",
        nargs="+",  # Accepts 1 or more arguments as a list.
        type=str,
        default=["high", "low", "high", "high", "low"],
        help="Used only for plotting `brierskillscore` metric. For each variable, whether to plot high or low quantile levels."
        "Expected same length as --vars. Can have multiple entried for the same var (ie. want to plot high and low quantile levels).",
    )
    parser.add_argument(
        "--rankhist_prediction_timedeltas",
        nargs="+",  # Accepts 1 or more arguments as a list.
        type=int,
        default=[1, 3, 5, 10, 15],
        help="Used only for plotting `rankhist` metric. For each variable, which lead times to plot.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force save plots if file already exists."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Whether to print debug statements to stdout."
    )
    parser.add_argument(
        "--save_xr_dataset",
        action="store_true",
        help="Whether to also save the metrics as xr dataset (if input is .pt file) for further analysis.",
    )

    args = parser.parse_args()

    assert len(args.metric_paths) == len(
        args.model_names_for_legend
    ), "Len of metric_paths != len of model_names_for_legend."

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Saving plots to:", output_dir)

    data = defaultdict(list)
    for model_name, metric_path in zip(args.model_names_for_legend, args.metric_paths):
        file_suffix = Path(metric_path).suffix
        if file_suffix == ".nc":
            ds = xr.open_dataset(metric_path)
        elif Path(metric_path).suffix == ".zarr":
            ds = xr.open_dataset(metric_path, engine="zarr")
        elif Path(metric_path).suffix == ".pt":
            labeled_dict = torch.load(metric_path, weights_only=True)
            extra_dimensions = ["prediction_timedelta"]
            if "brier" in metric_path:
                extra_dimensions = ["quantile", "prediction_timedelta"]
            if "rankhist" in metric_path:
                extra_dimensions = ["bins", "prediction_timedelta"]
            ds = convert_metric_dict_to_xarray(labeled_dict, extra_dimensions)

        data[model_name].append(ds)

    for model, ds_list in data.items():
        merged_ds = xr.merge(ds_list)
        data[model] = merged_ds
        vars = ds.variable.values
        metrics = list(merged_ds.data_vars)

        if args.save_xr_dataset:
            save_file = Path(output_dir).joinpath(f"{model}_metrics.nc")

    vars = args.vars or vars
    metrics = args.metrics or metrics
    for metric_name in metrics:
        if args.debug:
            print(metric_name)
        kwargs = plot_metric_kwargs.get(metric_name, {})
        if "brier" in metric_name:
            quantile_levels = args.brier_quantile_levels
            if len(vars) != len(quantile_levels):
                raise ValueError(
                    f"brier_quantile_levels length {len(quantile_levels)} is not equal to vars length {len(vars)}."
                )
            plot_brier_metric(
                data,
                vars,
                quantile_levels,
                metric_name,
                **kwargs,
                figsize=args.figsize,
                debug=args.debug,
            )
        elif "rankhist" in metric_name:
            plot_rankhist(
                data,
                vars,
                args.rankhist_prediction_timedeltas,
                metric_name,
                **kwargs,
                figsize=args.figsize,
                debug=args.debug,
            )
        else:
            plot_metric(data, vars, metric_name, **kwargs, figsize=args.figsize, debug=args.debug)

        save_file = Path(output_dir).joinpath(f"{metric_name}.png")
        if save_file.exists():
            if not args.force:
                raise ValueError(f"File {save_file} already exists. Did not save plot.")

        plt.savefig(save_file, bbox_inches="tight")


if __name__ == "__main__":
    main()
