import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_rmse_by_year(
    base_dir,
    metric_filename,
    save_path,
    year_range=(1979, 2018),
    metric_keys_left=["rmse_Z500", "rmse_Z850"],
    metric_keys_right=["rmse_U850", "rmse_V850"],
    ylabel_left=r"RMSE $[m^2/s^2]$",
    ylabel_right=r"RMSE $[m/s]$",
    ref_year=2020,
    force=False,
):
    """
    Plots the RMSE for a given range of years with a reference year's RMSE as a dashed line.

    Args:
        base_dir (str): The base directory containing the yearly data subdirectories.
        save_path (str or Path): The path to save the plot. If None, the plot will be shown but not saved.
        year_range (tuple): A tuple (start_year, end_year) for the plot's x-axis.
        metric_keys_left (list): A list of metric keys (e.g., ['rmse_Z500', 'rmse_Z850']) for the left subplot.
        metric_keys_right (list): A list of metric keys (e.g., ['rmse_U850', 'rmse_V850']) for the right subplot.
        ref_year (int): The year to use for the dashed reference line.
        force (bool): If True, forces saving the plot even if the file already exists.
    """
    if Path(save_path).exists() and not force:
        print(f"Plot file {save_path} already exists. Use --force to overwrite.")
        return

    # Generate the list of years to plot
    years = list(range(year_range[0], year_range[1] + 1))

    # Dictionaries to store the loaded data
    data_left = {metric: [] for metric in metric_keys_left}
    data_right = {metric: [] for metric in metric_keys_right}

    # Load data for the specified year range
    for year in years:
        file_path = os.path.join(base_dir, str(year), metric_filename)
        if not os.path.exists(file_path):
            print(f"Warning: File not found for year {year} at {file_path}. Skipping.")
            continue

        year_data = torch.load(file_path, map_location=torch.device("cpu"), weights_only=False)

        # Store the data for the left subplot
        for metric in metric_keys_left:
            data_left[metric].append(year_data[metric].item())

        # Store the data for the right subplot
        for metric in metric_keys_right:
            data_right[metric].append(year_data[metric].item())

    # Load the reference year data separately
    ref_file_path = os.path.join(base_dir, str(ref_year), metric_filename)
    if not os.path.exists(ref_file_path):
        print(f"Error: Reference year data not found at {ref_file_path}.")
        return

    ref_data = torch.load(ref_file_path, map_location=torch.device("cpu"), weights_only=False)

    # Set font family and use LaTeX for consistent plotting style
    plt.rc("font", family="serif")
    # plt.rc("text", usetex=True)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.5))

    # Define the desired x-axis ticks
    desired_ticks = [1980, 1990, 2000, 2010]
    ax1.set_xticks(desired_ticks)
    ax2.set_xticks(desired_ticks)

    # --- Left Subplot (Geopotential) ---
    # ax1.set_title("Geopotential RMSE")
    ax1.set_xlabel("Year")
    ax1.set_ylabel(ylabel_left)
    ax1.grid(True, linestyle="-", color="lightgray")

    # Plot the RMSE lines
    colors = plt.cm.tab10.colors
    for i, metric in enumerate(metric_keys_left):
        if data_left[metric]:
            label = metric.split("_")[-1]  # e.g., 'rmse_Z500' -> 'Z500'
            ax1.plot(years, data_left[metric], label=label, color=colors[i])

            # Plot the dashed reference line
            ref_value = ref_data[metric].item()
            ax1.axhline(y=ref_value, color=colors[i], linestyle=":", linewidth=1.5)

    ax1.legend()

    # --- Right Subplot (Wind Speed) ---
    # ax2.set_title("Wind Speed RMSE")
    ax2.set_xlabel("Year")
    ax2.set_ylabel(ylabel_right)
    ax2.grid(True, linestyle="-", color="lightgray")

    # Plot the RMSE lines
    for i, metric in enumerate(metric_keys_right):
        if data_right[metric]:
            label = metric.split("_")[-1]  # e.g., 'rmse_Z500' -> 'Z500'
            ax2.plot(years, data_right[metric], label=label, color=colors[i])

            # Plot the dashed reference line
            ref_value = ref_data[metric].item()
            ax2.axhline(y=ref_value, color=colors[i], linestyle=":", linewidth=1.5)

    ax2.legend()

    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


# Example usage:
# This part assumes a directory structure and some dummy data for demonstration.
# You would replace this with your actual data path.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RMSE per year.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/scratch/resingh/weather/evaluation/era5_pred_archesweather-S/",
        help="Base directory containing yearly data subdirectories (named `${base_dir}/{year}`)",
    )
    parser.add_argument(
        "--metric_filename",
        type=str,
        default="test-multistep=1-era5_deterministic_metrics_with_spatial_and_hemisphere.pt",
        help="Filename of the metric data to load.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="plots/",
        help="Path to save the plot.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force saving plot even if the plot file already exists.",
    )
    args = parser.parse_args()

    # Replace 'dummy_data' with your actual path, e.g., ''
    plot_rmse_by_year(
        base_dir=args.base_dir,
        metric_filename=args.metric_filename,
        save_path=Path(args.save_dir) / "rmse_per_year.png",
        year_range=(1979, 2018),
        metric_keys_left=["rmse_Z500", "rmse_Z850"],
        metric_keys_right=["rmse_U850", "rmse_V850"],
        ref_year=2020,
        force=args.force,
    )
    plot_rmse_by_year(
        base_dir=args.base_dir,
        metric_filename=args.metric_filename,
        save_path=Path(args.save_dir) / "north_rmse_per_year.png",
        year_range=(1979, 2018),
        metric_keys_left=["rmse-north_Z500", "rmse-north_Z850"],
        metric_keys_right=["rmse-north_U850", "rmse-north_V850"],
        ref_year=2020,
        force=args.force,
    )
    plot_rmse_by_year(
        base_dir=args.base_dir,
        metric_filename=args.metric_filename,
        save_path=Path(args.save_dir) / "south_rmse_per_year.png",
        year_range=(1979, 2018),
        metric_keys_left=["rmse-south_Z500", "rmse-south_Z850"],
        metric_keys_right=["rmse-south_U850", "rmse-south_V850"],
        ref_year=2020,
        force=args.force,
    )
