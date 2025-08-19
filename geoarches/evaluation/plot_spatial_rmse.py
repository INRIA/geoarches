import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_spatial_rmse(
    base_dirs: list[str] | list[Path],
    metric_filename: str | Path,
    save_path: str | Path,
    metric_key: str = "rmse_per_gridpoint_V850",
    titles: list[str] | None = None,
    force: bool = False,
    cbar_label: str | None = None,
):
    """
    Plots a 2D spatial RMSE array as a world map grid using Cartopy.

    Args:
        spatial_data (torch.Tensor or np.ndarray): A 2D array of spatial RMSE values
                                                   with shape (lat, lon).
    """
    if Path(save_path).exists() and not force:
        print(f"Plot file {save_path} already exists. Use --force to overwrite.")
        return

    spatial_datas = []
    for base_dir in base_dirs:
        spatial_data = torch.load(
            Path(base_dir) / metric_filename,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        spatial_datas.append(spatial_data[metric_key])

    # Determine global min and max for a consistent color scale across all plots
    all_data = np.concatenate([data.flatten() for data in spatial_datas])
    vmin, vmax = np.min(all_data), np.max(all_data)

    # Set font family and use LaTeX for consistent plotting style
    plt.rc("font", family="serif")

    # Set up the plot with a PlateCarree projection, suitable for global data.
    fig = plt.figure(figsize=(4, 2))
    num_plots = len(spatial_datas)
    fig, axes = plt.subplots(
        1, num_plots, figsize=(6 * num_plots, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    # Ensure axes is an array even for a single subplot
    if num_plots == 1:
        axes = [axes]

    axes[0].set_ylabel("Latitude")

    for i, data in enumerate(spatial_datas):
        ax = axes[i]

        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.add_feature(cfeature.OCEAN)
        ax.set_global()

        # Convert to numpy if needed
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # Convert to numpy if needed
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # num_lat, num_lon = data.shape
        # lons = np.arange(0, 360, 360 / num_lon)
        # lats = np.linspace(90, -90, num_lat)

        # Use imshow to plot the data on top of the map with a normalized color scale
        im = ax.imshow(
            data,
            cmap="plasma",
            origin="upper",
            extent=[-180, 180, -90, 90],
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
        )

        # Set plot title and labels
        if titles:
            ax.set_title(titles[i])

        ax.set_xlabel("Longitude")

        # Set ticks and gridlines for clarity
        ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
        ax.grid(True, linestyle="-", color="gray")

    # Add a single color bar for the entire figure
    # Adjust subplots and color bar for better proportions
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])

    cbar_ax = fig.add_axes([0.91, axes[0].get_position().y0, 0.02, axes[0].get_position().height])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    if cbar_label:
        cbar.set_label(cbar_label)

    # Add an overall title to the figure
    fig.suptitle(metric_key.split("_")[-1].upper(), y=0.9, fontsize=16)

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RMSE per year.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/scratch/resingh/weather/evaluation/era5_pred_archesweather-S/",
        help="Base directory containing yearly data subdirectories.",
    )
    parser.add_argument(
        "--metric_filename",
        type=str,
        default="test-multistep=1-era5_deterministic_metrics_with_spatial.pt",
        help="Filename of the metric data to load.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="plots",
        help="Path to save the plot. If None, the plot will be shown but not saved.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force saving plot even if the plot file already exists.",
    )
    args = parser.parse_args()

    for var, cbar_label in zip(
        ["V850", "U850", "Z500"], ["RMSE $[m/s]$", "RMSE $[m/s]$", "RMSE $[m^2/s^2]$"]
    ):
        plot_spatial_rmse(
            base_dirs=[Path(args.base_dir) / "1979_1999", Path(args.base_dir) / "2000_2018"],
            metric_filename=args.metric_filename,
            metric_key=f"rmse_per_gridpoint_{var}",
            save_path=Path(args.save_dir) / f"spatial_rmse_{var}.png",
            force=args.force,
            titles=["1979-1999", "2000-2018"],
            cbar_label=cbar_label,
        )
