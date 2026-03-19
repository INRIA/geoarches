"""Utility functions for handling Nans in data."""

import enum

import tensordict
import torch
import xarray as xr

from geoarches.utils.tensordict_utils import tensordict_apply


class NanInterpolationMethod(str, enum.Enum):
    """Methods for interpolating Nans in data."""

    NONE = "none"
    # Fill NaNs with global mean per variable.
    GLOBAL_MEAN = "global_mean"
    # Fill NaNs with 0.0.
    ZERO = "zero"
    # Fill NaNs in SIC with 0.0 and fill the rest with the latitude mean.
    # Forward fill the data if all values in a latitude are NaN.
    LAT_MEAN_SIC_ZERO = "lat_mean_sic_zero"

    # Fill NaNs with 0.0 after normalization.
    ZERO_AFTER_NORM = "zero_after_norm"


# Methods that should be applied after normalization.
POST_NORM_METHODS = [NanInterpolationMethod.ZERO_AFTER_NORM, NanInterpolationMethod.GLOBAL_MEAN]


def pre_norm_interpolate_nans(
    ds: xr.Dataset,
    nan_interpolation_method: NanInterpolationMethod | None,
) -> xr.Dataset:
    """Interpolates Nans in an xarray dataset."""
    if not nan_interpolation_method or nan_interpolation_method == NanInterpolationMethod.NONE:
        return ds
    if nan_interpolation_method in POST_NORM_METHODS:
        return ds
    elif nan_interpolation_method == NanInterpolationMethod.ZERO:
        ds = ds.fillna(value=0.0)
    # Fill NaNs in SIC with 0.0 and fill the rest with the latitude mean.
    # Forward fill (ffill) the data if all values in a latitude are NaN.
    # Ffill fills the data with the closest valid value from previous latitudes.
    elif nan_interpolation_method == NanInterpolationMethod.LAT_MEAN_SIC_ZERO:
        if "sea_ice_cover" in ds.data_vars:
            ds["sea_ice_cover"] = ds["sea_ice_cover"].fillna(value=0.0)
        ds = ds.fillna(value=ds.mean(dim=["longitude"], skipna=True)).ffill("latitude", limit=None)
    else:
        raise ValueError(f"Unknown nan interpolation method: {nan_interpolation_method}")

    return ds


def post_norm_interpolate_nans(
    t: torch.Tensor | tensordict.TensorDict,
    nan_interpolation_method: NanInterpolationMethod | None,
):
    """Optionally interpolates Nans in a torch tensor. Meant for after normalization."""
    if not nan_interpolation_method or nan_interpolation_method == NanInterpolationMethod.NONE:
        return t
    if nan_interpolation_method not in POST_NORM_METHODS:
        return t
    if nan_interpolation_method == NanInterpolationMethod.GLOBAL_MEAN:
        # Average over the spatial dims (lat and lon).
        if isinstance(t, torch.Tensor):
            fill_value = t.nanmean(dim=(-2, -1), keepdim=True)
        else:
            fill_value = tensordict_apply(torch.nanmean, t, dim=(-2, -1), keepdim=True)
    elif nan_interpolation_method == NanInterpolationMethod.ZERO_AFTER_NORM:
        fill_value = 0.0
    else:
        raise ValueError(f"Unknown nan interpolation method: {nan_interpolation_method}")

    if isinstance(t, torch.Tensor):
        return torch.where(torch.isnan(t), fill_value, t)
    else:
        return tensordict_apply(torch.where, t.isnan(), fill_value, t)
