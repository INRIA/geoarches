import tensordict
import torch
import xarray as xr

from geoarches.dataloaders import nan_util


class TestPreNormInterpolateNans:
    """Tests for the pre_norm_interpolate_nans function."""

    @pytest.mark.parametrize(
        "method",
        [
            nan_util.NanInterpolationMethod.ZERO,
            nan_util.NanInterpolationMethod.LAT_MEAN_SIC_ZERO,
        ],
    )
    def test_no_nans(self, method):
        """Expects no-op when there are no NaNs in data."""
        ds = xr.Dataset(
            {
                "a": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                ),
                "b": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                ),
            },
            coords={"latitude": [0, 1, 2], "longitude": [0, 1]},
        )
        result_ds = nan_util.pre_norm_interpolate_nans(ds, method)
        xr.testing.assert_allclose(result_ds, ds)

    def test_none(self):
        """Expects no-op when method is None."""
        ds = xr.Dataset(
            {
                "a": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]],
                ),
                "b": (
                    ("latitude", "longitude"),
                    [[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]],
                ),
            },
            coords={"latitude": [0, 1, 2], "longitude": [0, 1]},
        )
        result_ds = nan_util.pre_norm_interpolate_nans(ds, nan_interpolation_method=None)
        xr.testing.assert_allclose(result_ds, ds)

    def test_zero(self):
        """Expects that function replaces NaNs with zeros."""
        ds = xr.Dataset(
            {
                "a": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]],
                ),
                "b": (
                    ("latitude", "longitude"),
                    [[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]],
                ),
            },
            coords={"latitude": [0, 1, 2], "longitude": [0, 1]},
        )
        expected_ds = xr.Dataset(
            {
                "a": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [0.0, 4.0], [5.0, 6.0]],
                ),
                "b": (
                    ("latitude", "longitude"),
                    [[1.0, 0.0], [3.0, 4.0], [5.0, 6.0]],
                ),
            },
            coords={"latitude": [0, 1, 2], "longitude": [0, 1]},
        )
        result_ds = nan_util.pre_norm_interpolate_nans(ds, nan_util.NanInterpolationMethod.ZERO)
        xr.testing.assert_allclose(result_ds, expected_ds)

    def test_lat_mean_sic_zero_when_no_ffill(self):
        """Tests when no ffill is needed (non-nan values in each latitude)."""
        ds = xr.Dataset(
            {
                "sea_ice_cover": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [np.nan, 4.0], [5.0, np.nan]],
                ),
                "sea_surface_temperature": (
                    ("latitude", "longitude"),
                    [[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]],
                ),
            },
            coords={"latitude": [0, 1, 2], "longitude": [0, 1]},
        )
        expected_ds = xr.Dataset(
            {
                "sea_ice_cover": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [0.0, 4.0], [5.0, 0.0]],  # Fill SIC with 0.0.
                ),
                "sea_surface_temperature": (
                    ("latitude", "longitude"),
                    [
                        [1.0, 1.0],
                        [3.0, 4.0],
                        [6.0, 6.0],
                    ],  # Fill rest with lat mean.
                ),
            },
            coords={"latitude": [0, 1, 2], "longitude": [0, 1]},
        )
        result_ds = nan_util.pre_norm_interpolate_nans(
            ds, nan_util.NanInterpolationMethod.LAT_MEAN_SIC_ZERO
        )
        xr.testing.assert_allclose(result_ds, expected_ds)

    def test_lat_mean_sic_zero_when_just_ffill(self):
        """Tests when ffill is needed (all values in a latitude are NaN)."""
        ds = xr.Dataset(
            {
                "sea_ice_cover": (
                    ("latitude", "longitude"),
                    [
                        [1.0, 2.0],
                        [np.nan, 4.0],
                        [np.nan, np.nan],
                    ],  # Fill SIC with 0.0.
                ),
                "sea_surface_temperature": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [np.nan, np.nan], [5.0, 6.0]],
                ),
            },
            coords={"latitude": [0, 1, 2], "longitude": [0, 1]},
        )
        expected_ds = xr.Dataset(
            {
                "sea_ice_cover": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [0.0, 4.0], [0.0, 0.0]],
                ),
                "sea_surface_temperature": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [1.0, 2.0], [5.0, 6.0]],
                ),
            },
            coords={"latitude": [0, 1, 2], "longitude": [0, 1]},
        )
        result_ds = nan_util.pre_norm_interpolate_nans(
            ds, nan_util.NanInterpolationMethod.LAT_MEAN_SIC_ZERO
        )
        xr.testing.assert_allclose(result_ds, expected_ds)

    def test_lat_mean_sic_zero_when_fill_and_ffill(self):
        ds = xr.Dataset(
            {
                "sea_ice_cover": (
                    ("latitude", "longitude"),
                    [
                        [1.0, 2.0],
                        [np.nan, 4.0],
                        [np.nan, np.nan],
                    ],  # Fill SIC with 0.0.
                ),
                "sea_surface_temperature": (
                    ("latitude", "longitude"),
                    [[1.0, np.nan], [np.nan, np.nan], [5.0, 6.0]],
                ),
            },
            coords={"latitude": [0, 1, 2], "longitude": [0, 1]},
        )
        expected_ds = xr.Dataset(
            {
                "sea_ice_cover": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [0.0, 4.0], [0.0, 0.0]],
                ),
                "sea_surface_temperature": (
                    ("latitude", "longitude"),
                    [[1.0, 1.0], [1.0, 1.0], [5.0, 6.0]],
                ),
            },
            coords={"latitude": [0, 1, 2], "longitude": [0, 1]},
        )
        result_ds = nan_util.pre_norm_interpolate_nans(
            ds, nan_util.NanInterpolationMethod.LAT_MEAN_SIC_ZERO
        )
        xr.testing.assert_allclose(result_ds, expected_ds)

    def test_lat_mean_sic_zero_with_flipped_latitude(self):
        """Tests that latitude ordering is preserved."""
        ds = xr.Dataset(
            {
                "sea_ice_cover": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [np.nan, 4.0], [np.nan, np.nan]],
                ),
                "sea_surface_temperature": (
                    ("latitude", "longitude"),
                    [[1.0, np.nan], [np.nan, np.nan], [5.0, 6.0]],
                ),
            },
            coords={"latitude": [2, 1, 0], "longitude": [0, 1]},
        )
        expected_ds = xr.Dataset(
            {
                "sea_ice_cover": (
                    ("latitude", "longitude"),
                    [[1.0, 2.0], [0.0, 4.0], [0.0, 0.0]],
                ),
                "sea_surface_temperature": (
                    ("latitude", "longitude"),
                    [[1.0, 1.0], [1.0, 1.0], [5.0, 6.0]],
                ),
            },
            coords={"latitude": [2, 1, 0], "longitude": [0, 1]},
        )
        result_ds = nan_util.pre_norm_interpolate_nans(
            ds, nan_util.NanInterpolationMethod.LAT_MEAN_SIC_ZERO
        )
        xr.testing.assert_allclose(result_ds, expected_ds)


class TestPostNormInterpolateNans:
    """Tests for the interpolate_nans_after_norm function."""

    def test_global_mean_with_tensor(self):
        """Expects that function replaces NaNs with global mean per variable."""
        x = torch.tensor(
            [
                [[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]],  # first var
                [[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]],  # second var
            ]
        )
        expected_out = torch.tensor(
            [  # computes global mean per variable.
                [[1.0, 2.0], [3.6, 4.0], [5.0, 6.0]],
                [[1.0, 3.8], [3.0, 4.0], [5.0, 6.0]],
            ]
        )
        out = nan_util.post_norm_interpolate_nans(x, nan_util.NanInterpolationMethod.GLOBAL_MEAN)
        torch.testing.assert_close(out, expected_out)

    def test_zero_after_norm_with_tensor(self):
        x = torch.tensor(
            [
                [1.0, 2.0],
                [np.nan, 4.0],
                [5.0, 6.0],
            ],
        )
        out = nan_util.post_norm_interpolate_nans(
            x, nan_util.NanInterpolationMethod.ZERO_AFTER_NORM
        )
        torch.testing.assert_close(
            out,
            torch.tensor(
                [
                    [1.0, 2.0],
                    [0.0, 4.0],
                    [5.0, 6.0],
                ]
            ),
        )

    def test_tensor_zero_after_norm_with_tensordict(self):
        x = tensordict.TensorDict(
            {
                "a": torch.tensor(
                    [
                        [1.0, 2.0],
                        [np.nan, 4.0],
                        [5.0, 6.0],
                    ],
                ),
                "b": torch.tensor(
                    [
                        [1.0, np.nan],
                        [3.0, 4.0],
                        [5.0, 6.0],
                    ],
                ),
            }
        )
        expected_out = tensordict.TensorDict(
            {
                "a": torch.tensor(
                    [
                        [1.0, 2.0],
                        [0.0, 4.0],
                        [5.0, 6.0],
                    ]
                ),
                "b": torch.tensor(
                    [
                        [1.0, 0.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                    ]
                ),
            }
        )
        out = nan_util.post_norm_interpolate_nans(
            x, nan_util.NanInterpolationMethod.ZERO_AFTER_NORM
        )
        assert all(
            tensordict_apply(torch.allclose, out, expected_out).values(),
        )
