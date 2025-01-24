import torch

from geoarches.metrics.spherical_power_spectrum import Era5PowerSpectrum, PowerSpectrum


class TestPowerSpectrum:
    def test_update_nsamples(self):
        bs, mem, timedelta, var, lev, lat, lon = 2, 3, 5, 2, 3, 120, 240
        targets = torch.zeros(bs, timedelta, var, lev, lat, lon)
        preds = torch.zeros(bs, mem, timedelta, var, lev, lat, lon)
        metric = PowerSpectrum()

        metric.update(targets=targets, preds=preds)
        assert metric.nsamples == 2

        metric.update(targets=targets, preds=preds)
        assert metric.nsamples == 4


class TestEra5PowerSpectrum:
    def test_output_dimensions(self):
        bs, mem, lev, lat, lon = 2, 3, 3, 121, 240
        metric = Era5PowerSpectrum(
            level_variables=["geopotential", "temperature"],
            pressure_levels=[500, 700, 850],
        )
        preds = {
            "surface": torch.randn(bs, mem, 4, 1, lat, lon),
            "level": torch.randn(bs, mem, 2, lev, lat, lon),
        }
        targets = {
            "surface": torch.randn(bs, 4, 1, lat, lon),
            "level": torch.randn(bs, 2, lev, lat, lon),
        }

        metric.update(targets, preds)
        metric.update(targets, preds)

        output_xarray = metric.compute()

        for coord in ["metric", "level"]:
            assert coord in output_xarray.coords

    def test_output_dimensions_with_timdelta_dimension(self):
        bs, mem, timedelta, lev, lat, lon = 2, 3, 5, 3, 121, 240
        metric = Era5PowerSpectrum(
            rollout_iterations=timedelta,
            lead_time_hours=6,
            level_variables=["geopotential", "temperature"],
            pressure_levels=[500, 700, 850],
        )
        preds = {
            "surface": torch.randn(bs, mem, timedelta, 4, 1, lat, lon),
            "level": torch.randn(bs, mem, timedelta, 2, lev, lat, lon),
        }
        targets = {
            "surface": torch.randn(bs, timedelta, 4, 1, lat, lon),
            "level": torch.randn(bs, timedelta, 2, lev, lat, lon),
        }

        metric.update(targets, preds)
        metric.update(targets, preds)

        output_xarray = metric.compute()

        for coord in ["prediction_timedelta", "metric", "level"]:
            assert coord in output_xarray.coords
