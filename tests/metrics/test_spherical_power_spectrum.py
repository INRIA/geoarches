import torch
from geoarches.metrics.spherical_power_spectrum import PowerSpectrum


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
