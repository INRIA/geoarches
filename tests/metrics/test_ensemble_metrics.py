import pytest
import torch
from geoarches.metrics.ensemble_metrics import EnsembleMetrics, Era5EnsembleMetrics

LAT = 3
LON = 5
DATA_SHAPE = (2, 1)


@pytest.fixture
def unweighted_metric():
    def _compute_uniform_weights(latitude_resolution):
        return torch.ones(latitude_resolution)[:, None]

    return EnsembleMetrics(compute_lat_weights_fn=_compute_uniform_weights, data_shape=DATA_SHAPE)


@pytest.fixture
def batch():
    return torch.zeros(1, *DATA_SHAPE, LAT, LON)


@pytest.fixture
def preds1():
    """Fake predictions of 2 variables for 3 members."""
    var1_preds = torch.arange(0, 3)  # nonzero variance across members.
    var1_preds = var1_preds.reshape(1, 3, 1, 1, 1, 1).repeat(1, 1, 1, 1, LAT, LON)
    var2_preds = torch.ones(1, 3, 1, 1, LAT, LON) * 2  # zero variance.
    return torch.concat([var1_preds, var2_preds], dim=2)


@pytest.fixture
def preds2():
    """Similar to preds1 except that var1 has 4x variance."""
    var1_preds = torch.arange(0, 3) * 2  # 4x variance of preds1.
    var1_preds = var1_preds.reshape(1, 3, 1, 1, 1, 1).repeat(1, 1, 1, 1, LAT, LON)
    var2_preds = torch.ones(1, 3, 1, 1, LAT, LON) * 2  # zero variance.
    return torch.concat([var1_preds, var2_preds], dim=2)


class TestEnsembleMetrics:
    def test_update_nsamples(self, unweighted_metric, batch):
        preds = torch.zeros(1, 3, *DATA_SHAPE, LAT, LON)  # (batch, nmembers, ...)

        unweighted_metric.update(batch, preds)
        assert unweighted_metric.nsamples == 1

        unweighted_metric.update(batch, preds)
        assert unweighted_metric.nsamples == 2

    def test_update_nmembers(self, unweighted_metric, batch):
        preds = torch.zeros(1, 3, *DATA_SHAPE, LAT, LON)  # (batch, nmembers, ...)

        unweighted_metric.update(batch, preds)
        assert unweighted_metric.nmembers == 3

        unweighted_metric.update(batch, preds)
        assert unweighted_metric.nmembers == 6

    def test_update_and_compute_mse(self, unweighted_metric, batch, preds1, preds2):
        unweighted_metric.update(batch, preds1)
        torch.testing.assert_close(unweighted_metric.mse, torch.tensor([[1.0], [4.0]]))

        unweighted_metric.update(batch, preds2)
        torch.testing.assert_close(unweighted_metric.mse, torch.tensor([[5.0], [8.0]]))

        output = unweighted_metric.compute()
        torch.testing.assert_close(output["mse"], torch.tensor([[2.5], [4.0]]))  # Divide by 2.

    def test_update_and_compute_var(self, unweighted_metric, batch, preds1, preds2):
        unweighted_metric.update(batch, preds1)
        torch.testing.assert_close(unweighted_metric.var, torch.tensor([[1.0], [0]]))

        unweighted_metric.update(batch, preds2)
        torch.testing.assert_close(unweighted_metric.var, torch.tensor([[5.0], [0]]))

        output = unweighted_metric.compute()
        torch.testing.assert_close(output["var"], torch.tensor([[2.5], [0.0]]))

    def test_compute_spskr(self, unweighted_metric, batch, preds1, preds2):
        unweighted_metric.update(batch, preds1)
        unweighted_metric.update(batch, preds2)

        output = unweighted_metric.compute()
        torch.testing.assert_close(
            output["spskr"], torch.tensor([[torch.tensor(4.0 / 3).sqrt()], [0.0]])
        )

    def test_update_mae(self, unweighted_metric, batch, preds1, preds2):
        unweighted_metric.update(batch, preds1)
        torch.testing.assert_close(unweighted_metric.mae, torch.tensor([[1.0], [2.0]]))

        unweighted_metric.update(batch, preds2)
        torch.testing.assert_close(unweighted_metric.mae, torch.tensor([[3.0], [4.0]]))

    def test_update_dispersion(self, unweighted_metric, batch, preds1, preds2):
        unweighted_metric.update(batch, preds1)
        # Sum of abs differences between ensemble predictions = (1 + 1 + 2) * 2 = 8.
        torch.testing.assert_close(unweighted_metric.dispersion, torch.tensor([[8.0 / 9], [0.0]]))

        unweighted_metric.update(batch, preds2)
        # Sum of abs differences between ensemble predictions = (2 + 2 + 4) * 2 = 16.
        torch.testing.assert_close(unweighted_metric.dispersion, torch.tensor([[24.0 / 9], [0.0]]))

    def test_update_dispersion_save_memory(self, unweighted_metric, batch, preds1, preds2):
        unweighted_metric.save_memory = True
        unweighted_metric.update(batch, preds1)
        torch.testing.assert_close(unweighted_metric.dispersion, torch.tensor([[8.0 / 9], [0.0]]))

        unweighted_metric.update(batch, preds2)
        torch.testing.assert_close(unweighted_metric.dispersion, torch.tensor([[24.0 / 9], [0.0]]))

    def test_update_energy_dispersion(self, unweighted_metric, batch, preds1, preds2):
        unweighted_metric.update(batch, preds1)
        torch.testing.assert_close(
            unweighted_metric.energy_dispersion, torch.tensor([[8.0 / 9], [0.0]])
        )

        unweighted_metric.update(batch, preds2)
        torch.testing.assert_close(
            unweighted_metric.energy_dispersion, torch.tensor([[24.0 / 9], [0.0]])
        )

    def test_update_energy_dispersion_save_memory(self, unweighted_metric, batch, preds1, preds2):
        unweighted_metric.save_memory = True
        unweighted_metric.update(batch, preds1)
        torch.testing.assert_close(
            unweighted_metric.energy_dispersion, torch.tensor([[8.0 / 9], [0.0]])
        )

        unweighted_metric.update(batch, preds2)
        torch.testing.assert_close(
            unweighted_metric.energy_dispersion, torch.tensor([[24.0 / 9], [0.0]])
        )

    def test_compute_crps(self, unweighted_metric, batch, preds1, preds2):
        unweighted_metric.update(batch, preds1)
        unweighted_metric.update(batch, preds2)

        output = unweighted_metric.compute()

        torch.testing.assert_close(
            output["crps"], torch.tensor([[torch.tensor(1.5 - 2.0 / 3)], [2.0]])
        )

    def test_handles_list_of_member_predictions(self, unweighted_metric, batch):
        preds = []
        for _ in range(3):
            preds.append(torch.zeros(1, *DATA_SHAPE, LAT, LON))  # (batch, nmembers, ...)

        unweighted_metric.update(batch, preds)
        assert unweighted_metric.nmembers == 3

        unweighted_metric.update(batch, preds)
        assert unweighted_metric.nmembers == 6

    def test_handles_extra_time_dimension_by_default(self, unweighted_metric, batch):
        preds = torch.arange(0.0, 4)  # Preds for 4 different forecast lead times.
        preds = preds.reshape(1, 1, 4, 1, 1, 1, 1).repeat(
            (1, 3, 1, *DATA_SHAPE, LAT, LON)
        )  # (batch, nmembers, lead time, ...)
        batch = torch.zeros(1, 4, *DATA_SHAPE, LAT, LON)  # (batch, time, ...)

        unweighted_metric.update(batch, preds)
        unweighted_metric.update(batch, preds)
        output = unweighted_metric.compute()

        torch.testing.assert_close(output["mse"][:, 0, 0], torch.tensor([0.0, 1.0, 4.0, 9.0]))


@pytest.fixture
def weighted_metric():
    def _compute_linear_weights(latitude_resolution):
        return torch.arange(0.0, latitude_resolution)[:, None]

    return EnsembleMetrics(compute_lat_weights_fn=_compute_linear_weights, data_shape=DATA_SHAPE)


class TestWeightedEnsembleMetrics:
    def test_update_and_compute_mse(self, weighted_metric, batch, preds1):
        weighted_metric.update(batch, preds1)

        torch.testing.assert_close(weighted_metric.mse, torch.tensor([[1.0], [4.0]]))

    def test_update_and_compute_var(self, weighted_metric, batch, preds1):
        weighted_metric.update(batch, preds1)

        torch.testing.assert_close(weighted_metric.var, torch.tensor([[1.0], [0.0]]))

    def test_update_and_compute_mae(self, weighted_metric, batch, preds1):
        weighted_metric.update(batch, preds1)

        torch.testing.assert_close(weighted_metric.mae, torch.tensor([[1.0], [2.0]]))


class TestEra5EnsembleMetrics:
    def test_output_keys(self):
        level_variables = ["geopotential", "specific_humidity"]
        pressure_levels = [500, 700]
        metric = Era5EnsembleMetrics(
            level_variables=level_variables, pressure_levels=pressure_levels
        )
        lev_vars, lev = len(level_variables), len(pressure_levels)
        timedelta, lat, lon = 10, 121, 240
        preds = {
            "surface": torch.randn(1, 3, timedelta, 4, 1, lat, lon),
            "level": torch.randn(1, 3, timedelta, lev_vars, lev, lat, lon),
        }
        targets = {
            "surface": torch.randn(1, timedelta, 4, 1, lat, lon),
            "level": torch.randn(1, timedelta, lev_vars, lev, lat, lon),
        }

        metric.update(targets, preds)
        metric.update(targets, preds)

        output = metric.compute()

        expected_metric_keys = [  # All expected keys for one metric.
            "mse_U10m",
            "mse_V10m",
            "mse_T2m",
            "mse_SP",
            "mse_Z500",
            "mse_Q700",
        ]
        for expected_metric_key in expected_metric_keys:
            assert expected_metric_key in output
            assert output[expected_metric_key].numel() == timedelta
