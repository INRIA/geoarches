import functools

import pytest
import torch
from einops import rearrange

from geoarches.metrics.brier_skill_score import BrierSkillScore, Era5BrierSkillScore

LAT = 3
LON = 5
DATA_SHAPE = (2, 1)


@pytest.fixture
def batch():
    return torch.zeros(1, *DATA_SHAPE, LAT, LON)


@pytest.fixture
def weighted_brier_score():
    return BrierSkillScore(data_shape=DATA_SHAPE)


class TestBrierSkillScore:
    def test_update_nsamples(self, weighted_brier_score, batch):
        preds = torch.zeros(1, 3, *DATA_SHAPE, LAT, LON)  # (batch, nmembers, ...)

        weighted_brier_score.update(batch, preds)
        assert weighted_brier_score.nsamples == 1

        weighted_brier_score.update(batch, preds)
        assert weighted_brier_score.nsamples == 2

    def test_one_gridpoint_one_example(self):
        metric = BrierSkillScore(data_shape=(1, 1))  # 1 var and lev.
        batch = torch.ones(1, 1, 1, 1, 1)  # bs, var, lev, lat, lon.
        pred = torch.tensor([0.4, 0.9, 0.8])  # 3 ensemble members with average of 0.7.
        pred = pred.reshape(1, 3, 1, 1, 1, 1)  # bs, nmembers, var, lev, lat, lon.

        metric.update(batch, pred)

        output = metric.compute()
        torch.testing.assert_close(output["brierscore"], torch.tensor(0.09))  # (0.7-1)^2
        torch.testing.assert_close(output["brierclimscore"], torch.tensor(0.0))  # (1-1)^2

    def test_one_gridpoint_batch(self):
        metric = BrierSkillScore(data_shape=(1, 1))  # 1 var and lev.
        batch = torch.tensor([1, 0]).reshape(2, 1, 1, 1, 1)  # bs=2, var, lev, lat, lon.
        pred = torch.tensor([0.7, 0]).reshape(
            2, 1, 1, 1, 1, 1
        )  # bs=2, nmembers, var, lev, lat, lon.

        metric.update(batch, pred)

        output = metric.compute()
        torch.testing.assert_close(
            output["brierscore"], torch.tensor(0.045)
        )  # avg of (0.7-1)^2 and (0-0)^2.
        torch.testing.assert_close(
            output["brierclimscore"], torch.tensor(0.25)
        )  # avg of (0.5-1)^2 and (0.5-0)^2.
        torch.testing.assert_close(
            output["brierskillscore"], torch.tensor(0.82)
        )  # 1 - 0.045 / 0.25.

    def test_perfect_score(self, weighted_brier_score):
        """Brier skill score is 1 if we predict the targets."""
        for _ in range(5):
            target = torch.randint(low=0, high=2, size=(4, *DATA_SHAPE, LAT, LON))
            pred = target.unsqueeze(1)  # pred matches target
            weighted_brier_score.update(target, pred)
            torch.testing.assert_close(
                weighted_brier_score.brierscore, torch.tensor([[0.0], [0.0]])
            )

        output = weighted_brier_score.compute()
        torch.testing.assert_close(output["brierskillscore"], torch.tensor([[1.0], [1.0]]))

    def test_zero_score(self, weighted_brier_score):
        """Brier skill score is 0 if we predict the climatological probability."""
        # Create random targets.
        batches = []
        batch_size = 4
        for _ in range(5):  # 5 batches total.
            batch = torch.randint(low=0, high=2, size=(batch_size, *DATA_SHAPE, LAT, LON))
            batches.append(batch)

        # Predict climatological probability.
        targets = torch.concat(batches, dim=0)
        lat_coeffs = weighted_brier_score.compute_lat_weights_fn(LAT)
        clim_prob = targets.mul(lat_coeffs).mean((-2, -1)).mean(0)
        preds = (
            clim_prob.reshape(1, *DATA_SHAPE, 1, 1).repeat(batch_size, 1, 1, LAT, LON).unsqueeze(1)
        )
        for i in range(5):
            batch = batches[i]
            weighted_brier_score.update(batch, preds)

        output = weighted_brier_score.compute()
        torch.testing.assert_close(output["brierscore"], output["brierclimscore"])
        torch.testing.assert_close(output["brierskillscore"], torch.tensor([[0.0], [0.0]]))

    def test_compare_iterative_implementation(self, weighted_brier_score):
        targets = []
        batch_size = 4
        num_batches = 5
        nsamples = batch_size * num_batches
        clim_prob = torch.zeros(DATA_SHAPE)
        lat_coeffs = weighted_brier_score.compute_lat_weights_fn(LAT)

        # Create random targets.
        for _ in range(num_batches):
            target = torch.randint(low=0, high=2, size=(batch_size, *DATA_SHAPE, LAT, LON))
            targets.append(target)
            # Calculate climatological probability (location-independent average).
            clim_prob += target.mul(lat_coeffs).mean((-2, -1)).sum(0)  # Latitude weighted mean.
        clim_prob /= nsamples
        clim_prob = clim_prob.reshape(1, *DATA_SHAPE, 1, 1).repeat(batch_size, 1, 1, LAT, LON)

        # Calculate brier skill score to compare.
        expected_brier_score = torch.zeros(DATA_SHAPE)
        expected_clim_brier_score = torch.zeros(DATA_SHAPE)

        # Compute with random predictions.
        nmembers = 10
        for i in range(num_batches):
            target = targets[i]
            pred = torch.randint(
                low=0, high=2, size=(batch_size, nmembers, *DATA_SHAPE, LAT, LON)
            ).float()

            weighted_brier_score.update(target, pred)

            expected_brier_score += (
                (pred.mean(1) - target).pow(2).mul(lat_coeffs).mean((-2, -1)).sum(0)
            )
            expected_clim_brier_score += (
                (clim_prob - target).pow(2).mul(lat_coeffs).mean((-2, -1)).sum(0)
            )
        expected_brier_score /= nsamples
        expected_clim_brier_score /= nsamples
        expected_brier_skill_score = 1 - (expected_brier_score / expected_clim_brier_score)

        output = weighted_brier_score.compute()

        assert weighted_brier_score.nsamples == nsamples
        torch.testing.assert_close(output["brierscore"], expected_brier_score)
        torch.testing.assert_close(output["brierclimscore"], expected_clim_brier_score)
        torch.testing.assert_close(output["brierskillscore"], expected_brier_skill_score)

    def test_optional_preprocess(self):
        torch.torch.manual_seed(0)

        def _preprocess(batch, pred):
            # 90th and 99th quantiles for each var and lev across gridpoints and batch.
            quantiles = torch.tensor([[[1.28]], [[2.3263]]])  # (quantiles, var, lev)
            quantiles = quantiles[:, None, :, :, None, None]

            batch = batch > quantiles  # (quantiles, bs, var, lev, lat, lon)
            batch = rearrange(batch, "q b var lev lat lon -> b q var lev lat lon")

            pred = pred > quantiles.unsqueeze(1)
            pred = rearrange(pred, "q b mem var lev lat lon -> b mem q var lev lat lon")

            return batch, pred

        metric = BrierSkillScore(
            data_shape=(1, 1),  # 1 var and lev.
            preprocess=_preprocess,
        )
        batch = torch.randn(100000, 1, 1, 1, 1)  # bs, var, lev, lat, lon.
        pred = torch.randn(100000, 1, 1, 1, 1, 1)  # bs, nmembers, var, lev, lat, lon.

        metric.update(batch, pred)

        output = metric.compute()
        assert_equal = functools.partial(torch.testing.assert_close, rtol=0.1, atol=0.001)
        # For qth quantile, mean prediction will be wrong (100-q) % of the time.
        assert_equal(output["brierclimscore"], torch.tensor([[[0.1]], [[0.01]]]))
        # For qth quantile, normal random predictions will be wrong 2q(100-q) % of the time.
        assert_equal(output["brierscore"], torch.tensor([[[0.18]], [[0.0198]]]))


class TestEra5BrierSkillScore:
    def test_output_keys(self):
        metric = Era5BrierSkillScore(
            level_variables=["geopotential", "temperature"], pressure_levels=[500, 700, 850]
        )
        bs, mem, timedelta, lev, lat, lon = 2, 5, 10, 3, 121, 240
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

        output = metric.compute()
        expected_metric_keys = [  # All expected keys for final metric for one variable.
            "brierskillscore_U10m_99.0%",
            "brierskillscore_U10m_99.9%",
            "brierskillscore_U10m_99.99%",
            "brierskillscore_U10m_1.0%",
            "brierskillscore_U10m_0.1%",
            "brierskillscore_U10m_0.01%",
        ]
        for expected_metric_key in expected_metric_keys:
            assert expected_metric_key in output
            assert output[expected_metric_key].numel() == timedelta

    def test_output_keys_with_timedelta_dimension(self):
        bs, mem, timedelta, lev, lat, lon = 2, 5, 2, 3, 121, 240
        metric = Era5BrierSkillScore(
            level_variables=["geopotential", "temperature"],
            pressure_levels=[500, 700, 850],
            lead_time_hours=24,
            rollout_iterations=timedelta,
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

        output = metric.compute()
        expected_metric_keys = [  # All expected keys for final metric for one variable.
            # For 24h lead time.
            "brierskillscore_U10m_99.0%_24h",
            "brierskillscore_U10m_99.9%_24h",
            "brierskillscore_U10m_99.99%_24h",
            "brierskillscore_U10m_1.0%_24h",
            "brierskillscore_U10m_0.1%_24h",
            "brierskillscore_U10m_0.01%_24h",
            # For 48h lead time.
            "brierskillscore_U10m_99.0%_48h",
            "brierskillscore_U10m_99.9%_48h",
            "brierskillscore_U10m_99.99%_48h",
            "brierskillscore_U10m_1.0%_48h",
            "brierskillscore_U10m_0.1%_48h",
            "brierskillscore_U10m_0.01%_48h",
        ]
        for expected_metric_key in expected_metric_keys:
            assert expected_metric_key in output
            assert output[expected_metric_key].numel() == 1

    def test_variable_indices(self):
        metric = Era5BrierSkillScore()

        for i, var_quantile in enumerate(
            [
                "U10m_99.0%",
                "U10m_99.9%",
                "U10m_99.99%",
                "U10m_1.0%",
                "U10m_0.1%",
                "U10m_0.01%",
            ]
        ):
            assert metric.metrics["surface"].variable_indices[var_quantile] == (i, 0, 0)
