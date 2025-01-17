import numpy as np
import torch

from geoarches.metrics.rank_histogram import Era5RankHistogram, RankHistogram


class TestRankHistogram:
    def test_one_batch(self):
        rank_histogram = RankHistogram(n_members=3, data_shape=(1, 1))
        target = torch.tensor([1, 2, 3, 4, 1]).reshape(5, 1, 1, 1, 1)  # (bs=5, var, lev, lat, lon)
        # Predictions for 3 ensemble members.
        pred = torch.tensor(
            [
                [2, 3, 4],  # Rank 1.
                [1, 3, 4],  # Rank 2.
                [1, 2, 4],  # Rank 3.
                [1, 2, 3],  # Rank 4.
                [2, 3, 4],  # Rank 1.
            ]
        ).reshape(5, 3, 1, 1, 1, 1)  # (bs=5, nmembers=3, var, lev, lat, lon)

        rank_histogram.update(target, pred)
        output = rank_histogram.compute()

        torch.testing.assert_close(output["rankhist"], torch.tensor([[[1.6, 0.8, 0.8, 0.8]]]))

    def test_two_batches(self):
        rank_histogram = RankHistogram(n_members=3, data_shape=(1, 1))
        target = torch.tensor([1, 2, 3, 4, 1]).reshape(5, 1, 1, 1, 1)  # (bs=5, var, lev, lat, lon)
        # Predictions for 3 ensemble members.
        pred1 = torch.tensor(
            [
                [2, 3, 4],  # Rank 1.
                [1, 3, 4],  # Rank 2.
                [1, 2, 4],  # Rank 3.
                [1, 2, 3],  # Rank 4.
                [2, 3, 4],  # Rank 1.
            ]
        ).reshape(5, 3, 1, 1, 1, 1)  # (bs=5, nmembers=3, var, lev, lat, lon)
        pred2 = torch.tensor(
            [
                [0, 2, 3],  # Rank 2.
                [0, 1, 3],  # Rank 3.
                [0, 1, 2],  # Rank 4.
                [0, 1, 2],  # Rank 4.
                [0, 2, 3],  # Rank 2.
            ]
        ).reshape(5, 3, 1, 1, 1, 1)  # (bs=5, nmembers=3, var, lev, lat, lon)

        rank_histogram.update(target, pred1)
        rank_histogram.update(target, pred2)
        output = rank_histogram.compute()

        torch.testing.assert_close(output["rankhist"], torch.tensor([[[0.8, 1.2, 0.8, 1.2]]]))

    def test_one_batch_two_vars(self):
        rank_histogram = RankHistogram(
            n_members=3,
            data_shape=(2, 1),  # 2 vars.
        )
        target = torch.tensor([[1, 2], [2, 2]]).reshape(
            2, 2, 1, 1, 1
        )  # (bs=2, var=2, lev, lat, lon)
        # Predictions for 3 ensemble members and 2 vars.
        pred = torch.tensor(
            [
                [[2, 3, 4], [1, 3, 4]],  # Rank for var 1: 1, Rank for var 2: 2
                [[1, 3, 4], [1, 3, 4]],  # Rank for var 1: 2, Rank for var 2: 2
            ]
        ).reshape(2, 3, 2, 1, 1, 1)  # (bs=2, nmembers=3, var=2, lev, lat, lon)

        rank_histogram.update(target, pred)
        output = rank_histogram.compute()

        torch.testing.assert_close(
            output["rankhist"], torch.tensor([[[2.0, 2.0, 0, 0]], [[0, 4.0, 0, 0]]])
        )

    def test_random_uniform(self):
        torch.manual_seed(0)
        rank_histogram = RankHistogram(
            n_members=5,
            data_shape=(3, 2),  # 3 vars, 2 lev.
        )
        target = torch.zeros(size=(10000, 3, 2, 1, 1))  # (bs=10e4, var=3, lev=2, lat, lon)
        # Predictions for 5 ensemble members and 6 vars.
        pred = torch.randint(
            0, 1, size=(10000, 5, 3, 2, 1, 1)
        )  # (bs=10e4, mem=5, var=3, lev=2, lat, lon)

        rank_histogram.update(target, pred)
        output = rank_histogram.compute()

        # Normalized bin values should be close to 1 everywhere.
        torch.testing.assert_close(output["rankhist"], torch.ones(3, 2, 6), atol=0, rtol=0.1)

    def test_handle_ties(self):
        rank_histogram = RankHistogram(n_members=3, data_shape=(1, 1))
        target = torch.tensor([3]).reshape(1, 1, 1, 1, 1)  # (bs=1, var, lev, lat, lon)
        # Predictions for 3 ensemble members.
        pred = torch.tensor(
            [
                [1, 2, 3],  # Rank 3 or 4.
            ]
        ).reshape(1, 3, 1, 1, 1, 1)  # (bs=1, nmembers=3, var, lev, lat, lon)

        np.random.seed(0)
        rank_histogram.update(target, pred)
        output = rank_histogram.compute()

        torch.testing.assert_close(output["rankhist"], torch.tensor([[[0, 0, 4.0, 0]]]))

        np.random.seed(1)
        output = rank_histogram.reset()
        rank_histogram.update(target, pred)
        output = rank_histogram.compute()
        torch.testing.assert_close(output["rankhist"], torch.tensor([[[0, 0, 0, 4.0]]]))


class TestEra5RankHistogram:
    def test_output_keys(self):
        bs, mem, timedelta, lev, lat, lon = 2, 3, 5, 3, 121, 240
        metric = Era5RankHistogram(
            n_members=mem,
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

        output = metric.compute()
        expected_metric_keys = [  # All expected keys for final metric for one variable.
            "rankhist_U10m_1_6h",
            "rankhist_U10m_1_12h",
            "rankhist_U10m_1_18h",
            "rankhist_U10m_1_24h",
            "rankhist_U10m_1_30h",
            "rankhist_U10m_2_6h",
            "rankhist_U10m_2_12h",
            "rankhist_U10m_2_18h",
            "rankhist_U10m_2_24h",
            "rankhist_U10m_2_30h",
            "rankhist_U10m_3_6h",
            "rankhist_U10m_3_12h",
            "rankhist_U10m_3_18h",
            "rankhist_U10m_3_24h",
            "rankhist_U10m_3_30h",
            "rankhist_U10m_4_6h",
            "rankhist_U10m_4_12h",
            "rankhist_U10m_4_18h",
            "rankhist_U10m_4_24h",
            "rankhist_U10m_4_30h",
        ]
        for expected_metric_key in expected_metric_keys:
            assert expected_metric_key in output
            assert output[expected_metric_key].numel() == 1
