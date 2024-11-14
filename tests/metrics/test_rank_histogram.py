import numpy as np
import pytest
import torch
from geoarches.metrics import rank_histogram as ensemble_metric

LAT = 3
LON = 5
INPUT_KEY = "output"
DATA_SHAPE = (2, 1)


@pytest.fixture
def rank_histogram():
    return


class TestRankHistogram:
    def test_one_batch(self):
        rank_histogram = ensemble_metric.RankHistogram(
            n_members=3,
            data_shape=(1, 1),
            input_key=INPUT_KEY,
            display_metrics=[("var1", 0, 0)],
        )
        batch = torch.tensor([1, 2, 3, 4, 1]).reshape(5, 1, 1, 1, 1)  # (bs=5, var, lev, lat, lon)
        batch = {INPUT_KEY: batch}
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
        pred = {INPUT_KEY: pred}

        rank_histogram.update(batch, pred)
        output = rank_histogram.compute()

        torch.testing.assert_close(
            output["var1_rank_histogram"], torch.tensor([1.6, 0.8, 0.8, 0.8])
        )

    def test_two_batches(self):
        rank_histogram = ensemble_metric.RankHistogram(
            n_members=3,
            data_shape=(1, 1),
            input_key=INPUT_KEY,
            display_metrics=[("var1", 0, 0)],
        )
        batch = torch.tensor([1, 2, 3, 4, 1]).reshape(5, 1, 1, 1, 1)  # (bs=5, var, lev, lat, lon)
        batch = {INPUT_KEY: batch}
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
        pred1 = {INPUT_KEY: pred1}
        pred2 = torch.tensor(
            [
                [0, 2, 3],  # Rank 2.
                [0, 1, 3],  # Rank 3.
                [0, 1, 2],  # Rank 4.
                [0, 1, 2],  # Rank 4.
                [0, 2, 3],  # Rank 2.
            ]
        ).reshape(5, 3, 1, 1, 1, 1)  # (bs=5, nmembers=3, var, lev, lat, lon)
        pred2 = {INPUT_KEY: pred2}

        rank_histogram.update(batch, pred1)
        rank_histogram.update(batch, pred2)
        output = rank_histogram.compute()

        torch.testing.assert_close(
            output["var1_rank_histogram"], torch.tensor([0.8, 1.2, 0.8, 1.2])
        )

    def test_one_batch_two_vars(self):
        rank_histogram = ensemble_metric.RankHistogram(
            n_members=3,
            data_shape=(2, 1),  # 2 vars.
            input_key=INPUT_KEY,
            display_metrics=[("var1", 0, 0), ("var2", 1, 0)],
        )
        batch = torch.tensor([[1, 2], [2, 2]]).reshape(
            2, 2, 1, 1, 1
        )  # (bs=2, var=2, lev, lat, lon)
        batch = {INPUT_KEY: batch}
        # Predictions for 3 ensemble members and 2 vars.
        pred = torch.tensor(
            [
                [[2, 3, 4], [1, 3, 4]],  # Rank for var 1: 1, Rank for var 2: 2
                [[1, 3, 4], [1, 3, 4]],  # Rank for var 1: 2, Rank for var 2: 2
            ]
        ).reshape(2, 3, 2, 1, 1, 1)  # (bs=2, nmembers=3, var=2, lev, lat, lon)
        pred = {INPUT_KEY: pred}

        rank_histogram.update(batch, pred)
        output = rank_histogram.compute()

        torch.testing.assert_close(output["var1_rank_histogram"], torch.tensor([2.0, 2.0, 0, 0]))
        torch.testing.assert_close(output["var2_rank_histogram"], torch.tensor([0, 4.0, 0, 0]))

    def test_handle_ties(self):
        rank_histogram = ensemble_metric.RankHistogram(
            n_members=3,
            data_shape=(1, 1),
            input_key=INPUT_KEY,
            display_metrics=[("var1", 0, 0)],
        )
        batch = torch.tensor([3]).reshape(1, 1, 1, 1, 1)  # (bs=1, var, lev, lat, lon)
        batch = {INPUT_KEY: batch}
        # Predictions for 3 ensemble members.
        pred = torch.tensor(
            [
                [1, 2, 3],  # Rank 3 or 4.
            ]
        ).reshape(1, 3, 1, 1, 1, 1)  # (bs=1, nmembers=3, var, lev, lat, lon)
        pred = {INPUT_KEY: pred}

        np.random.seed(0)
        rank_histogram.update(batch, pred)
        output = rank_histogram.compute()

        torch.testing.assert_close(output["var1_rank_histogram"], torch.tensor([0, 0, 4.0, 0]))

        np.random.seed(1)
        output = rank_histogram.reset()
        rank_histogram.update(batch, pred)
        output = rank_histogram.compute()
        torch.testing.assert_close(output["var1_rank_histogram"], torch.tensor([0, 0, 0, 4.0]))
