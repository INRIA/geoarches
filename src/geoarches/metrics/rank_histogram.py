from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from scipy.stats import rankdata
from torchmetrics import Metric


class RankHistogram(Metric):
    """Iterative calculation of rank histogram as defined in the GenCast paper.

    Rank historgram measures reliability of the ensemble distribution or how well the ensemble distribution spread models the observation distribution.
    by measuring the frequency of where observations fall (rank) within the spread of sorted ensemble member predictions.
    Expectation is uniform distribution over all possible ranks.

    Given predictions from M ensembles, calculates a rank histogram with M+1 bins (ranks), accumulated over all evaluation times and grid cells.
    Returns one rank histogram (with M+1 bins) per variable and level.

    How to assign rank (ref: https://psl.noaa.gov/people/tom.hamill/sref2.pdf):
    Rank is the position of the groundtruth V within the ordered M ensemble predictions x_1, ..., x_m.
    If there is a "tie", ie. target is equal to one or more ensemble prediction, rank is randomly chosen among possible ranks.

    Cannot handle arbitrary number of dimensions for target and predicitions.
    """

    def __init__(
        self,
        n_members: int,
        data_shape: tuple = (4, 1),
        input_key: str = "surface",
        display_metrics: list[Tuple[str, int, int]] = [("T2m", 2, 0), ("U10", 0, 0), ("SP", 3, 0)],
    ):
        """
        Args:
            n_members: Number of ensemble members.
            data_shape: Shape of data [var, lev].
            input_key: Key for output tensor in the predictions and targets in `pred` and `batch` dictionaries.
            display_metrics: List of tuples holding (variable name, variable index, level index) to define for which variables to return metrics for.
        """
        Metric.__init__(self)
        self.n_members = n_members
        self.data_shape = data_shape
        self.display_metrics = display_metrics
        self.input_key = input_key

        self.add_state("n_batches", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "histogram", default=torch.zeros(*data_shape, n_members + 1), dist_reduce_fx="sum"
        )

    def _check_dimensions(self, n_members, n_vars, n_levels):
        """Verify input dimensions are expected."""
        if self.n_members != n_members:
            raise ValueError(
                f"Number of ensemble members {n_members} is not expected. Expected: {self.n_members}."
            )
        if self.data_shape[0] != n_vars:
            raise ValueError(
                f"Number of variables {n_vars} is not expected. Expected: {self.data_shape[0]}."
            )
        if self.data_shape[1] != n_levels:
            raise ValueError(
                f"Number of levels {n_levels} is not expected. Expected: {self.data_shape[1]}."
            )

    def update(self, batch, preds) -> None:
        """Update internal state with a batch of targets and predictions.

        Expects inputs to this function to be denormalized.
        Cannot handle arbitrary number of dimensions.

        Args:
            batch: Dictionary of target tensors. Expected input shape is (batch var, level, lat, lon)
            preds: Dictionary or list of dictionaries holding ensemble member predictions.
                   If dict, expected input shape is (batch, nmembers, var, level, lat, lon). If list, (batch, var, level, lat, lon).
        Returns:
            None
        """
        if isinstance(preds, list):
            preds = {self.input_key: torch.stack([x[self.input_key] for x in preds], dim=1)}

        pred = preds[self.input_key].float()
        target = batch[self.input_key].float()

        _, n_members, n_vars, n_levels, _, _ = pred.shape
        self._check_dimensions(n_members, n_vars, n_levels)

        # Compute ranks of the targets with respect to ensemble predictions.
        combined = torch.cat([target.unsqueeze(1), pred], dim=1)
        ranks = rankdata(combined, method="min", axis=1)

        # Handle rank ties.
        ties = np.sum(ranks[:, [0], ...] == ranks[:, 1:, ...], axis=1)  # Count ties.
        ranks = ranks[:, 0, ...]  # Get rank of targets: (batch, var, level, lat, lon).
        index = ties > 0
        ranks[index] = [
            np.random.randint(rank, rank + num_ties + 1)
            for rank, num_ties in zip(ranks[index], ties[index])
        ]  # Randomly select rank amongst ties.

        # Count frequency of ranks across lat, lon, batch.
        # (Might have smarter ways at the expense of memory: https://stackoverflow.com/questions/69429586/how-to-get-a-histogram-of-pytorch-tensors-in-batches)
        ranks = rearrange(ranks, "b var lev lat lon -> (b lat lon) var lev")
        # Loop over variables since histogramdd can only handle one extra dimension.
        for i in range(n_vars):
            x = torch.from_numpy(ranks[:, i, :]).float()
            self.histogram[i] += torch.histogramdd(
                x, bins=n_members + 1, range=(1, n_members + 2), density=True
            ).hist

    def compute(self) -> torch.Tensor:
        """Compute final metrics utliizing internal states."""
        self.histogram = self.histogram / self.histogram.sum(dim=-1)  # Compute frequency.
        metrics = dict(
            rank_histogram=self.histogram
            * (self.n_members + 1),  # Normalize by expected frequency (1/m+1).
        )
        out = dict()
        for name, var, lev in self.display_metrics:
            out.update({name + "_" + k: v[var, lev] for k, v in metrics.items()})

        return out
