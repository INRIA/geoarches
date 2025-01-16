import numpy as np
import torch
from einops import rearrange
from geoarches.dataloaders import era5
from geoarches.metrics.label_wrapper import LabelDictWrapper, add_timedelta_index
from scipy.stats import rankdata
from torchmetrics import Metric

from .metric_base import TensorDictMetricBase


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

    Accepted tensor shapes:
        targets: (batch, ..., lat, lon)
        preds: (batch, nmembers, ..., lat, lon)

    Return dictionary of metrics reduced over batch, lat, lon.
    """

    def __init__(
        self,
        n_members: int,
        data_shape: tuple = (4, 1),
    ):
        """
        Args:
            n_members: Number of ensemble members.
            data_shape: Shape of tensor to hold computed metric.
                e.g. if targets are shape (batch, timedelta, var, lev, lat, lon) then data_shape = (timedelta, var, lev).
                This class computes metric across batch, lat, lon dimensions.
        """
        Metric.__init__(self)
        self.n_members = n_members
        self.data_shape = data_shape

        self.add_state("n_batches", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "histogram", default=torch.zeros(*data_shape, n_members + 1), dist_reduce_fx="sum"
        )

    def update(self, targets, preds) -> None:
        """Update internal state with a batch of targets and predictions.

        Expects inputs to this function to be denormalized.
        Cannot handle arbitrary number of dimensions.

        Args:
            targets: Dictionary of target tensors. Expected input shape is (batch, ..., lat, lon)
            preds: Dictionary or list of dictionaries holding ensemble member predictions.
                   If dict, expected input shape is (batch, nmembers, ..., lat, lon). If list, (batch, ..., lat, lon).
        Returns:
            None
        """
        if isinstance(preds, list):
            preds = torch.stack(preds, dim=1)

        n_members = preds.shape[1]

        # Compute ranks of the targets with respect to ensemble predictions.
        # only works on cpu
        device = targets.device
        targets, preds = targets.cpu(), preds.cpu()
        combined = torch.cat([targets.unsqueeze(1), preds], dim=1)
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
        assert self.data_shape == ranks.shape[1:-2]
        ranks = rearrange(ranks, "b ... lat lon -> (b lat lon) (...)")
        bins = n_members + 1
        num_histograms = ranks.shape[-1]
        histograms = torch.zeros(num_histograms, bins)
        for i in range(num_histograms):
            x = torch.from_numpy(ranks[:, i]).float()
            histograms[i] += torch.histogram(
                x, bins=bins, range=(1, n_members + 2), density=True
            ).hist

        histograms = histograms.reshape(*(self.data_shape), bins)  # Get original shape.
        self.histogram += histograms.to(device)

    def compute(self) -> torch.Tensor:
        """Compute final metrics utliizing internal states."""
        total = self.histogram.sum(dim=-1).unsqueeze(-1)
        self.histogram = self.histogram / total  # Compute frequency.
        return dict(
            rankhist=self.histogram
            * (self.n_members + 1),  # Normalize by expected frequency (1/m+1).
        )


class Era5RankHistogram(TensorDictMetricBase):
    """Wrapper class around EnsembleMetrics for computing over surface and level variables.

    Handles batches coming from Era5 Dataloader.

    Accepted tensor shapes:
        targets: (batch, timedelta, var, level, lat, lon)
        preds: (batch, nmembers, timedelta, var, level, lat, lon)

    Return dictionary of metrics reduced over batch, lat, lon.
    """

    def __init__(
        self,
        n_members,
        surface_variables=era5.surface_variables,
        level_variables=era5.level_variables,
        pressure_levels=era5.pressure_levels,
        lead_time_hours: None | int = None,
        rollout_iterations: None | int = None,
        return_raw_dict: bool = False,
    ):
        """
        Args:
            n_members: Number of ensemble members.
            surface_variables: Names of level variables (used to get `variable_indices`).
            level_variables: Names of surface variables (used to get `variable_indices`).
            pressure_levels: pressure levels in data (used to get `variable_indices`).
            lead_time_hours: Timedelta between timestamps in multistep rollout.
                Set to explicitly handle predictions from multistep rollout.
                This option labels each timestep separately in output metric dict.
                Assumes that data shape of predictions/targets are [batch, ..., multistep, var, lev, lat, lon].
                FYI when set to None, Era5RankHistogram still handles natively any extra dimensions in targets/preds.
            rollout_iterations: Size of timedelta dimension (number of rollout iterations in multistep predictions).
                Set to explicitly handle metrics computed on predictions from multistep rollout.
                See param `lead_time_hours`.
            return_raw_dict: Whether to also return the raw output from the metrics.
        """
        if rollout_iterations:
            surface_data_shape = (rollout_iterations, len(surface_variables), 1)
            level_data_shape = (rollout_iterations, len(level_variables), len(pressure_levels))
        else:
            surface_data_shape = (len(surface_variables), 1)
            level_data_shape = (len(level_variables), len(pressure_levels))

        # Variable indices include quantile (var, lev) --> (var, lev, histogram bin number).
        # Enable LabelDictWrapper to extract metrics properly from RankHistogram output.
        def _add_bin_index(variable_indices):
            out = {}
            for var, var_lev_idx in variable_indices.items():
                for bin_idx in range(n_members + 1):
                    out[f"{var}_{bin_idx+1}"] = (*var_lev_idx, bin_idx)
            return out

        # Initialize separate metrics for level vars and surface vars.
        kwargs = {}
        if surface_variables:
            kwargs["surface"] = LabelDictWrapper(
                RankHistogram(data_shape=surface_data_shape, n_members=n_members),
                variable_indices=add_timedelta_index(
                    _add_bin_index(era5.get_surface_variable_indices(surface_variables)),
                    lead_time_hours=lead_time_hours,
                    rollout_iterations=rollout_iterations,
                ),
                return_raw_dict=return_raw_dict,
            )
        if level_variables:
            kwargs["level"] = LabelDictWrapper(
                RankHistogram(data_shape=level_data_shape, n_members=n_members),
                variable_indices=add_timedelta_index(
                    _add_bin_index(
                        era5.get_headline_level_variable_indices(pressure_levels, level_variables)
                    ),
                    lead_time_hours=lead_time_hours,
                    rollout_iterations=rollout_iterations,
                ),
                return_raw_dict=return_raw_dict,
            )
        super().__init__(**kwargs)
