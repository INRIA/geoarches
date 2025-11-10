import numpy as np
import pandas as pd

from geoarches.dataloaders import era5


class Era5ForecastWithPrediction(era5.Era5Forecast):
    """
    loads both input to forecast data and prediction made by archesweather mode
    """

    def __init__(
        self,
        stats_cfg,
        path="data/era5_240/full/",
        domain="train",
        filename_filter=None,
        lead_time_hours=24,
        pred_path="data/era5_pred_archesweather-S/",
        load_prev=False,
        load_hard_neg=False,
        variables=None,
        dimension_indexers=None,
        pred_dimension_indexers=None,
        **kwargs,
    ):
        """Args:

        path: Single filepath or directory holding groundtruth files.
        domain: Specify data split for the default filename filters (eg. train,
        val, test, testz0012..).
        filename_filter: To filter files within `path` based on filename.  If
        set, does not use `domain` param.
            If None, filters files based on `domain`.
        lead_time_hours: Time difference between current state and previous and
        future states.
        pred_path: Single filepath or directory holding model prediction files
        to also load.
        load_prev: Whether to load state at previous timestamp (current time -
        lead_time_hours).
        load_hard_neg: Whether to additionallty load hard negative example for
        contrastive learning.
        variables: Variables to load from dataset. Dict holding variable lists
        mapped by their keys to be processed into tensordict.
            e.g. {surface:[...], level:[...] By default uses standard 6 level
            and 4 surface vars.
        """
        super().__init__(
            stats_cfg=stats_cfg,
            path=path,
            domain=domain,
            lead_time_hours=lead_time_hours,
            filename_filter=filename_filter,
            load_prev=load_prev,
            variables=variables,
            dimension_indexers=dimension_indexers,
            **kwargs,
        )
        self.load_prev = load_prev
        self.load_hard_neg = load_hard_neg
        # self.filename_filter is already init
        if pred_path is not None:
            self.pred_ds = era5.Era5Dataset(
                path=pred_path,
                dimension_indexers=dimension_indexers | pred_dimension_indexers,
                filename_filter=self.filename_filter,
                variables=self.variables,
            )

            if "train" not in domain:
                # re-select timestamps.
                # we always evaluate on a single 'target eval year', but sometimes we
                # need some extra context from the year before & after (if the first
                # prediction is on jan 1 for instance, we need the state from the end
                # of the previous year).
                # filename_filter filters file crudely but sometimes we need to remove
                # some dates in the files. typically when we evaluate, we evaluate on a
                # single year but we need context from the previous and later year,
                # hence we need to load 3 years.
                # if this is the case that we have loaded 3 years then we take the
                # middle year, which is the one that is supposed to be fully evaluated.
                # if there is a single year, we don't reset timestamp bounds.
                # TODO(geco): the clean way would be to pass in a date range as argument
                # and to compute the correct filename filter from those dates.
                # then given the date range we can also adjust timestamp bounds.

                # find year for which we want to keep timestamps
                allowed_years = [
                    y
                    for y in range(1979, 2024)
                    if any(
                        self.filename_filter(f"{y}_{hour}h") for hour in ("0", "06", "12", "18")
                    )
                ]
                year = allowed_years[0] if len(allowed_years) == 1 else allowed_years[1]
                start_time = np.datetime64(f"{year}-01-01T00:00:00")
                if self.load_prev:
                    start_time = start_time - self.lead_time_hours * np.timedelta64(1, "h")
                end_time = np.datetime64(
                    f"{year + 1}-01-01T00:00:00"
                ) + self.lead_time_hours * np.timedelta64(1, "h")
                self.pred_ds.set_timestamp_bounds(start_time, end_time)

        # TODO: is the stats file in geoarches ?
        # geoarches_stats_path = importlib.resources.files(geoarches_stats)
        # deltapred_path = geoarches_stats_path / "stats/deltapred24_aw-s_stats.pt"
        # deltapred_stats = torch.load(deltapred_path, weights_only=True)

    def __len__(self):
        di = self.lead_time_hours // self.timedelta if "train" in self.domain else 0
        return super().__len__() - di  # because we cannot access first element

    def __getitem__(self, i, normalize=True, load_hard_neg=True):
        out = {}
        di = self.lead_time_hours // self.timedelta
        shift_main = di if "train" in self.domain else 0  # because we cannot access first element
        out = super().__getitem__(
            i + shift_main, normalize=False
        )  # get original data, +di because we need to fetch next one

        # handle prediction. if load_prev, we have to fetch next one
        if hasattr(self, "pred_ds"):
            out["pred_state"], pred_timestamp = self.pred_ds.__getitem__(
                i + di if self.load_prev else i, return_timestamp=True
            )
            assert out["timestamp"] == pred_timestamp, (
                f"badly aligned {i}:"
                + pd.Timestamp(out["timestamp"].int().item() * 10**9).strftime("%Y-%m-%d-%H-%M")
                + "/"
                + pd.Timestamp(pred_timestamp.int().item() * 10**9).strftime("%Y-%m-%d-%H-%M")
            )

        if normalize:
            out = self.normalize(out)

        if self.load_hard_neg and load_hard_neg:
            rb = 2 * np.random.randint(2) - 1
            ri = np.random.randint(1, 9) * rb  # check effets de bords
            if i + ri < 0 or i + ri >= len(self):
                ri = -ri
            out["neg_next_state"] = self.__getitem__(
                i + ri, normalize=normalize, load_hard_neg=False
            )

        return out

    def normalize(self, batch):
        """
        same as parent class for now
        """
        out = super().normalize(batch)
        return out
