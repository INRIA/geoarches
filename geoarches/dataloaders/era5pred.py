import numpy as np

from geoarches.dataloaders import era5, nan_util


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
        pred_path: str | None = None,
        load_prev=False,
        load_hard_neg=False,
        variables=None,
        dimension_indexers=None,
        pred_dimension_indexers=None,
        interpolate_input: nan_util.NanInterpolationMethod | None = None,
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
            interpolate_input=interpolate_input,
            **kwargs,
        )
        self.load_prev = load_prev
        self.load_hard_neg = load_hard_neg
        self.interpolate_input = interpolate_input
        # self.filename_filter is already init
        if pred_path is not None:
            self.pred_ds = era5.Era5Dataset(
                path=pred_path,
                domain="all",
                dimension_indexers=dimension_indexers | pred_dimension_indexers,
                filename_filter=self.filename_filter,
                variables=self.variables,
                interpolate_nans=interpolate_input,
            )
            # pred_ds should be synchronized with main ds, so we adjust timestamp bounds
            # accordingly.
            start_time = min(x[-1] for x in self.timestamps)
            end_time = max(x[-1] for x in self.timestamps)
            self.pred_ds.set_timestamp_bounds(start_time, end_time + np.timedelta64(1, "s"))

        # TODO: is the stats file in geoarches ?
        # geoarches_stats_path = importlib.resources.files(geoarches_stats)
        # deltapred_path = geoarches_stats_path / "stats/deltapred24_aw-s_stats.pt"
        # deltapred_stats = torch.load(deltapred_path, weights_only=True)

    def __getitem__(self, i, normalize=True, load_hard_neg=True):
        out = super().__getitem__(i)

        if hasattr(self, "pred_ds"):
            nptime = np.datetime64(out["timestamp"].int().item(), "s")
            out["pred_state"] = self.pred_ds.select_from_nptime(nptime)
            out["pred_state"] = self.normalize(out["pred_state"])
            out["pred_state"] = nan_util.post_norm_interpolate_nans(
                out["pred_state"], self.interpolate_input
            )

        # if normalize:
        #   out = self.normalize(out)

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
