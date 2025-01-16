import numpy as np
import pandas as pd
from geoarches.dataloaders import era5, netcdf


class Era5ForecastWithPrediction(era5.Era5Forecast):
    """
    loads both input to forecast data and prediction made by archesweather mode
    """

    def __init__(
        self,
        path="data/era5_240/full/",
        domain="train",
        filename_filter=None,
        lead_time_hours=24,
        pred_path="data/era5_pred_archesweather-S/",
        load_prev=False,
        norm_scheme="pangu",
        load_hard_neg=False,
        variables=None,
        **kwargs,
    ):
        super().__init__(
            path=path,
            domain=domain,
            lead_time_hours=lead_time_hours,
            filename_filter=filename_filter,
            norm_scheme=norm_scheme,
            load_prev=load_prev,
            **kwargs,
        )
        self.load_prev = load_prev
        self.load_hard_neg = load_hard_neg
        # self.filename_filter is already init
        if pred_path is not None:
            self.pred_ds = netcdf.NetcdfDataset(
                path=pred_path,
                filename_filter=self.filename_filter,
                variables=self.variables,
            )
            self.pred_ds.convert_to_tensordict = self.convert_to_tensordict

            if domain in ("val", "test", "test_z0012"):
                # re-select timestamps
                year = 2019 if domain == "val" else 2020
                start_time = np.datetime64(f"{year}-01-01T00:00:00")
                if self.load_prev:
                    start_time = start_time - self.lead_time_hours * np.timedelta64(
                        1, "h"
                    )
                end_time = np.datetime64(
                    f"{year+1}-01-01T00:00:00"
                ) + self.lead_time_hours * np.timedelta64(1, "h")
                self.pred_ds.set_timestamp_bounds(start_time, end_time)

        # TODO: is the stats file in geoarches ?
        # geoarches_stats_path = importlib.resources.files(geoarches_stats)
        # deltapred_path = geoarches_stats_path / "stats/deltapred24_aw-s_stats.pt"
        # deltapred_stats = torch.load(deltapred_path, weights_only=True)

    def __len__(self):
        di = self.lead_time_hours // self.timedelta if self.domain == "train" else 0
        return super().__len__() - di  # because we cannot access first element

    def __getitem__(self, i, normalize=True, load_hard_neg=True):
        out = {}
        di = self.lead_time_hours // self.timedelta
        shift_main = (
            di if self.domain == "train" else 0
        )  # because we cannot access first element
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
                + pd.Timestamp(out["timestamp"].int().item() * 10**9).strftime(
                    "%Y-%m-%d-%H-%M"
                )
                + "/"
                + pd.Timestamp(pred_timestamp.int().item() * 10**9).strftime(
                    "%Y-%m-%d-%H-%M"
                )
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
