import omegaconf
import torch
from tensordict.tensordict import TensorDict

from geoarches.lightning_modules.forecast import ForecastModuleWithCond
from tests.fixtures.forecast import cfg as det_cfg

omegaconf.OmegaConf.set_struct(det_cfg, True)
with omegaconf.open_dict(det_cfg):
    det_cfg.module.embedder.forcings_ch = 2
    det_cfg.module.embedder.forcings_embedding = "surface"


class TestForecastModule:
    def build_model(self, build_det_model=False):
        det_module = ForecastModuleWithCond(det_cfg.module, det_cfg.stats, **det_cfg.module.module)

        return det_module

    def get_dims(self):
        surface_ch = det_cfg.module.embedder.surface_ch
        level_ch = det_cfg.module.embedder.level_ch
        forc_ch = det_cfg.module.embedder.forcings_ch
        img_size = det_cfg.module.embedder.img_size
        return surface_ch, level_ch, forc_ch, img_size

    def test_loss_with_nans_in_gt(self):
        # Create a dummy ForecastModule
        # To make DummyStats instantiable by Hydra, wrap it in a dict with _target_
        forecast_module = ForecastModuleWithCond(
            cfg=det_cfg.module,
            stats_cfg=det_cfg.stats,
        )
        # The loss_coeffs will be computed within ForecastModule.__init__
        # using the instantiated self.DummyStats.
        forecast_module.to("cpu")
        surf_ch, level_ch, _, img_size = self.get_dims()
        # Create pred without NaNs.
        pred = TensorDict(
            {
                "level": torch.zeros(1, level_ch, *img_size),
                "surface": torch.zeros(1, surf_ch, *img_size[-2:]),
            },
            batch_size=[],
        )

        # Create gt with NaNs.
        gt = TensorDict(
            {
                "level": torch.zeros(1, level_ch, *img_size),
                "surface": torch.zeros(1, surf_ch, *img_size[-2:]),
            },
            batch_size=[],
        )
        gt["level"][0, 0, 0, 0] = torch.nan
        gt["surface"][0, 0, 0, 0] = torch.nan

        loss = forecast_module.loss(pred, gt)

        assert not torch.isnan(loss)

    def test_forward_multistep_updates_day_of_year(self):
        forecast_module = ForecastModuleWithCond(
            cfg=det_cfg.module,
            stats_cfg=det_cfg.stats,
            cond_times=["day_of_year", "hour_of_day"],
            **det_cfg.module.module,
        )
        forecast_module.to("cpu")
        forecast_module.eval()

        surf_ch, level_ch, forc_ch, img_size = self.get_dims()

        # 2020-01-01 00:00:00 UTC = 1577836800 (day_of_year = 1, month = 1, hour = 0)
        start_ts = 1577836800
        batch = TensorDict(
            {
                "prev_state": {
                    "level": torch.randn(1, level_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "state": {
                    "level": torch.randn(1, level_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "next_state": {
                    "level": torch.randn(1, level_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "forcings": torch.randn(1, forc_ch, *img_size[-2:]),
                "future_forcings": torch.randn(1, 4, forc_ch, *img_size[-2:]),
                "timestamp": torch.tensor([start_ts]),
                "lead_time_hours": torch.tensor([24]),
                "month": torch.tensor([1]),
                "day_of_year": torch.tensor([1]),
                "hour_of_day": torch.tensor([0]),
            },
            batch_size=[1],
        )

        with torch.no_grad():
            preds_future, loop_batch = forecast_module.forward_multistep(
                batch, iters=3, return_loop_batch=True
            )

        assert preds_future["level"].shape[1] == 3
        assert preds_future["surface"].shape[1] == 3
        # After 3 iterations of 24h: day_of_year should be 1 + 3 = 4
        assert loop_batch["day_of_year"].item() == 4
        assert loop_batch["month"].item() == 1
        assert loop_batch["hour_of_day"].item() == 0
        assert loop_batch["timestamp"].item() == start_ts + 3 * 24 * 3600
