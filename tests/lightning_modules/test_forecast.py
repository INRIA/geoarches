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
