import torch
from hydra import compose, initialize
from tensordict.tensordict import TensorDict

from geoarches.lightning_modules.forecast import ForecastModule

with initialize(config_path="../../geoarches/configs"):
    cfg = compose(config_name="test_awdet.yaml")


class TestForecastModule:
    class DummyStats:
        def __init__(self):
            self.variables = {"level": ["var1"], "surface": ["var2"]}
            self.levels = [500]

        def compute_loss_coeffs(self):
            # Return loss coeffs of 1 for all variables
            level_coeffs = torch.ones(1, 1, 1, 1)
            surface_coeffs = torch.ones(1, 1, 1)
            return TensorDict({"level": level_coeffs, "surface": surface_coeffs}, batch_size=[])

    class DummyCfg:
        def __init__(self):
            self.compute_loss_coeffs_args = {}
            self.train = self
            self.val = self
            self.inference = self
            self.metrics = {}  # Should be a dict, not a list
            self.metrics_kwargs = {}

    def test_loss_with_nans_in_gt(self):
        # Create a dummy ForecastModule
        # To make DummyStats instantiable by Hydra, wrap it in a dict with _target_

        forecast_module = ForecastModule(
            cfg=cfg.module,
            stats_cfg=cfg.stats,
        )
        # The loss_coeffs will be computed within ForecastModule.__init__
        # using the instantiated self.DummyStats.
        forecast_module.to("cpu")
        img_size = cfg.module.embedder.img_size
        surf_ch = len(cfg.stats.module.variables["surface"])
        level_ch = len(cfg.stats.module.variables["level"])

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
