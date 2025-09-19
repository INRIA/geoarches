import torch
from tensordict.tensordict import TensorDict

from geoarches.lightning_modules.forecast import ForecastModule


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
        stats_cfg = self.DummyCfg()
        # To make DummyStats instantiable by Hydra, wrap it in a dict with _target_
        stats_cfg.module = {"_target_": self.DummyStats}
        module_cfg = self.DummyCfg()
        module_cfg.backbone = None
        module_cfg.embedder = None
        forecast_module = ForecastModule(
            cfg=module_cfg,
            stats_cfg=stats_cfg,
        )
        # The loss_coeffs will be computed within ForecastModule.__init__
        # using the instantiated self.DummyStats.
        forecast_module.to("cpu")

        # Create dummy pred and gt tensordicts
        pred = TensorDict(
            {
                "level": torch.zeros(1, 1, 2, 2),
                "surface": torch.zeros(1, 1, 2, 2),
            },
            batch_size=[],
        )

        # Create gt with NaNs
        gt = TensorDict(
            {
                "level": torch.randn(1, 1, 2, 2),
                "surface": torch.randn(1, 1, 2, 2),
            },
            batch_size=[],
        )
        gt["level"][0, 0, 0, 0] = torch.nan
        gt["surface"][0, 0, 0, 0] = torch.nan

        # Compute loss
        loss = forecast_module.loss(pred, gt)

        # Assert that the loss is not NaN
        assert not torch.isnan(loss)
