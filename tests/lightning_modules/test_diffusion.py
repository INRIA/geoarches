from pprint import pprint

import torch
from hydra import compose, initialize
from tensordict.tensordict import TensorDict

from geoarches.dataloaders.era5_constants import (
    arches_default_level_variables,
    arches_default_surface_variables,
)
from geoarches.lightning_modules.diffusion import DiffusionModule

with initialize(version_base="1.2", config_path="../../geoarches/configs"):
    cfg = compose(
        config_name="test_agen",
        overrides=["module.module.load_deterministic_model=Null", "module.embedder.forcings_ch=2"],
    )
    pprint(cfg)


class TestDiffusionModule:
    def build_model(self):
        img_size = cfg.module.embedder.img_size
        surf_ch = len(arches_default_surface_variables)
        lvl_ch = len(arches_default_level_variables)
        forc_ch = cfg.module.embedder.forcings_ch
        module = DiffusionModule(cfg.module, cfg.stats, **cfg.module.module)

        return module, img_size, surf_ch, lvl_ch, forc_ch

    def test_forward_with_forcing(self):
        module, img_size, surf_ch, lvl_ch, forc_ch = self.build_model()

        batch = TensorDict(
            {
                "prev_state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "next_state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "pred_state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "forcing": torch.randn(1, forc_ch, *img_size[-2:]),
                "timestamp": torch.tensor(
                    [
                        0,
                    ]
                ),
                "lead_time_hours": torch.tensor(
                    [
                        24,
                    ]
                ),
            },
            batch_size=[1],
        )

        # The loss_coeffs will be computed within ForecastModule.__init__
        # using the instantiated self.DummyStats.
        out = module.forward(
            batch=batch,
            noisy_next_state=batch["next_state"] * 0.1,
            timesteps=torch.tensor([0.5], dtype=torch.float32),
        )

        assert not any(torch.isnan(v).any() for v in out.values()), "Output contains NaNs"

    def test_loss_with_nans_in_gt(self):
        # Create a dummy ForecastModule
        module, img_size, surf_ch, lvl_ch, forc_ch = self.build_model()
        pred = TensorDict(
            {
                "level": torch.zeros(1, lvl_ch, *img_size),
                "surface": torch.zeros(1, surf_ch, *img_size[-2:]),
            },
            batch_size=[],
        )

        # Create gt with NaNs.
        gt = TensorDict(
            {
                "level": torch.randn(1, *img_size),
                "surface": torch.randn(1, 1, *img_size[-2:]),
            },
            batch_size=[],
        )
        lat_half, lon_half = img_size[-2] // 2, img_size[-1] // 2
        gt["level"][0, 0, lat_half - 10 : lat_half, lon_half - 10 : lon_half] = torch.nan
        gt["surface"][0, 0, lat_half - 10 : lat_half, lon_half - 10 : lon_half] = torch.nan

        loss = module.loss(pred, gt, timesteps=torch.tensor([0.5], dtype=torch.float32))

        assert not torch.isnan(loss)

    def test_sample_rollout_w_forcings(self):
        module, img_size, surf_ch, lvl_ch, forc_ch = self.build_model()

        batch = TensorDict(
            {
                "prev_state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "pred_state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "forcing": torch.randn(1, forc_ch, *img_size[-2:]),
                "timestamp": torch.tensor(
                    [
                        0,
                    ]
                ),
                "lead_time_hours": torch.tensor(
                    [
                        24,
                    ]
                ),
            },
            batch_size=[
                1,
            ],
        )

        rollout = module.sample_rollout(
            batch,
            iterations=2,
        )

        assert len(rollout) == 2
        for state in rollout:
            assert not any(torch.isnan(v).any() for v in state.values()), "Output contains NaNs"

    def test_sample_rollout_w_forcings_custom_update(self):
        module, img_size, surf_ch, lvl_ch, forc_ch = self.build_model()

        batch = TensorDict(
            {
                "pred_state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "prev_state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "forcing": torch.randn(1, forc_ch, *img_size[-2:]),
                "timestamp": torch.tensor(
                    [
                        0,
                    ]
                ),
                "lead_time_hours": torch.tensor(
                    [
                        24,
                    ]
                ),
            },
            batch_size=[
                1,
            ],
        )

        def custom_update(batch, sample, iteration):
            batch["prev_state"] = batch["state"].clone()
            batch["state"] = sample
            batch["timestamp"] = batch["timestamp"] + batch["lead_time_hours"] * 3600
            batch["forcing"] = batch["forcing"] + 0.05  # Arbitrary modification to forcing
            return batch

        rollout = module.sample_rollout(batch, iterations=2, update_fnc=custom_update)

        assert len(rollout) == 2
        for state in rollout:
            assert not any(torch.isnan(v).any() for v in state.values()), "Output contains NaNs"
