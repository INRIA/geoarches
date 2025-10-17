import torch
from hydra import compose, initialize
from tensordict.tensordict import TensorDict

from geoarches.lightning_modules.diffusion import DiffusionModule
from geoarches.lightning_modules.forecast import ForecastModuleWithCond

with initialize(version_base="1.2", config_path="../../geoarches/configs"):
    cfg = compose(
        config_name="test_awgen",
        overrides=[
            "module.module.load_deterministic_model=Null",
            "module.embedder.forcings_ch=2",
            "module.embedder.forcings_embedding=surface",
        ],
    )

    det_cfg = compose(
        config_name="test_awdet",
        overrides=["module.embedder.forcings_ch=2", "module.embedder.forcings_embedding=surface"],
    )


class TestDiffusionModule:
    def build_model(self, build_det_model=False):
        module = DiffusionModule(cfg.module, cfg.stats, **cfg.module.module)

        if build_det_model:
            det_module = ForecastModuleWithCond(
                det_cfg.module, det_cfg.stats, **det_cfg.module.module
            )

            return module, det_module

        else:
            return module

    def get_dims(self):
        surface_ch = cfg.module.embedder.surface_ch
        level_ch = cfg.module.embedder.level_ch
        forc_ch = cfg.module.embedder.forcings_ch
        img_size = cfg.module.embedder.img_size
        return surface_ch, level_ch, forc_ch, img_size

    def test_forward_with_forcing(self):
        module = self.build_model(build_det_model=False)
        surf_ch, lvl_ch, forc_ch, img_size = self.get_dims()

        batch = TensorDict(
            {
                "prev_state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, 1, *img_size[-2:]),
                },
                "state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, *img_size[-2:]),
                },
                "next_state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, 1, *img_size[-2:]),
                },
                "pred_state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, 1, *img_size[-2:]),
                },
                "forcings": torch.randn(1, forc_ch, *img_size[-2:]),
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
        module = self.build_model()
        surf_ch, lvl_ch, _, img_size = self.get_dims()

        # Create pred
        pred = TensorDict(
            {
                "level": torch.zeros(1, lvl_ch, *img_size),
                "surface": torch.zeros(1, surf_ch, 1, *img_size[-2:]),
            },
            batch_size=[],
        )

        # Create gt with NaNs.
        gt = TensorDict(
            {
                "level": torch.randn(1, lvl_ch, *img_size),
                "surface": torch.randn(1, surf_ch, 1, *img_size[-2:]),
            },
            batch_size=[],
        )
        lat_half, lon_half = img_size[-2] // 2, img_size[-1] // 2
        gt["level"][0, 0, lat_half - 10 : lat_half, lon_half - 10 : lon_half] = torch.nan
        gt["surface"][0, 0, lat_half - 10 : lat_half, lon_half - 10 : lon_half] = torch.nan

        loss = module.loss(pred, gt, timesteps=torch.tensor([0.5], dtype=torch.float32))

        assert not torch.isnan(loss)

    def test_sample_rollout_w_forcings(self):
        module, det_module = self.build_model(build_det_model=True)
        surf_ch, lvl_ch, forc_ch, img_size = self.get_dims()
        module.det_model = det_module

        batch = TensorDict(
            {
                "prev_state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, 1, *img_size[-2:]),
                },
                "state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, 1, *img_size[-2:]),
                },
                "forcings": torch.randn(1, forc_ch, *img_size[-2:]),
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

        rollout = module.sample_rollout(
            batch,
            iterations=2,
        )

        for state in rollout:
            assert not any(torch.isnan(v).any() for v in state.values()), "Output contains NaNs"

    def test_sample_rollout_w_forcings_custom_update(self):
        module, det_module = self.build_model(build_det_model=True)
        surf_ch, lvl_ch, forc_ch, img_size = self.get_dims()
        module.det_model = det_module  # Manually set the deterministic model

        batch = TensorDict(
            {
                "prev_state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, 1, *img_size[-2:]),
                },
                "state": {
                    "level": torch.randn(1, lvl_ch, *img_size),
                    "surface": torch.randn(1, surf_ch, 1, *img_size[-2:]),
                },
                "forcings": torch.randn(1, forc_ch, *img_size[-2:]),
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
            batch["forcings"] = batch["forcings"] + 0.05  # Arbitrary modification to forcing
            return batch

        rollout = module.sample_rollout(batch, iterations=2, update_fnc=custom_update)

        for state in rollout:
            assert not any(torch.isnan(v).any() for v in state.values()), "Output contains NaNs"
