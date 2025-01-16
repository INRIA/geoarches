# this combines multiple diffusion models in various ways
import random

import lightning as L  # noqa N812
import torch
import torch.nn as nn
from tqdm import tqdm

from geoarches.lightning_modules.base_module import load_module
from geoarches.metrics.brier_skill_score import Era5BrierSkillScore
from geoarches.metrics.ensemble_metrics import Era5EnsembleMetrics

from .diffusion import DiffusionModule

torch.set_grad_enabled(False)


class EnsembleDiffusionModule(DiffusionModule, L.LightningModule):
    def __init__(
        self,
        cfg,
        name="diffusion_ensemble",
        model_uids=None,
        model_kwargs={},
        deterministic_model_uids=None,
        generative_mode="separate",
        deterministic_mode="separate",
    ):
        """
        This is a class for sampling with different diffusion models
        mode: how diffusion models are combined.
        separate: make trajectories with each model, separately.
        mix: at each step, sample with a different model, randomly
        """
        L.LightningModule.__init__(self)

        self.cfg = cfg
        self.name = name
        self.generative_mode = generative_mode
        self.deterministic_mode = deterministic_mode

        self.models = nn.ModuleList(
            [
                load_module(
                    p,
                    return_config=False,
                    **(dict(load_deterministic_model=False) if p != model_uids[0] else {}),
                    **model_kwargs,
                )
                for p in model_uids
            ]
        )
        for mod in self.models:
            mod.cfg.inference.update(self.cfg.inference)

        self.det_models = None
        if deterministic_model_uids is not None:
            self.det_models = nn.ModuleList(
                [load_module(p, return_config=False) for p in deterministic_model_uids]
            )

        else:
            # unique deterministic model, otherwise blow memory
            self.det_models = nn.ModuleList([self.models[0].det_model])
            for model in self.models:
                del model.det_model

        # init test metrics
        test_kwargs = dict(lead_time_hours=24, rollout_iterations=cfg.inference.rollout_iterations)

        self.test_metrics = nn.ModuleList(
            [Era5EnsembleMetrics(**test_kwargs), Era5BrierSkillScore(**test_kwargs)]
        )

    def sample(self, batch, seed, member=0, *args, **kwargs):
        # if model_idx is None, sample with a different model at each step

        if self.deterministic_mode == "mix":
            det_model_idx = random.randint(0, len(self.det_models) - 1)

        elif self.deterministic_mode == "separate":
            det_model_idx = member % len(self.det_models)

        if self.generative_mode == "mix":
            gen_model_idx = random.randint(0, len(self.models) - 1)
        elif self.generative_mode == "separate":
            gen_model_idx = member % len(self.models)

        with torch.no_grad():
            batch["pred_state"] = self.det_models[det_model_idx](batch).detach()
            seed = seed + gen_model_idx * 10**9
            sample = self.models[gen_model_idx].sample(batch, seed, *args, **kwargs)

        return sample

    def sample_rollout(
        self,
        batch,
        batch_nb,
        iterations=1,
        member=0,
        disable_tqdm=True,
        return_format="tensordict",
        *args,
        **kwargs,
    ):
        torch.set_grad_enabled(False)
        preds_future = []
        loop_batch = {k: v for k, v in batch.items()}

        for i in tqdm(range(iterations), disable=disable_tqdm):
            seed_i = member + 1000 * i + batch_nb * 10**6
            sample = self.sample(loop_batch, seed=seed_i, member=member, disable_tqdm=True)
            preds_future.append(sample)
            loop_batch = dict(
                prev_state=loop_batch["state"],
                state=sample,
                timestamp=loop_batch["timestamp"] + batch["lead_time_hours"] * 3600,
            )

        if return_format == "list":
            return preds_future
        preds_future = torch.stack(preds_future, dim=1)

        return preds_future
