import importlib
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import diffusers
import pandas as pd
import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from hydra.utils import instantiate
from tensordict.tensordict import TensorDict
from tqdm import tqdm

import geoarches.stats as geoarches_stats
from geoarches.backbones.dit import TimestepEmbedder
from geoarches.dataloaders import era5, zarr
from geoarches.lightning_modules import BaseLightningModule
from geoarches.metrics.brier_skill_score import Era5BrierSkillScore
from geoarches.metrics.ensemble_metrics import Era5EnsembleMetrics
from geoarches.utils.tensordict_utils import tensordict_apply, tensordict_cat

geoarches_stats_path = importlib.resources.files(geoarches_stats)


class DiffusionModule(BaseLightningModule):
    """
    this module is for learning the diffusion stuff
    """

    def __init__(
        self,
        cfg,
        name="diffusion",
        cond_dim=32,
        num_train_timesteps=1000,
        scheduler="flow",  # only available option
        prediction_type="sample",  # or velocity
        beta_schedule="squaredcos_cap_v2",
        beta_start=0.0001,
        beta_end=0.012,
        loss_weighting_strategy=None,
        conditional="",  # things that the model is conditioned
        load_deterministic_model=False,
        loss_delta_normalization=False,
        state_normalization=False,
        pow=2,
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-5,
        num_warmup_steps=1000,
        num_training_steps=300000,
        num_cycles=0.5,
        learn_residual=False,
        sd3_timestep_sampling=True,
        **kwargs,
    ):
        """
        loss_delta_normalization: str, either delta or pred.
        """
        super().__init__()
        self.__dict__.update(locals())

        self.cfg = cfg
        self.backbone = instantiate(cfg.backbone)  # necessary to put it on device
        self.embedder = instantiate(cfg.embedder)

        self.det_model = None
        if load_deterministic_model:
            # TODO: confirm that it works
            from geoarches.lightning_modules.base_module import AvgModule, load_module

            if isinstance(load_deterministic_model, str):
                self.det_model, _ = load_module(load_deterministic_model)
            else:
                # load averaging model
                self.det_model = AvgModule(load_deterministic_model)

        # cond_dim should be given as arg to the backbone

        self.month_embedder = TimestepEmbedder(cond_dim)
        self.hour_embedder = TimestepEmbedder(cond_dim)
        self.timestep_embedder = TimestepEmbedder(cond_dim)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps
        )

        self.inference_scheduler = deepcopy(self.noise_scheduler)

        area_weights = torch.arange(-90, 90 + 1e-6, 1.5).mul(torch.pi / 180).cos()
        area_weights = (area_weights / area_weights.mean())[:, None]

        # set up metrics
        save_memory = cfg.inference.num_members > 10
        val_kwargs = dict(lead_time_hours=24, rollout_iterations=1, save_memory=save_memory)
        test_kwargs = dict(
            lead_time_hours=24,
            rollout_iterations=cfg.inference.rollout_iterations,
            save_memory=save_memory,
        )

        self.val_metrics = nn.ModuleList(
            [Era5EnsembleMetrics(**val_kwargs), Era5BrierSkillScore(**val_kwargs)]
        )
        self.test_metrics = nn.ModuleList(
            [Era5EnsembleMetrics(**test_kwargs), Era5BrierSkillScore(**test_kwargs)]
        )
        # define coeffs for loss

        pressure_levels = torch.tensor(era5.pressure_levels).float()
        vertical_coeffs = (pressure_levels / pressure_levels.mean()).reshape(-1, 1, 1)

        # define relative surface and level weights
        total_coeff = 6 + 1.3
        surface_coeffs = 4 * torch.tensor([0.1, 0.1, 1.0, 0.1]).reshape(
            -1, 1, 1, 1
        )  # graphcast, mul 4 because we do a mean
        level_coeffs = 6 * torch.tensor(1).reshape(-1, 1, 1, 1)

        self.loss_coeffs = TensorDict(
            surface=area_weights * surface_coeffs / total_coeff,
            level=area_weights * level_coeffs * vertical_coeffs / total_coeff,
        )
        # scaling loss or states
        pangu_stats = torch.load(
            geoarches_stats_path / "pangu_norm_stats2_with_w.pt", weights_only=True
        )
        pangu_scaler = TensorDict(
            level=pangu_stats["level_std"], surface=pangu_stats["surface_std"]
        )

        if state_normalization == "delta":
            scaler = TensorDict(
                level=torch.tensor(
                    [5.9786e02, 7.4878e00, 8.9492e00, 2.7132e00, 9.5222e-04, 0.3]
                ).reshape(-1, 1, 1, 1),
                surface=torch.tensor([3.8920, 4.5422, 2.0727, 584.0980]).reshape(-1, 1, 1, 1),
            )

        elif state_normalization == "pred":
            scaler = TensorDict(
                **torch.load(
                    geoarches_stats_path / "deltapred24_aws_denorm.pt", weights_only=False
                )
            )
            scaler["level"][-1] *= 3  # we don't care too much about vertical velocity

            self.state_scaler = scaler / pangu_scaler  # inverse because we divide by state_scaler

        self.test_outputs = []
        self.validation_samples = {}

    def forward(self, batch, noisy_next_state, timesteps, is_sampling=False):
        device = batch["state"].device

        input_state = noisy_next_state
        conditional_keys = self.conditional.split("+")  # all the things we condition on

        # whether we need to run the deterministic model
        if "det" in conditional_keys:
            assert "pred_state" in batch
            pred_state = batch["pred_state"]

        if "prev" in conditional_keys:
            prev_state = batch["prev_state"]
            input_state = tensordict_cat([prev_state, input_state], dim=1)
        if "det" in conditional_keys:
            input_state = tensordict_cat([pred_state, input_state], dim=1)

        # conditional by default
        times = pd.to_datetime(batch["timestamp"].cpu().numpy() * 10**9).tz_localize(None)
        month = torch.tensor(times.month).to(device)
        month_emb = self.month_embedder(month)
        hour = torch.tensor(times.hour).to(device)
        hour_emb = self.hour_embedder(hour)
        timestep_emb = self.timestep_embedder(timesteps)

        cond_emb = month_emb + hour_emb + timestep_emb

        x = self.embedder.encode(batch["state"], input_state)

        x = self.backbone(x, cond_emb)
        out = self.embedder.decode(x)  # we get tdict

        if is_sampling and self.prediction_type == "sample":
            sigmas = timesteps / self.noise_scheduler.config.num_train_timesteps
            sigmas = sigmas[:, None, None, None, None]  # shape (bs,)

            # to transform model_output=sample to output=noise - sample
            out = (noisy_next_state - out).apply(lambda x: x / sigmas)

        return out

    def training_step(self, batch, batch_nb):
        # sample timesteps
        device, bs = batch["state"].device, batch["state"].shape[0]

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device
        ).long()

        noise = torch.randn_like(batch["next_state"])  # amazing it works

        # by default
        next_state = batch["next_state"]

        if self.learn_residual == "default":
            next_state = batch["next_state"] - batch["state"]

        elif self.learn_residual == "pred":
            if "pred_state" not in batch:
                with torch.no_grad():
                    batch["pred_state"] = self.det_model(batch).detach()
            next_state = batch["next_state"] - batch["pred_state"]

        # weighting scheme: logit normal
        if self.sd3_timestep_sampling:
            u = torch.normal(mean=0, std=1, size=(bs,), device="cpu").sigmoid()
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        else:
            indices = timesteps.cpu()

        timesteps = self.noise_scheduler.timesteps[indices].to(device)

        schedule_timesteps = self.noise_scheduler.timesteps.to(device)

        sigmas = self.noise_scheduler.sigmas.to(
            device=device, dtype=next(iter(batch["state"].values())).dtype
        )

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()[:, None, None, None, None]  # shape (bs,)

        noisy_next_state = noise.apply(lambda x: x * sigma) + next_state.apply(
            lambda x: x * (1.0 - sigma)
        )
        if self.prediction_type == "sample":
            target_state = next_state
        else:
            target_state = noise - next_state

        pred = self.forward(
            batch,
            noisy_next_state,
            timesteps,
        )
        # compute loss

        loss = self.loss(pred, target_state, timesteps)

        self.mylog(loss=loss)

        return loss

    def loss(self, pred, gt, timesteps=None, **kwargs):
        loss_coeffs = self.loss_coeffs.to(self.device)
        if self.prediction_type == "sample":
            # loss weighting strategy
            sigmas = timesteps / self.noise_scheduler.config.num_train_timesteps
            snr_weights = (1 - sigmas) / sigmas
            snr_weights = snr_weights.to(self.device)[:, None, None, None, None]
            loss_coeffs = loss_coeffs.apply(lambda x: x * snr_weights)

        weighted_error = (pred - gt).abs().pow(self.pow).mul(loss_coeffs)
        loss = sum(weighted_error.mean().values())
        return loss

    def sample(
        self,
        batch,
        seed=None,
        num_steps=None,
        disable_tqdm=False,
        scale_input_noise=None,
        **kwargs,
    ):
        """
        kwargs args are fed to scheduler_step
        """

        scheduler = self.inference_scheduler
        num_steps = num_steps or self.cfg.inference.num_steps
        scheduler.set_timesteps(num_steps)

        # get args to scheduler
        scheduler_kwargs = dict(s_churn=self.cfg.inference.s_churn)
        scheduler_kwargs.update(kwargs)

        generator = torch.Generator(device=self.device)

        if seed is not None:
            generator.manual_seed(seed)

        noisy_state = batch["state"].apply(
            lambda x: torch.empty_like(x).normal_(generator=generator)
        )

        scale_input_noise = scale_input_noise or getattr(
            self.cfg.inference, "scale_input_noise", None
        )
        if scale_input_noise is not None:
            noisy_state = noisy_state * scale_input_noise

        if self.learn_residual == "pred" and "pred_state" not in batch:
            with torch.no_grad():
                batch["pred_state"] = self.det_model(batch).detach()

        loop_batch = {k: v for k, v in batch.items() if "next" not in k}  # ensures no data leakage

        with torch.no_grad():
            for t in tqdm(scheduler.timesteps, disable=disable_tqdm):
                # 1. predict noise model_output
                pred = self.forward(
                    loop_batch,
                    noisy_state,
                    timesteps=torch.tensor([t]).to(self.device),
                    is_sampling=True,
                )

                # due to weird behavior of scheduler we need to use the following
                step_index = getattr(scheduler, "_step_index", None)

                def scheduler_step(*args, **kwargs):
                    out = scheduler.step(*args, **kwargs)
                    if hasattr(scheduler, "_step_index"):
                        scheduler._step_index = step_index
                    return out.prev_sample

                noisy_state = tensordict_apply(
                    scheduler_step, pred, t, noisy_state, **scheduler_kwargs
                )
                # at the end
                if step_index is not None:
                    scheduler._step_index = step_index + 1

        final_state = noisy_state.detach()

        if self.state_normalization:
            final_state = tensordict_apply(
                torch.mul, final_state, self.state_scaler.to(self.device)
            )

        if self.learn_residual == "default":
            final_state = batch["state"] + final_state

        elif self.learn_residual == "pred":
            final_state = batch["pred_state"] + final_state

        return final_state

    def sample_rollout(
        self,
        batch,
        batch_nb=0,
        member=0,
        iterations=1,
        disable_tqdm=False,
        return_format="tensordict",
        **kwargs,
    ):
        torch.set_grad_enabled(False)
        preds_future = []
        loop_batch = {k: v for k, v in batch.items()}

        for i in tqdm(range(iterations), disable=disable_tqdm):
            seed_i = member + 1000 * i + batch_nb * 10**6
            sample = self.sample(loop_batch, seed=seed_i, disable_tqdm=True, **kwargs)
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

    def validation_step(self, batch, batch_nb):
        # for the validation, we make some generations and log them
        val_num_members = 5
        val_rollout_iterations = 2
        samples = [
            self.sample_rollout(
                batch,
                batch_nb=batch_nb,
                iterations=val_rollout_iterations,
                member=j,
                disable_tqdm=True,
            )
            for j in tqdm(range(val_num_members))
        ]
        denormalize = self.trainer.val_dataloaders.dataset.denormalize

        for metric in self.val_metrics:
            metric.update(
                denormalize(batch["future_states"][:, :val_rollout_iterations]),
                [denormalize(sample) for sample in samples],
            )
        self.validation_samples[batch_nb] = [samples[0][:, 0], samples[1][:, 0]]

    def on_validation_epoch_end(self):
        for metric in self.val_metrics:
            scores = metric.compute()
            self.log_dict(scores, sync_dist=True)  # dont put on_epoch = True here
            print(scores)
            metric.reset()
        self.validation_samples.clear()

    def on_test_epoch_start(self):
        dataset = self.trainer.test_dataloaders.dataset
        self.test_outputs = []

        suffix = getattr(self.cfg.inference, "test_filename_suffix", "")
        now = datetime.today().strftime("%m%d%H%M")
        self.test_filename = f"{dataset.domain}-{now}-num_steps={self.cfg.inference.num_steps}-members={self.cfg.inference.num_members}-{suffix}.zarr"
        Path("evalstore").joinpath(self.name).mkdir(exist_ok=True, parents=True)
        if self.cfg.inference.save_test_outputs:
            self.zarr_writer = zarr.ZarrIterativeWriter(
                Path("evalstore") / self.name / self.test_filename
            )

    def test_step(self, batch, batch_nb):
        dataset = self.trainer.test_dataloaders.dataset

        samples = [
            self.sample_rollout(
                batch,
                batch_nb=batch_nb,
                iterations=self.cfg.inference.rollout_iterations,
                member=j,
                disable_tqdm=True,
            )
            for j in range(self.cfg.inference.num_members)
        ]

        if self.cfg.inference.save_test_outputs:
            import xarray as xr

            xr_dataset_list = [
                dataset.convert_trajectory_to_xarray(
                    sample,
                    timestamp=batch["timestamp"],
                    denormalize=True,
                    levels=[300, 500, 700, 850],
                )
                for sample in samples
            ]
            xr_dataset = xr.concat(
                xr_dataset_list,
                pd.Index(
                    list(range(self.cfg.inference.num_members)), name="number"
                ),  # weirdy the dimension is named "number"
            )
            self.zarr_writer.write(xr_dataset, append_dim="time")

        # compute metrics
        if self.cfg.inference.save_test_outputs != "without_metrics":
            for metric in self.test_metrics:
                metric.update(
                    dataset.denormalize(
                        batch["future_states"]
                    ),  # TODO: do eval with future states
                    [dataset.denormalize(sample) for sample in samples],
                )

        if hasattr(self, "zarr_writer") and not (batch_nb + 1) % 2:
            self.zarr_writer.to_netcdf(dump_id=batch_nb)
        # log 24h metrics
        return None

    def on_test_epoch_end(self):
        # save results
        if self.cfg.inference.save_test_outputs == "without_metrics":
            return

        all_metrics = {}
        for metric in self.test_metrics:
            scores = metric.compute()

            self.log_dict(scores, sync_dist=True)  # dont put on_epoch = True here
            all_metrics.update(scores)
            metric.reset()
        all_metrics["hparams"] = dict(self.cfg.inference)

        fname = self.test_filename.replace(".zarr", "_metrics.pt")
        torch.save(all_metrics, Path("evalstore") / self.name / fname)

        if hasattr(self, "zarr_writer"):
            self.zarr_writer.to_netcdf(dump_id="final")

    def configure_optimizers(self):
        print("configure optimizers")
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        sched = diffusers.optimization.get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=self.num_cycles,
        )
        sched = {
            "scheduler": sched,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }
        return [opt], [sched]
