import importlib
from copy import deepcopy
from pathlib import Path

import diffusers
import pandas as pd
import torch
from diffusers.training_utils import compute_snr
from hydra.utils import instantiate
from tensordict.tensordict import TensorDict
from tqdm import tqdm

import geoarches.stats as geoarches_stats
from geoarches.backbones.dit import TimestepEmbedder
from geoarches.dataloaders import era5, zarr
from geoarches.lightning_modules import BaseLightningModule
from geoarches.utils.tensordict_utils import tensordict_apply, tensordict_cat

geoarches_stats_path = importlib.resources.files(geoarches_stats)

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from geoarches.metrics.ensemble_metrics import Era5EnsembleMetrics


class DiffusionModule(BaseLightningModule):
    """
    this module is for learning the diffusion stuff
    """

    def __init__(
        self,
        cfg,
        name="diffusion",
        add_input_state="default",
        cond_dim=32,
        num_train_timesteps=1000,
        scheduler="ddpm",
        prediction_type="v_prediction",
        beta_schedule="squaredcos_cap_v2",
        beta_start=0.0001,
        beta_end=0.012,
        loss_weighting_strategy=None,
        snr_gamma=None,
        conditional=False,
        uncond_proba=0.0,
        ckpt_path=None,  # do we still need this ?
        ft_lr=1e-5,  # and this ?
        selected_vars=False,
        num_inference_steps=50, #now in cfg.inference.num_steps
        cf_guidance=1, #now in cfg.inference
        num_members=4,# now in cfg.inference
        load_deterministic_model=False,
        loss_delta_normalization=False,
        state_normalization=False,
        use_graphcast_coeffs=True,
        pow=2,
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-5,
        num_warmup_steps=1000,
        num_training_steps=300000,
        num_cycles=0.5,
        learn_residual=False,
        use_fm_sigma_scaling=False,
        **kwargs,
        # save_test_outputs=False, now in cfg.inference
    ):
        """
        loss_delta_normalization: str, either delta or pred.
        add_input_state: either "default" or "pred"
        """
        super().__init__()
        self.__dict__.update(locals())

        # check config
        if loss_delta_normalization:
            assert (
                prediction_type == "sample"
            ), "only sample prediction type is supported when scaling loss"

        self.cfg = cfg
        self.backbone = instantiate(cfg.backbone)  # necessary to put it on device
        self.embedder = instantiate(cfg.embedder)

        if load_deterministic_model:
            # TODO: confirm that it works
            from geoarches.lightning_modules import load_module

            self.det_model, _ = load_module(load_deterministic_model)

        # cond_dim should be given as arg to the backbone

        self.month_embedder = TimestepEmbedder(cond_dim)
        self.hour_embedder = TimestepEmbedder(cond_dim)
        self.timestep_embedder = TimestepEmbedder(cond_dim)
        if scheduler in ("ddpm", "ddim"):
            _class = diffusers.DDPMScheduler if scheduler == "ddpm" else diffusers.DDIMScheduler
            self.noise_scheduler = _class(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                prediction_type=prediction_type,
                clip_sample=False,
                clip_sample_range=1e6,
                rescale_betas_zero_snr=True,
            )

        elif scheduler == "heun":
            self.noise_scheduler = diffusers.HeunDiscreteScheduler(
                num_train_timesteps=num_train_timesteps
            )

        elif scheduler == "flow":
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps
            )

        self.inference_scheduler = deepcopy(self.noise_scheduler)

        if ckpt_path is not None:
            print(next(self.backbone.named_parameters()))
            print("init from ckpt", ckpt_path)
            self.init_from_ckpt(ckpt_path)
            print(next(self.backbone.named_parameters()))

        area_weights = torch.arange(-90, 90 + 1e-6, 1.5).mul(torch.pi / 180).cos()
        area_weights = (area_weights / area_weights.mean())[:, None]

        # set up metrics
        self.val_ensemble_metrics = Era5EnsembleMetrics()  # only one timestep

        self.val_metrics = [self.val_ensemble_metrics]
        # self.test_metrics = [self.test_ensemble_metrics]
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

        if loss_delta_normalization in ("delta", True) or state_normalization == "delta":
            scaler = TensorDict(
                level=torch.tensor(
                    [5.9786e02, 7.4878e00, 8.9492e00, 2.7132e00, 9.5222e-04, 0.3]
                ).reshape(-1, 1, 1, 1),
                surface=torch.tensor([3.8920, 4.5422, 2.0727, 584.0980]).reshape(-1, 1, 1, 1),
            )

        elif loss_delta_normalization == "pred" or state_normalization == "pred":
            scaler = TensorDict(**torch.load("stats/deltapred_24h_stds.pt", weights_only=False))
            # even if it's not the exact same model, we dont care too much
            # humidity causes problems at high altitude, let's just average it
            scaler["level"] = scaler["level"].mean(dim=-3, keepdim=True)

        if loss_delta_normalization:
            self.loss_coeffs = tensordict_apply(
                torch.mul, self.loss_coeffs, (pangu_scaler / scaler).pow(self.pow)
            )
        elif state_normalization:
            self.state_scaler = scaler / pangu_scaler  # inverse because we divide by state_scaler

        self.test_outputs = []
        self.validation_samples = {}

    def forward(self, batch, noisy_next_state, timesteps, use_condition=True, is_sampling=False):
        device = batch["state"].device
        bs = batch["state"].shape[0]
        if type(use_condition) is bool:
            sel = torch.tensor(int(use_condition)).to(device)
        else:
            sel = use_condition.to(device)

        input_state = noisy_next_state
        conditional_keys = self.conditional.split("+")  # all the things we condition on

        # whether we need to run the deterministic model
        if "det" in conditional_keys:
            assert "pred_state" in batch
            pred_state = batch["pred_state"]

        if "prev" in conditional_keys:
            prev_state = batch["prev_state"].apply(lambda x: x * sel)
            input_state = tensordict_cat([prev_state, input_state], dim=1)
        if "det" in conditional_keys:
            pred_state = pred_state.apply(lambda x: x * sel)
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

        if is_sampling and self.prediction_type == "sample" and self.scheduler == "flow":
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

        if self.state_normalization:
            next_state = tensordict_apply(torch.div, next_state, self.state_scaler.to(self.device))

        # if self.flow_matching:
        if self.scheduler == "flow":
            # weighting scheme: logit normal
            u = torch.normal(mean=0, std=1, size=(bs,), device="cpu").sigmoid()

            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
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
            target_state = noise - next_state
            if self.prediction_type == "sample":
                target_state = next_state

        else:
            noisy_next_state = tensordict_apply(
                self.noise_scheduler.add_noise, next_state, noise, timesteps
            ).to(device)

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target_state = noise

            elif self.noise_scheduler.config.prediction_type == "sample":
                target_state = next_state

            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target_state = tensordict_apply(
                    self.noise_scheduler.get_velocity,
                    next_state,
                    noise,
                    timesteps,
                )

            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

        # create uncond
        use_condition = torch.rand((bs,), device=device) > self.uncond_proba
        pred = self.forward(
            batch,
            noisy_next_state,
            timesteps,
            use_condition.float()[:, None, None, None, None],
        )

        # compute loss

        loss = self.loss(pred, target_state, timesteps)

        self.mylog(loss=loss)

        return loss

    def loss(self, pred, gt, timesteps=None, **kwargs):
        loss_coeffs = self.loss_coeffs.to(self.device)
        if self.loss_weighting_strategy is not None and timesteps is not None:
            if self.scheduler == "flow":
                sigmas = timesteps / self.noise_scheduler.config.num_train_timesteps
                snr = ((1 - sigmas) / sigmas).pow(2)
            else:
                snr = compute_snr(self.noise_scheduler, timesteps)

            # reference weights below are for the "sample" prediction type
            if self.loss_weighting_strategy == "sample":
                snr_weights = torch.tensor(1)
            elif self.loss_weighting_strategy == "epsilon":
                snr_weights = snr
            elif self.loss_weighting_strategy == "debiased":  # or geometric_average
                snr_weights = snr.sqrt()
            elif self.loss_weighting_strategy == "arithmetic_average":
                snr_weights = (1 + snr) / 2

            elif self.loss_weighting_strategy == "minsnrgamma":
                snr_weights = torch.stack(
                    [snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]

            if self.prediction_type == "sample":
                pass  # snr_weights are defined for "sample" coeffs by default
            elif self.prediction_type == "epsilon":
                snr_weights = snr_weights / snr
            elif self.prediction_type == "v_prediction":
                snr_weights = snr_weights + 1

            snr_weights = snr_weights.to(self.device)[:, None, None, None, None]
            loss_coeffs = loss_coeffs.apply(lambda x: x * snr_weights)

        # for some reason there is a scaling of the loss in flow matching
        # find where it is already, I dont think this is relevant
        # if it does not give good results we will activate it
        if self.scheduler == "flow" and self.use_fm_sigma_scaling:
            sigma = timesteps / self.noise_scheduler.config.num_train_timesteps
            loss_coeffs = loss_coeffs * sigma.pow(2).to(self.device)[:, None, None, None, None]

        weighted_error = (pred - gt).abs().pow(self.pow).mul(loss_coeffs)
        loss = sum(weighted_error.mean().values())
        return loss

    def sample(
        self,
        batch,
        seed=None,
        num_steps=None,
        cf_guidance=None,
        disable_tqdm=False,
        rescale=1.0,
        **kwargs,
    ):
        """
        kwargs args are fed to scheduler_step
        """

        if cf_guidance is None:
            cf_guidance = int(self.cf_guidance)

        scheduler = self.inference_scheduler
        num_steps = num_steps or self.num_inference_steps
        scheduler.set_timesteps(num_steps)

        if seed is not None:
            torch.manual_seed(seed)

        # get args to scheduler
        if self.scheduler == "flow":
           scheduler_kwargs = dict(s_churn=self.s_churn)
            #scheduler_kwargs = dict()  # dict(eta=self.cfg.inference.eta)

        else:
            # TODO: put some churning here
            scheduler_kwargs = dict()  # dict(eta=self.cfg.inference.eta)

        # scheduler_kwargs.update(kwargs)


        noisy_state = batch["state"].apply(torch.randn_like)

        scale_input_noise = self.scale_input_noise
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
                    use_condition=(cf_guidance > 0),
                    is_sampling=True,
                )
                if cf_guidance > 1:
                    uncond_pred = self.forward(
                        loop_batch,
                        noisy_state,
                        timesteps=torch.tensor([t]).to(self.device),
                        use_condition=0,
                        is_sampling=True,
                    )
                    # compute epsilon from uncond_pred
                    pred = pred + cf_guidance * (pred - uncond_pred)

                # due to weird behavior of scheduler we need to use the following
                step_index = getattr(scheduler, "_step_index", None)

                def scheduler_step(*args, **kwargs):
                    out = scheduler.step(*args, **kwargs)
                    if hasattr(scheduler, "_step_index"):
                        scheduler._step_index = step_index
                    return out.prev_sample


                noisy_state = tensordict_apply(scheduler_step, pred, t, noisy_state, **scheduler_kwargs)
                # at the end
                if step_index is not None:
                    scheduler._step_index = step_index + 1


        final_state = noisy_state.detach()

        if self.state_normalization:
            final_state = tensordict_apply(
                torch.mul, final_state, self.state_scaler.to(self.device)
            )

        if self.learn_residual == "default":
            final_state = batch["state"] + final_state * rescale

        elif self.learn_residual == "pred":
            final_state = batch["pred_state"] + final_state * rescale

        return final_state

    def sample_rollout(
        self,
        batch,
        *args,
        iterations=1,
        disable_tqdm=False,
        return_format="tensordict",
        **kwargs,
    ):
        torch.set_grad_enabled(False)
        preds_future = []
        loop_batch = {k: v for k, v in batch.items()}

        for i in tqdm(range(iterations), disable=disable_tqdm):
            sample = self.sample(loop_batch, *args, **kwargs, disable_tqdm=True)
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
        samples = [
            self.sample(batch, seed=j, disable_tqdm=True) for j in tqdm(range(val_num_members))
        ]
        denormalize = self.trainer.val_dataloaders.dataset.denormalize

        for metric in self.val_metrics:
            metric.update(
                denormalize(batch["next_state"]),
                [denormalize(sample) for sample in samples],
            )
        self.validation_samples[batch_nb] = samples

    def on_validation_epoch_end(self):
        for metric in self.val_metrics:
            scores = metric.compute()
            self.log_dict(scores, sync_dist=True)  # dont put on_epoch = True here
            print(scores)
            metric.reset()
        self.validation_samples.clear()

    def on_test_epoch_start(self):
        self.test_outputs = []
        if not self.cfg.inference.save_test_outputs:
            return
        suffix = getattr(self.cfg.inference, "test_filename_suffix", "")
        self.test_filename = f"test2020-num_steps={self.cfg.inference.num_steps}-members={self.cfg.inference.num_members}-{suffix}.zarr"
        self.zarr_writer = zarr.ZarrIterativeWriter(
            Path("evalstore") / self.name / self.test_filename
        )

    def test_step(self, batch, batch_nb):
        dataset = self.trainer.test_dataloaders.dataset

        # if self.cfg.inference.limit_test_samples > 0 and batch_nb not in self.test_batch_nb_set:
        #    return None

        samples = [
            self.sample_rollout(
                batch, iterations=self.cfg.inference.rollout_iterations, seed=j, disable_tqdm=True
            )
            for j in range(self.cfg.inference.num_members)
        ]

        if self.cfg.inference.save_test_outputs:
            import xarray as xr

            xr_dataset_list = [
                dataset.convert_trajectory_to_xarray(
                    sample, timestamp=batch["timestamp"], denormalize=True, compressed=True
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
        for metric in self.test_metrics:
            metric.update(
                dataset.denormalize(batch["future_states"]),  # TODO: do eval with future states
                [dataset.denormalize(sample) for sample in samples],
            )

        # log 24h metrics
        return None

    def on_test_epoch_end(self):
        # save results
        all_metrics = {}
        for metric in self.test_metrics:
            scores = metric.compute()

            self.log_dict(scores, sync_dist=True)  # dont put on_epoch = True here
            all_metrics.update(scores)
            metric.reset()

        fname = self.test_filename.replace(".zarr", "_metrics.pt")
        torch.save(all_metrics, Path("evalstore") / self.name / fname)
        self.zarr_writer.to_netcdf()
        # now each entry is a tensor that has length multistep

    def configure_optimizers(self):
        print("configure optimizers")
        if self.ckpt_path is not None:
            opt = torch.optim.AdamW(
                [
                    {
                        "params": self.backbone.parameters(),
                        "lr": self.ft_lr,
                    },  # finetune
                    {"params": self.noisy_branch.parameters()},
                    {"params": self.month_embedder.parameters()},
                    {"params": self.hour_embedder.parameters()},
                    {"params": self.timestep_embedder.parameters()},
                ],
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        else:
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
