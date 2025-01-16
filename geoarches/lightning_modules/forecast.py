import importlib
import importlib.resources
from pathlib import Path

import diffusers
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.checkpoint as gradient_checkpoint
from hydra.utils import instantiate
from tensordict.tensordict import TensorDict

from geoarches.dataloaders import era5, zarr
from geoarches.metrics.deterministic_metrics import Era5DeterministicMetrics
from geoarches.metrics.metric_base import compute_lat_weights, compute_lat_weights_weatherbench

from .. import stats as geoarches_stats
from .base_module import BaseLightningModule

geoarches_stats_path = importlib.resources.files(geoarches_stats)


class ForecastModule(BaseLightningModule):
    def __init__(
        self,
        cfg,  # instead of backbone
        name="forecast",
        dataset=None,
        pow=2,  # 2 is standard mse
        loss_delta_normalization=True,
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-5,
        num_warmup_steps=1000,
        num_training_steps=300000,
        num_cycles=0.5,
        use_graphcast_coeffs=True,
        increase_multistep_period=2,
        add_input_state=False,
        save_test_outputs=False,
        use_weatherbench_lat_coeffs=True,
        rollout_iterations=1,
        test_filename_suffix="",
        **kwargs,
    ):
        """should create self.encoder and self.decoder in subclasses"""
        super().__init__()
        # self.save_hyperparameters()
        self.__dict__.update(locals())
        self.backbone = instantiate(cfg.backbone)  # necessary to put it on device
        self.embedder = instantiate(cfg.embedder)

        # define coeffs for loss

        compute_weights_fn = (
            compute_lat_weights_weatherbench
            if use_weatherbench_lat_coeffs
            else compute_lat_weights
        )
        area_weights = compute_weights_fn(121)

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

        if loss_delta_normalization:
            # assumes include vertical wind component

            pangu_stats = torch.load(
                geoarches_stats_path / "pangu_norm_stats2_with_w.pt", weights_only=True
            )

            # mul by first to remove norm, div by second to apply fake delta normalization
            self.loss_delta_scaler = TensorDict(
                level=pangu_stats["level_std"]
                / torch.tensor(
                    [5.9786e02, 7.4878e00, 8.9492e00, 2.7132e00, 9.5222e-04, 0.3]
                ).reshape(-1, 1, 1, 1),
                surface=pangu_stats["surface_std"]
                / torch.tensor([3.8920, 4.5422, 2.0727, 584.0980]).reshape(-1, 1, 1, 1),
            )
            self.loss_coeffs = self.loss_coeffs * self.loss_delta_scaler.pow(self.pow)

        self.metrics = Era5DeterministicMetrics(
            compute_lat_weights_fn=compute_lat_weights_weatherbench
            if use_weatherbench_lat_coeffs
            else compute_lat_weights,
            lead_time_hours=24,
            rollout_iterations=rollout_iterations,
        )

        # self.test_outputs = []

    def forward(self, batch, *args, **kwargs):
        x = self.embedder.encode(batch["state"], batch.get("prev_state", None))

        x = self.backbone(x, *args, **kwargs)
        out = self.embedder.decode(x)  # we get tdict

        if self.add_input_state:
            out += batch["state"]

        return out

    def forward_multistep(self, batch, iters=None, return_format="tensordict", use_avg=True):
        # multistep forward with gradient checkpointing to save GPU memory
        if use_avg and self.avg_modules is not None:
            out = self.forward_multistep(batch, iters=iters, use_avg=False)
            for m in self.avg_modules:
                out = out + m.forward_multistep(batch, iters=iters, use_avg=False)
            return out / (1 + len(self.avg_modules))

        preds_future = []
        loop_batch = {k: v for k, v in batch.items()}
        for _ in range(iters):
            if torch.is_grad_enabled():
                pred = gradient_checkpoint.checkpoint(
                    self.forward, loop_batch, use_reentrant=False
                )
            else:
                pred = self.forward(loop_batch)
            preds_future.append(pred)
            # compute next batch
            loop_batch = dict(
                prev_state=loop_batch["state"],
                state=pred,
                timestamp=loop_batch["timestamp"] + batch["lead_time_hours"] * 3600,
            )

        if return_format == "list":
            return preds_future
        preds_future = torch.stack(preds_future, dim=1)
        return preds_future

    def loss(self, pred, gt, multistep=False, **kwargs):
        loss_coeffs = self.loss_coeffs.to(self.device)

        if multistep:  # means we have to compute multistep loss
            # discount for multistep loss
            lead_iter = next(iter(gt.values())).shape[1]
            future_coeffs = (
                torch.tensor([1 / (1 + i) ** 2 for i in range(lead_iter)])
                .to(self.device)
                .reshape(-1, 1, 1, 1, 1)
            )

            loss_coeffs.apply(lambda x: x * future_coeffs)

        weighted_error = (pred - gt).abs().pow(self.pow).mul(loss_coeffs)

        loss = sum(weighted_error.mean().values())

        return loss

    def training_step(self, batch, batch_nb):
        denormalize = self.trainer.train_dataloader.dataset.denormalize
        self.metrics.reset()

        if "future_states" not in batch:
            # standard prediction
            pred = self.forward(batch)
            loss = self.loss(pred, batch["next_state"])
            self.mylog(loss=loss)

            self.metrics.update(
                denormalize(batch["next_state"])[:, None], denormalize(pred)[:, None]
            )
            outputs = self.metrics.compute()

            self.mylog(**outputs)

        else:
            # multistep prediction
            lead_iter = batch["future_states"].shape[1]
            pred_future_states = self.forward_multistep(batch, iters=lead_iter)
            loss = self.loss(pred_future_states, batch["future_states"], multistep=True)

            self.mylog(lead_iter=lead_iter)
            self.mylog(loss=loss)
            # metrics for next state
            self.metrics.update(
                denormalize(batch["future_states"][:, :1]), denormalize(pred_future_states[:, :1])
            )
            outputs = self.metrics.compute()
            self.mylog(**outputs)

        return loss

    def on_validation_epoch_start(self):
        self.metrics.reset()

    def validation_step(self, batch, batch_nb):
        dataset = self.trainer.val_dataloaders.dataset
        pred = self.forward(batch)
        loss = self.loss(pred, batch["next_state"])
        self.mylog(loss=loss)

        self.metrics.update(
            dataset.denormalize(batch["next_state"])[:, None], dataset.denormalize(pred)[:, None]
        )

        return loss

    def on_validation_epoch_end(self):
        outputs = self.metrics.compute()
        self.mylog(**outputs, mode="val_")
        self.metrics.reset()

    def on_test_epoch_start(self):
        dataset = self.trainer.test_dataloaders.dataset
        self.metrics.reset()
        Path("evalstore").joinpath(self.name).mkdir(exist_ok=True, parents=True)
        self.test_filename = (
            Path("evalstore")
            / self.name
            / f"{dataset.domain}{self.test_filename_suffix}_metrics.pt"
        )
        if self.save_test_outputs:
            self.zarr_writer = zarr.ZarrIterativeWriter(
                self.test_filename.parent / f"{dataset.domain}.zarr"
            )

    def test_step(self, batch, batch_nb):
        # are we doing multistep ?
        dataset = self.trainer.test_dataloaders.dataset
        step_iterations = dataset.multistep
        preds_future = self.forward_multistep(batch, iters=step_iterations)

        # compute metrics
        if "future_states" in batch:
            ref_state = batch["future_states"]
        else:
            ref_state = batch["next_state"][:, None]
        self.metrics.update(
            dataset.denormalize(ref_state),
            dataset.denormalize(preds_future),
        )

        if self.save_test_outputs:
            xr_dataset = dataset.convert_trajectory_to_xarray(
                preds_future,
                timestamp=batch["timestamp"],
                denormalize=True,
                levels=[300, 500, 700, 850],
            )
            self.zarr_writer.write(xr_dataset, append_dim="time")

        if hasattr(self, "zarr_writer") and not (batch_nb + 1) % 25:
            self.zarr_writer.to_netcdf(dump_id=batch_nb)

    def on_test_epoch_end(self):
        outputs = self.metrics.compute()
        torch.save(outputs, self.test_filename)

        if self.save_test_outputs:
            self.zarr_writer.to_netcdf(dump_id="final")

        self.metrics.reset()
        return outputs

    def on_train_epoch_start(self, *args, **kwargs):
        dataset = self.trainer.train_dataloader.dataset
        if dataset.multistep > 1:
            # increase multistep every 2 epochs
            dataset.multistep = 2 + self.current_epoch // self.increase_multistep_period

    def on_train_epoch_end(self, *args, **kwargs):
        dataset = self.trainer.train_dataloader.dataset
        dataset.iteration_hook(self)

    def configure_optimizers(self):
        decay_params = {
            k: True for k, v in self.named_parameters() if "weight" in k and "norm" not in k
        }
        opt = torch.optim.AdamW(
            [
                {"params": [v for k, v in self.named_parameters() if k in decay_params]},
                {
                    "params": [v for k, v in self.named_parameters() if k not in decay_params],
                    "weight_decay": 0,
                },
            ],
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


class ForecastModuleWithCond(ForecastModule):
    """
    module that can take additional information:
    - month and hour
    - previous state
    - pred state (e.g. prediction of other weather model)
    """

    def __init__(
        self,
        *args,
        cond_dim=32,
        use_prev=False,
        use_avg=False,
        avg_with_modules=[],
        **kwargs,
    ):
        from geoarches.backbones import dit

        super().__init__(*args, **kwargs)
        # cond_dim should be given as arg to the backbone
        self.month_embedder = dit.TimestepEmbedder(cond_dim)
        self.hour_embedder = dit.TimestepEmbedder(cond_dim)
        self.use_prev = use_prev
        self.use_avg = use_avg

        self.avg_modules = None
        if avg_with_modules:
            from geoarches.lightning_modules.base_module import load_module

            self.avg_modules = nn.ModuleList(
                [load_module(m, return_config=False) for m in avg_with_modules]
            )
            self.strict_loading = False

    def forward(self, batch, use_avg=True):
        device = batch["state"].device
        # convert time into str

        times = pd.to_datetime(batch["timestamp"].cpu().numpy(), unit="s").tz_localize(None)
        month = torch.tensor(times.month).to(device)
        month_emb = self.month_embedder(month)
        hour = torch.tensor(times.hour).to(device)
        hour_emb = self.hour_embedder(hour)

        cond_emb = month_emb + hour_emb

        return super().forward(batch, cond_emb)
