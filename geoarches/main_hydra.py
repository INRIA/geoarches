"""Main script to run pipeline for training or inference (specify `mode` arg as "train" or "test").

Arguments are configured with hydra (which reads the `configs/` folder to compose the config).
You can change arguments either by modifying the config files or through commandline.

Example:
    python -m geoarches.main_hydra
    module=archesweather  # Uses module/archesweather.yaml
    dataloader=era5   # Uses dataloader/era5.yaml
    ++name=default_run  # Dir to save model checkpoints and name of Wandb run.
"""

import os
import signal
import warnings
from pathlib import Path

import hydra
import lightning as L  # noqa N812
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import TQDMProgressBar
from omegaconf import DictConfig, OmegaConf


def get_random_code():
    import random
    import string

    # generate random code that alternates letters and numbers
    chars = random.choices(string.ascii_lowercase, k=3)
    nums = random.choices(string.digits, k=3)
    return "".join([f"{chars}{num}" for char, num in zip(chars, nums)])


def collate_fn(lst):
    return {k: torch.stack([x[k] for x in lst]) for k in lst[0]}


def select_checkpoint(path: str | Path, ckpt_filename_match: str | None = None) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint source does not exist: {path}")

    if path.is_file():
        if ckpt_filename_match is not None and ckpt_filename_match not in path.name:
            raise ValueError(
                f"Checkpoint file {path} does not match ckpt_filename_match="
                f"{ckpt_filename_match!r}."
            )
        return path

    ckpt_dir = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getmtime)
    if ckpt_filename_match is not None:
        ckpts = [x for x in ckpts if str(ckpt_filename_match) in x.name]
    if not ckpts:
        msg = f"No checkpoints found in {ckpt_dir}"
        if ckpt_filename_match is not None:
            msg += f" matching {ckpt_filename_match!r}"
        raise FileNotFoundError(msg + ".")
    print("Found checkpoints", ckpts)
    print("Using checkpoint", ckpts[-1])
    return ckpts[-1]


def merge_existing_run_config(cfg: DictConfig, exp_cfg: DictConfig):
    # we just copy cluster info
    cfg.module = exp_cfg.module
    cfg.dataloader = exp_cfg.dataloader
    # then we update with cli overrides
    print("\nhydra config", cfg)
    try:
        # if not submitit
        cli_overrides = HydraConfig.get().overrides.task
        print("got cli arguments from direct launch")
    except:  # noqa E722
        cli_overrides = getattr(cfg, "cli_overrides", [])

    cli_overrides = [x.removeprefix("++") for x in cli_overrides if x.startswith("+")]

    OmegaConf.set_struct(cfg, False)  # to merge
    cfg.merge_with_dotlist(cli_overrides)
    print("updated cfg", cfg, "\n")


class CheckpointEveryNSteps(L.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        dirpath="./",
        save_step_frequency=100000,
        prefix="checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.dirpath = dirpath

    def on_train_batch_end(self, trainer: L.Trainer, *args, **kwargs):
        """Check if we should save a checkpoint after every train batch"""
        if not hasattr(self, "trainer"):
            self.trainer = trainer

        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            self.save()

    def save(self, *args, trainer=None, **kwargs):
        if trainer is None and not hasattr(self, "trainer"):
            print("No trainer !")
            return
        if trainer is None:
            trainer = self.trainer

        global_step = trainer.global_step
        if self.use_modelcheckpoint_filename:
            filename = trainer.checkpoint_callback.filename
        else:
            filename = f"{self.prefix}_{global_step=}.ckpt"
        ckpt_path = Path(self.dirpath) / "checkpoints"
        print("saving checkpoint to", ckpt_path / filename)
        ckpt_path.mkdir(exist_ok=True, parents=True)
        trainer.save_checkpoint(ckpt_path / filename)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except:  # noqa E722
        pass

    warnings.simplefilter(action="ignore", category=FutureWarning)
    print("Working dir", os.getcwd())

    main_node = int(os.environ.get("SLURM_PROCID", 0)) == 0
    print("is main node", main_node)

    # init some variables
    logger = None
    ckpt_path = None
    # delete submitit handler to let PL take care of resuming
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    if cfg.mode not in {"train", "test"}:
        raise ValueError(f"Unsupported mode={cfg.mode!r}. Expected 'train' or 'test'.")

    exp_dir = Path(cfg.exp_dir)
    exp_cfg_path = exp_dir / "config.yaml"
    exp_exists = exp_dir.exists()
    load_ckpt = cfg.get("load_ckpt")
    load_existing_run = cfg.mode == "test" or cfg.resume

    if load_existing_run:
        if load_ckpt is not None:
            raise ValueError(
                "`load_ckpt` is only supported when starting a new training run (e.g., for fine-tuning)."
            )
        if not exp_exists:
            raise FileNotFoundError(f"Experiment does not exist: {exp_dir}")
        if not exp_cfg_path.exists():
            raise FileNotFoundError(f"Experiment config does not exist: {exp_cfg_path}")
        if cfg.resume:
            print("Experiment already exists. Resuming it.")
        exp_cfg = OmegaConf.load(exp_cfg_path)
        merge_existing_run_config(cfg, exp_cfg)
        ckpt_path = select_checkpoint(cfg.exp_dir, cfg.get("ckpt_filename_match"))
        print("Lightning will restore full trainer state from the checkpoint.")
    else:
        if exp_exists:
            raise FileExistsError(
                f"Experiment already exists: {exp_dir}. Use a new `name`/`exp_dir`, or set "
                "`resume=True` to continue training."
            )
        if load_ckpt is None and cfg.get("ckpt_filename_match") is not None:
            raise ValueError("`ckpt_filename_match` requires `load_ckpt` when starting a new run.")

    if cfg.log:
        print("wandb mode", cfg.cluster.wandb_mode)
        print("wandb service", os.environ.get("WANDB_DISABLE_SERVICE", "variable unset"))
        run_id = cfg.name + "-" + get_random_code() if cfg.cluster.use_custom_requeue else cfg.name
        logger = L.pytorch.loggers.WandbLogger(
            **(dict(entity=cfg.entity) if hasattr(cfg, "entity") and cfg.entity else {}),
            project=cfg.project,
            name=cfg.name,
            id=run_id,
            save_dir="wandblogs",
            offline=(cfg.cluster.wandb_mode != "online"),
        )

    if not load_existing_run and main_node:
        print("registering exp on main node")
        hparams = OmegaConf.to_container(cfg, resolve=True)
        if cfg.log:
            logger.log_hyperparams(hparams)
        exp_dir.mkdir(exist_ok=False, parents=True)
        with open(exp_dir / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.mode == "train":
        val_args = getattr(cfg.dataloader, "validation_args", {})
        valset = instantiate(cfg.dataloader.dataset, **val_args)
        trainset = instantiate(cfg.dataloader.dataset)  # will automatically pickup cfg split

        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=cfg.batch_size,
            num_workers=cfg.cluster.cpus,
            shuffle=True,
            collate_fn=collate_fn,
        )  # to viz shuffle samples

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=cfg.batch_size,
            num_workers=cfg.cluster.cpus,
            shuffle=True,
            collate_fn=collate_fn,
        )
    elif cfg.mode == "test":
        test_args = getattr(cfg.dataloader, "test_args", {})
        testset = instantiate(cfg.dataloader.dataset, **test_args)
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.batch_size,
            num_workers=cfg.cluster.cpus,
            shuffle=True,  # otherwise correlated batches
            collate_fn=collate_fn,
        )

    # Resolve interpolations in the entire config before passing `cfg.module`
    OmegaConf.resolve(cfg.module)
    pl_module = instantiate(cfg.module.module, cfg.module)

    if load_ckpt is not None:
        assert ckpt_path is None, (
            "`load_ckpt` initializes weights for a new run; it must not also resume a checkpoint."
        )
        # load weights w/o resuming optimizer/scheduler/global step
        load_ckpt_path = select_checkpoint(load_ckpt, cfg.get("ckpt_filename_match"))
        print(
            "Loading model weights from the checkpoint without resuming optimizer/scheduler/global step."
        )
        pl_module.load_state_dict(
            torch.load(load_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        )

    checkpointer = CheckpointEveryNSteps(
        dirpath=cfg.exp_dir, save_step_frequency=cfg.save_step_frequency
    )

    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    if cfg.cluster.use_custom_requeue and main_node:
        print("setting up custom slurm requeuing")

        def handler(*args, **kwargs):
            print("GCO: SIGTERM signal received. Requeueing job on main node.")
            if not hasattr(checkpointer, "is_handled"):
                checkpointer.is_handled = True
                checkpointer.save()
                from geoarches.submit import main as geoarches_submit

                cfg.resume = True
                geoarches_submit(cfg)
            exit()

        signal.signal(signal.SIGTERM, handler)

    torch.set_float32_matmul_precision("medium")
    L.seed_everything(cfg.seed)
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        strategy="ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto",
        precision=cfg.cluster.precision,
        log_every_n_steps=cfg.log_freq,
        profiler=getattr(cfg, "profiler", None),
        gradient_clip_val=1,
        max_steps=cfg.max_steps,
        enable_checkpointing=False,
        callbacks=[TQDMProgressBar(refresh_rate=100 if cfg.mode == "train" else 1), checkpointer],
        logger=logger,
        plugins=[],
        limit_train_batches=getattr(cfg, "limit_train_batches", None),
        limit_val_batches=cfg.limit_val_batches,
        limit_test_batches=getattr(cfg, "limit_test_batches", cfg.limit_val_batches),
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
    )

    if cfg.debug:
        breakpoint()

    if cfg.mode == "train":
        trainer.fit(pl_module, train_loader, val_loader, ckpt_path=ckpt_path)
    elif cfg.mode == "test":
        trainer.test(pl_module, test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
