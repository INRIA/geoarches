from pathlib import Path

import diffusers
import lightning as L  # noqa N812
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

# import ValueError


def load_module(
    path: str,
    device: str = "auto",
    dotlist: list = [],
    return_config: bool = True,
    ckpt_fname: str | None = None,
    **kwargs,
):
    """
    Args:
        path: Path under `modelstore` directory, holding hydra config `config.yaml` and lightning module checkpoint(s) under `checkpoints/*.chkpt`.
        dotlist: list of config overrides.
        return_config: Whether to return cfg along with module, or just the instantiated module.
        ckpt_fname: Optional. Checkpoint filename under `checkpoints/`, otherwise chooses most recent file.
    """
    if Path("modelstore").joinpath(path).exists():
        path = Path("modelstore").joinpath(path)
    else:
        path = Path(path)
    cfg = OmegaConf.load(path / "config.yaml")
    cfg.merge_with_dotlist(dotlist)
    module = instantiate(cfg.module.module, cfg.module, **kwargs)
    module.init_from_ckpt(path, ckpt_fname=ckpt_fname, missing_warning=False)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(device).eval()
    if not return_config:
        return module
    return module, cfg


class BaseLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()

    def mylog(self, dct={}, mode="auto", **kwargs):
        if mode == "auto":
            mode = "train_" if self.training else "val_"
        dct.update(kwargs)
        for k, v in dct.items():
            self.log(mode + k, v, prog_bar=True, sync_dist=True, add_dataloader_idx=True)

    def init_from_ckpt(
        self,
        path: str,
        ckpt_fname: str | None = None,
        ignore_keys: list = list(),
        missing_warning: bool = True,
    ):
        """
        Args:
            path: Directory holding lightning module checkpoint(s) under `checkpoints/*.chkpt`.
            ckpt_fname: Optional. Checkpoint filename under `checkpoints/`, otherwise chooses most recent file.
            ignore_keys: List of prefixes to ignore in keys in the checkpoint state_dict.
            missing_warning: Whether to warn if there keys in the lightning module that are missing from the checkpoint.
        """
        if Path(path).is_dir():
            path = Path(path) / "checkpoints"
            paths = list(Path(path).glob("*.ckpt"))
            if ckpt_fname is not None:
                paths = [p for p in paths if ckpt_fname in p.name]
            # sort by date
            path = sorted(paths, key=lambda x: x.stat().st_mtime)[-1]

        sd = torch.load(path, weights_only=False, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        missing_keys = set(
            [".".join(k.split(".")[:2]) for k in self.state_dict().keys() if k not in sd.keys()]
        )
        if len(missing_keys) and missing_warning:
            print("Missing keys", missing_keys)
        print(f"Restored from {path}")

    def configure_optimizers(self):
        print("configure optimizers")
        if self.ckpt_path is not None:
            opt = torch.optim.AdamW(
                self.parameters(),
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


class AvgModule(L.LightningModule):
    """
    Wrapper around several lightning modules to run forward and compute average prediction.
    """

    def __init__(self, module_paths):
        super().__init__()
        self.core = nn.ModuleList([load_module(p, return_config=False) for p in module_paths])

    def forward(self, *args, **kwargs):
        return torch.stack([m.forward(*args, **kwargs) for m in self.core], dim=0).mean(dim=0)
