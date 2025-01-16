"""
Wrapper around main_hydra.py to launch job on SLURM cluster using submitit.

Uses config from `configs/cluster/...`.
Stores logs in `sblogs/{cfg.name}`.
"""

import hydra
import submitit
from omegaconf import DictConfig, OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval)
except:  # noqa E722
    pass

from hydra.core.hydra_config import HydraConfig

from geoarches.main_hydra import main as geoarches_main


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    try:
        OmegaConf.set_struct(cfg, False)
        cfg["cli_overrides"] = HydraConfig.get().overrides.task
    except ValueError:
        pass
    aex = submitit.AutoExecutor(folder="sblogs/" + cfg.name, cluster="slurm")
    aex.update_parameters(**cfg.cluster.launcher)  # original launcher
    aex.submit(geoarches_main, cfg)


if __name__ == "__main__":
    main()
