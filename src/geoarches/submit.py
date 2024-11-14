# submitit file
import hydra
import submitit
from omegaconf import DictConfig, OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval)
except:
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
