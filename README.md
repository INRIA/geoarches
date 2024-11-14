# geoarches

## General presentation

This is a shared codebase for loading data and training models for the ARCHES team.

## Installation

You can start using (and modifying as needed) geoarches in your individual project, by depending on it as a package.

### Environment

Create an environment or activate the environment you are already using.

```sh
conda create --name weather python=3.10
conda activate weather
```

You can install the package in editable mode during development.
Editable mode allows you to make changes to the geoarches code locally, and these changes will automatically be reflected in your code that depends on it.

Move into this repo and type:
```sh
pip install -e .
pip install --no-dependencies tensordict
```
This also handles installing any dependencies.

### Useful directories

We recommend making the following symlinks in the codebase folder:
```sh
ln -s /path/to/data/ data
ln -s /path/to/models/ modelstore
ln -s /path/to/evaluation/ evalstore
ln -s /path/to/wandb/ wandblogs
```
Where `/path/to/models/` is where the trained models are stored, and `/path/to/evaluation/` is a folder used to store intermediate outputs from evaluating models. You can also simply create folders if you want to store data in the same folder.

## Use package

The recommended way to use the package is to depend on the package inside your own working directory.

You can use the geoarches tools directly by importing them from your directory, e.g.
```
from geoarches.dataloaders.era5 import Era5Forecast
```

If you want to train models with hydra, here are the necessary steps:

1) create a `main_hydra.py` training script with the following content:
```python
import hydra
from omegaconf import DictConfig
from geoarches.main_hydra import main as geoarches_main

__spec__ = None  # not sure why this is needed

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    geoarches_main(cfg)

if __name__ == "__main__":
    main()
```
This is so hydra can discover your local configs. Note that you can also reference geoarches's config files by their name.

If you want to submit to SLURM, you can create a `submit.py` file with
```python
# submitit file
import hydra
import submitit
from omegaconf import DictConfig, OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval)
except:
    pass

from hydra.core.hydra_config import HydraConfig

from geoarches.main_hydra import main as hydra_main


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # add cli override
    OmegaConf.set_struct(cfg, False)
    cfg["cli_overrides"] = HydraConfig.get().overrides.task
    aex = submitit.AutoExecutor(folder="sblogs/" + cfg.name, cluster="slurm")
    aex.update_parameters(**cfg.cluster.launcher)  # original launcher
    aex.submit(hydra_main, cfg)


if __name__ == "__main__":
    main()
```

2) create a `configs` folder for you hydra configuration files. In this folder, you can put your own configs, e.g. by copying config files from geoarches and modifying them. Please note the config files should be put in the appropriate folder (`cluster`, `dataloader` or `module`) in you own `configs` folder.

3) Add new lightning modules or architectures in your working directory (we recommend putting lightning modules in a `lightning_modules` folder, and pytorch-only backbone architectures in a `backbones` folder). To tell hydra to use these modules, you can create a module config file `custom_forecast.yaml` in `configs/module` as following:
```yaml
module:
  _target_: lightning_modules.custom_module.CustomLightningModule
  ...

backbone:
  _target_: backbones.custom_backbone.CustomBackbone
  ...
```
You can of course mix and match you custom modules and backbones with the ones in geoarches.

4) Start a training with hydra. If you are in a shell
```sh
python main_hydra.py cluster=local module=custom_forecast dataloader=era5 ++log=True ++name=default_run
```
If you want to submit to SLURM you can use `python submit.py` with the same arguments.

Useful options are 
```sh
python main_hydra.py \
++log=True \ # log metrics on weights and biases
++seed=0 \ # set global seed
++cluster.gpus=4 \ # number of gpus used for distributed training
++batch_size=1 \ # batch size per gpu
++max_steps=300000 \ # maximum number of steps for training, but it's good to leave this at 300k for era5 trainings
++save_step_frequency=50000 \ # if you need to save checkpoints at a higher frequency
++cluster.use_custom_requeue=False \ disable SLURM requeuing if not needed
```

to run evaluation of a model named `default_run` on the test set: 
```sh 
python main_hydra.py cluster=local ++mode=test ++name=default_run
```
It will automatically load the config file in `modelstore/default_run` and load the latest checkpoint from modelstore/default_run/checkpoints

TODO: for now it will not raise an error if the provided run does not exist.

Useful options for testing:
```sh python main_hydra.py ++mode=test ++name=default_run \
++ckpt_filename_match=100000 \ # substring that should be present in checkpoint file name, e.g. here for loading the checkpoint at step 100000
++limit_test_batches=0.1 \ # run test on only a fraction of test set for debugging
++module.module.rollout_iterations=10 \ # autoregressive rollout horizon, in which case the line below is also needed
++dataloader.test_args.multistep=10 \ # allow the dataloader to load trajectories of size 10

```


You can also make changes to the geoarches code, which will be reflected in your dependent code, which is great for fast prototyping but can make it more difficult to update geoarches. In that case, it's best to create a development branch, so as to rebase it on future updates of geoarches (see section **Pull changes** below).


## Contribute to geoarches
You can make changes on your own `dev` branch(s). This way you are not blocked by development on the `main` branch, but can still contribute to the `main` branch if you want to and can still incoroporate updates from other team members.

1. Create a `dev` branch from the `main` branch of geoarches to start making changes.
    ```sh
    cd geo_arches
    git checkout main
    git checkout -b dev_<name>
    ```

2. Commit and push your changes. 
3. Make sure tests pass by running `pytest tests/`.
4. To incorporate your changes into the `main` branch, make a merge request and wait for review.

## Pull changes

When the `main` branch of geoarches gets updated, and you want to incorporate changes.
This is important for both:
- Allowing you to take advantage of new code.
- Preemptively resolving any merge conflicts before merge requests.

The following steps will help you pull the changes from main and then apply your changes on top.
1. Either commit or stash your changes on your dev branch:
    ```sh
    git stash push -m "message"
    ```

2. Pull new changes into local main branch:
    ```sh
    git checkout main
    git pull origin main
    ```

3. Rebase your changes on top of new commits from main branch:
    ```sh
    git checkout dev_<name>
    git rebase main
    ```

    Resolve merge conflicts if needed. You can always decide to abort to undo the rebase completely:
    ```sh
    git rebase –abort
    ```

5. If you ran `git stash` in step 1, you can now apply your stashed changes on top.
    ```sh
    git stash pop
    ```

    Resolve merge conflicts if needed. To undo applying the stash:
    ```sh
    git reset --merge
    ```
    This will discard stashed changes, but stash contents won't be lost and can be re-applied later.