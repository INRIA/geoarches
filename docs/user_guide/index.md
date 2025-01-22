# User Guide

Detailed documentation for using geoarches. Check [Getting Started](../getting_started/) for installation and basic usage.

## Prerequisites

The package takes advantage of several tools. It might be helpful to become familiar with these tools first.

### Hydra

We use [Hydra](https://hydra.cc/docs/intro/) to easily configure training experiments.

The main python script (`main_hydra.py` that runs a model pipeline), is pointed to the `configs/` folder which tells geoarches which dataloader, lightning module, backbone, and their arguments to run. 

The config is constructed from the base config `configs/config.yaml` and is extended with configs under each folder such as `config/module/` and `config/dataloader/`.

You can also override arguments by CLI (see [Pipeline API](args.md) for full list of arguments).

Example:
```sh
python -m geoarches.main_hydra \
module=archesweather \ # Uses module/archesweather.yaml
dataloader=era5 \      # Uses dataloader/era5.yaml
++name=default_run \   # Name of run: used to name checkpoint dir and Wandb logging
```

### PyTorch and PyTorch Lightning

[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) is a wrapper around PyTorch and allows us to run training and inference loops without boilerplate code.

We mainly take advantage of the [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) API.

!!! note

    To just take advantage of data and evaluation modules, you do not need to use lightning in your project.

### Weights and Biases (WandB)

The training pipeline optionally uses [WandB](https://wandb.ai/site/) to log and track experiment metrics for your projects. You can create an account and project on the website.