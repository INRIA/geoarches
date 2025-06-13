# User Guide

This section provides detailed guidance for using `geoarches`.
If you’re just getting started, begin with the [Getting Started](../getting_started/installation.md) section for installation and basic usage.

## Prerequisites

`geoarches` builds on top of several open-source tools. While not strictly required, familiarity with the following tools will help you get the most out of the library.

### Hydra

We use [Hydra](https://hydra.cc/docs/intro/) for flexible and modular configuration of training experiments.

The main entry point is `main_hydra.py`, which builds the full configuration from components located under the `configs/` directory. This includes:

- The base config: `configs/config.yaml`
- Module-specific configs: e.g. `configs/module/archesweather.yaml`
- Dataloader configs: e.g. `configs/dataloader/era5.yaml`

You can override any argument via the command line (see [Pipeline API](api.md) for the full list).

!!! example

    ```sh
    python -m geoarches.main_hydra \
        module=archesweather \ # (1)!
        dataloader=era5 \ # (2)!
        ++name=default_run # (3)!
    ```

    1. Loads `configs/module/archesweather.yaml`
    2. Loads `configs/dataloader/era5.yaml`
    3. Unique name of your run, used for checkpointing and W&B logging

### PyTorch & PyTorch Lightning

We rely on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to simplify training and evaluation, removing much of the boilerplate around training loops.

In particular, we use the [`LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) abstraction to wrap backbone models, handle loss computation, optimizer setup, logging, and more.

!!! note

    If you're only interested in the data or evaluation utilities provided by `geoarches`, you **do not need** to use Lightning.

### Weights & Biases (W&B)

Optionally, you can log training metrics with [Weights & Biases](https://docs.wandb.ai). It provides experiment tracking for your runs.
