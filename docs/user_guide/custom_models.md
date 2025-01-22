
# Implementing your own models

## Step 1: Implement

Add new lightning modules or architectures in your working directory (we recommend putting lightning modules in a `lightning_modules` folder, and pytorch-only backbone architectures in a `backbones` folder).

## Step 2: Configure with Hydra

Create a `configs/` folder in your own project folder for your hydra configuration files. In this folder, you can put your own configs, e.g. by copying config files from geoarches and modifying them. Please note the config files should be put in the appropriate folder (`configs/cluster/`, `configs/dataloader/` or `configs/module/`). You will need a base `configs/config.yaml`. See `geoarches/configs/` for an example.

Tell hydra to use your custom modules: you can create a module config file `custom_forecast.yaml` under `configs/module/` and point to your new backbone and module classes:
    ```yaml
    module:
    _target_: lightning_modules.custom_module.CustomLightningModule
    ...

    backbone:
    _target_: backbones.custom_backbone.CustomBackbone
    ...
    ```
    You can of course mix and match your custom modules and backbones with the ones in geoarches.

## Step 3: Run

Training models only requires one to tell hydra to use your `configs` folder with

```sh
python -m geoarches.main_hydra --config-dir configs
```