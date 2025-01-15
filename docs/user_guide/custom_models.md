# Implementing your own models

You can easily extend `geoarches` with your own models. Here’s how:

## Step 1: Implement

Add your custom modules in your working directory.
We recommend the following structure:

- `lightning_modules/`: for Lightning modules (training logic, losses, logging, etc.)
- `backbones/`: for PyTorch-only architecture components (e.g. transformer blocks, CNNs)

## Step 2: Configure with Hydra

Create a `configs/` folder in your project to store your custom Hydra configuration files. You can copy and adapt files from `geoarches/configs/` as needed.

Organize configs under the appropriate subfolders:

- `configs/cluster/`
- `configs/dataloader/`
- `configs/module/`

You’ll also need a base `configs/config.yaml`.

To tell Hydra to use your custom classes, define a module config (e.g. `configs/module/custom_forecast.yaml`) like this:

```yaml
module:
  _target_: lightning_modules.custom_module.CustomLightningModule
  ...

backbone:
  _target_: backbones.custom_backbone.CustomBackbone
  ...
```

You can mix and match your own modules or backbones with those provided in `geoarches`.

## Step 3: Run

To train with your custom setup, simply point Hydra to your config directory:

```sh
python -m geoarches.main_hydra --config-dir configs
```
