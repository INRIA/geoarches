# User Guide

## Running models

### Using hydra configuration

We use [Hydra](https://hydra.cc/docs/intro/) to easily configure training experiments. `main_hydra.py` is pointed to the `configs/` folder which tells geoarches which dataloader, lightning module, backbone, and their arguments to run. You can also override arguments by CLI (see below for useful arguments). Please read [Hydra](https://hydra.cc/docs/intro/) documentation for more information.

### Train models with CLI

To train model named `default_run`, you can run
```sh
python -m geoarches.main_hydra \
module=archesweather \ # Uses module/archesweather.yaml
dataloader=era5 \ # Uses dataloader/era5.yaml
++name=default_run \ # Name of run, used for Wandb logging and checkpoint dir
```
This will start a training for the deterministic model `ArchesWeather` on ERA5 data.

The model config will be saved to `modelstore/default_run/config.yaml` and the model checkpoints will be saved to `modelstore/default_run/checkpoints`.

Useful training options are 
```sh
python -m geoarches.main_hydra \
++log=True \ # log metrics on weights and biases (See Wandb section below.)
++seed=0 \ # set global seed
++cluster.gpus=4 \ # number of gpus used for distributed training
++batch_size=1 \ # batch size per gpu
++max_steps=300000 \ # maximum number of steps for training, but it's good to leave this at 300k for era5 trainings
++save_step_frequency=50000 \ # if you need to save checkpoints at a higher frequency
```
#### Run on SLURM

To run on a SLURM cluster, you can create a `configs/cluster` folder inside your working directory and put a ``custom_slurm.yaml`` configuration file in it with custom arguments. Then you call tell geoarches to use this configuration file with

```sh
python -m geoarches.submit --config-dir configs cluster=custom_slurm
```

#### Log experiments to Wandb

Find your API key under User settings in your account (https://docs.wandb.ai/support/find_api_key/) and set the Wandb environment variable in your `~/.bashrc`.
```
export WANDB_API_KEY="..."
```

Then tell geoaches to log to Wandb.
```sh
python -m geoarches.main_hydra \
++log=True \ # log metrics on weights and biases
++cluster.wandb_mode=offline \ # online allows machine with internet connection to log directly to wandb. Otherwise offline mode logs locally and requires a separate step to sync with wandb.
```

## Evaluate models with CLI

To run evaluation of a model (e.g. ArchesWeather) on the test set (2020), you can run 
```sh 
MODEL=archesweather-m
python -m geoarches.main_hydra ++mode=test ++name=$MODEL
```
It will automatically load the config file in `modelstore/$MODEL` and load the latest checkpoint from ``modelstore/$MODEL/checkpoints``.
It will then run the metrics relevant for the loaded model (deterministic metrics for deterministic models and similarly for generative models)

Warning: if the provided model does not exist, it will not throw an error.

Useful options for testing:
```sh
python -m geoarches.main_hydra ++mode=test ++name=$MODEL \
++ckpt_filename_match=100000 \ # substring that should be present in checkpoint file name, e.g. here for loading the checkpoint at step 100000
++limit_test_batches=0.1 \ # run test on only a fraction of test set for debugging
++module.module.rollout_iterations=10 \ # autoregressive rollout horizon, in which case the line below is also needed
++dataloader.test_args.multistep=10 \ # allow the dataloader to load trajectories of size 10

++dataloader.test_args.

```

For testing the generative models, you can also use the following options:
```sh
++module.inference.num_steps=25 \ # num diffusion steps in generation
++module.inference.num_members=50 \ # num members in ensemble
++module.inference.rollout_iterations=10 \ # number of auto-regressive steps, 10 days by default.
```

## Compute model outputs and metrics separately

You can compute model outputs and metrics separately. In that case, you first run evaluation as following:
```sh
python -m geoarches.main_hydra ++mode=test ++name=$MODEL \
++module.inference.save_test_outputs=False \
```

Then, to compute metrics, you can run `evaluation/eval_multistep.py` which reads in inference output from xarray files, computes specified metrics, and dumps metrics to `output_dir`. Example:

```sh
python -m geoarches.evaluation.eval_multistep \
    --pred_path evalstore/modelx_predictions/ \
    --output_dir evalstore/modelx_predictions/ \
    --groundtruth_path data/era5_240/full/  \
    --multistep 10 \
    --metrics era5_ensemble_metrics --num_workers 2
```

Before running, you need to make sure the metrics are registered in `evaluation/metric_registry.py` using register_metric(). You can find examples in the file. Example: 

    register_metric(
        "era5_ensemble_metrics",
        Era5EnsembleMetrics,
        save_memory=True,
    )

Metrics are registered with a name, class, and any arguments.

## Implementing your own models


1) Create a `configs/` folder in your own project folder for your hydra configuration files. In this folder, you can put your own configs, e.g. by copying config files from geoarches and modifying them. Please note the config files should be put in the appropriate folder (`configs/cluster/`, `configs/dataloader/` or `configs/module/`). You will need a base `configs/config.yaml`. See geoarches for an example.

2) Add new lightning modules or architectures in your working directory (we recommend putting lightning modules in a `lightning_modules` folder, and pytorch-only backbone architectures in a `backbones` folder). To tell hydra to use these modules, you can create a module config file `custom_forecast.yaml` under `configs/module/` and point to your new backbone and module classes:
```yaml
module:
  _target_: lightning_modules.custom_module.CustomLightningModule
  ...

backbone:
  _target_: backbones.custom_backbone.CustomBackbone
  ...
```
You can of course mix and match your custom modules and backbones with the ones in geoarches.

3. Training models only requires one to tell hydra to use your `configs` folder with
```sh
python -m geoarches.main_hydra --config-dir configs
```

## Plot (WIP)

You can plot metrics for several models using the script `plot.py`. Just specify where the computed metrics are stored (either .nc or .pt files). Example:

python -m geoarches.evaluation.plot --output_dir plots/ \ --metric_paths /evalstore/modelx/...nc /evalstore/modely/...nc --model_names_for_legend ModelX ModelY \ 
--metrics rankhist --rankhist_prediction_timedeltas 1 7 \ --figsize 10 4 --vars Z500 Q700 T850 U850 V850