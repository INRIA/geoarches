# Train models with CLI

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

See [Pipeline API](args.md) for full list of arguments.
## Run on SLURM

To run on a SLURM cluster, you can create a `configs/cluster` folder inside your working directory and put a ``custom_slurm.yaml`` configuration file in it with custom arguments. Then you call tell geoarches to use this configuration file with

```sh
python -m geoarches.submit --config-dir configs cluster=custom_slurm
```

## Log experiments to Wandb

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