# Train models with the CLI

## Basic usage

To train a model named `default_run`, run:

```sh
python -m geoarches.main_hydra \
    module=archesweather \ # (1)!
    dataloader=era5 \ # (2)!
    ++name=default_run # (3)!
```

1. Loads `configs/module/archesweather.yaml`
2. Loads `configs/dataloader/era5.yaml`
3. Unique name of your run, used for checkpointing and W&B logging

This will start training the deterministic model **ArchesWeather** on **ERA5** data.

!!! note

    The configuration file will be saved to: `modelstore/default_run/config.yaml` and model checkpoints to: `modelstore/default_run/checkpoints/`

### Useful training options

```sh
python -m geoarches.main_hydra \
    ++log=True \                  # Log metrics to Weights & Biases
    ++seed=0 \                    # Set global seed
    ++cluster.gpus=4 \            # Number of GPUs to use
    ++batchsize=1 \               # Batch size per GPU
    ++max_steps=300000 \          # Maximum number of training steps
    ++save_step_frequency=50000   # Save checkpoints every N steps
```

Refer to the [Pipeline API](api.md#pipeline) for a full list of arguments.

---

## Run on SLURM

To run training on a SLURM cluster:

1. Create a `configs/cluster/` folder inside your working directory.
2. Add a `custom_slurm.yaml` file with your cluster-specific settings.
3. Launch the run using:
   ```sh
   python -m geoarches.submit cluster=custom_slurm
   ```

Refer to the [Pipeline API](api.md#cluster-arguments) for a full list of arguments.

!!! note

    Depending on your familiarity with SLURM, you can also create a custom `sbatch` script to run `geoarches.main_hydra` directly, instead of using `geoarches.submit`.

---

## Log experiments to Weights & Biases

1. Find your API key from your W&B account settings ([How do I find my API key?](https://docs.wandb.ai/support/find_api_key/))
2. Add the key to your shell configuration file, e.g. in `~/.bashrc`:
   ```sh
   export WANDB_API_KEY="your-key-here"
   ```
3. Enable logging by setting the appropriate flags:
    ```sh
    python -m geoarches.main_hydra \
       ++log=True \
       ++cluster.wandb_mode=offline # (1)!
    ```
    
    1. If the machines have internet access, you may use `'online'` to live sync the experiment. `'offline'` is useful for running on machines without internet access, where logs will be synced later.

Refer to the [Pipeline API](api.md#logging) for a full list of arguments.
