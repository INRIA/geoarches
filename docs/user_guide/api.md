# Hydra parameters

These are all the supported Hydra arguments that can be modified via configuration files or overridden directly in the CLI.

!!! tip "CLI usage"

    You can override any argument by using the `++arg_name=arg_value` syntax in the command line.

    ```sh
    python -m geoarches.main_hydra ++arg_name=arg_value
    ```

!!! note

    If you only need to remember two arguments, the most important are:

    1. `mode` selects between training (mode=`train`) and evaluation (mode=`test`).
    2. `name` is a unique identifier for your run, make it meaningful (and readable)!

## Pipeline

These arguments are used to configure the training or evaluation pipeline.

| arg_name                                                               | Default value   | Description                                                                                                                                                                |
| ---------------------------------------------------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`                                                                 | `'train'`       | `train` lauches training (i.e. `LightningModule.fit()`)<br/>`test` launches evaluation (i.e. `LightningModule.test()`)                                                     |
| `accumulate_grad_batches`                                              | 1               | Number of batches to accumulate before stepping the optimizer (see [Lightning API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#accumulate-grad-batches)). |
| `batch_size`                                                           | 1               | Batch size for train, validation and test dataloaders.                                                                                                                     |
| `limit_train_batches`<br/>`limit_val_batches`<br/>`limit_test_batches` | 1.0, _Optional_ | Limit the number of batches loaded in dataloaders (see [Lightning API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#limit-train-batches)).                 |
| `log_freq`                                                             | 100             | How often to log metrics (in steps).                                                                                                                                       |
| `max_steps`                                                            | 300000          | Maximum number of training steps.                                                                                                                                          |
| `seed`                                                                 | 0               | Seed used for reproducibility. Set via `L.seed_everything(seed)`.                                                                                                          |

## Checkpointing

These arguments are used to configure how checkpoints are saved and loaded during training or evaluation.

| arg_name              | Default value          | Description                                                                                                                                                                                                                                                                                                                      |
| --------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                | `'default-run'`        | A unique ID for your run. Used in checkpointing, W&B logging, etc. We recommend to change it **for each new run**.                                                                                                                                                                                                               |
| `exp_dir`             | `'modelstore/${name}'` | Directory for saving/loading checkpoints and configs.<br>Training will resume if a run exist. Evaluation will load checkpoint and config. By default, loads the latest checkpoint, unless `ckpt_filename_match` is specified. We **strongly** advise to **not change** this argument and instead change `name` for each new run. |
| `resume`              | `True`                 | Whether to resume training from a checkpoint when mode=`train`. If checkpoint does not exist, a new run will start. If set to `False`, a new run will always start.                                                                                                                                                              |
| `ckpt_filename_match` | _Optional_             | If specified, loads the checkpoint file whose name contains this substring. If multiple matches are found, the latest checkpoint will be loaded. Not compatible with `load_ckpt`.                                                                                                                                                     |
| `load_ckpt`           | _Optional_             | Path to a PyTorch Lightning checkpoint file to load for evaluation or inference only (does not resume training). Not compatible with `ckpt_filename_match`.                                                                                                                                                                      |
| `save_step_frequency` | 50000                  | How often to save checkpoints (in steps).                                                                                                                                                                                                                                                                                        |

## Logging

These arguments are used to configure experiment logging.

!!! warning

    Currently only supports W&B logging.
    See [User Guide](../user_guide/index.md#weights-biases-wb) for more information.

| arg_name             | Default value | Description                                                                                                                                                        |
| -------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `log`                | `False`       | Whether to enable logging. Set to `True` to log metrics.                                                                                                           |
| `cluster.wandb_mode` | `'offline'`   | `online` logs directly to W&B and requires internet connection.<br/>`offline` saves locally and needs [manual syncing](https://docs.wandb.ai/ref/cli/wandb-sync/). |
| `entity`             | _Optional_    | [W&B entity](https://docs.wandb.ai/ref/python/init/), usually your username or team.                                                                               |
| `project`            | _Optional_    | [W&B project](https://docs.wandb.ai/ref/python/init/) name. By default, inferred from `'${module.project}'`.                                                       |

## Module and backbones arguments

These arguments define your model configuration, including Lightning modules and backbones. For a comprehensive list and detailed documentation, refer to the source code and class docstrings. To get started, you can review existing configuration files in `configs/module/`.

## Dataloader arguments

These arguments define your dataloader configuration. For a comprehensive list and detailed documentation, refer to the source code and class docstrings. To get started, you can review existing configuration files in `configs/dataloader/`.

## Cluster arguments

These arguments are used to configure the cluster environment for training or evaluation. They are typically set in the `configs/cluster/` directory.

| arg_name                     | Default value | Description                                                                                                                    |
| ---------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `cluster.cpus`               | 1             | Number of CPUs to use. Used for dataloader multi-threading.                                                                    |
| `cluster.gpus`               | 1             | Number of GPUs to use. Set to `0` for CPU-only training.                                                                       |
| `cluster.precision`          | '16-mixed'    | Lightning [precision](https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision)                                  |
| `cluster.use_custom_requeue` | `False`       | Set `True` to handle job prematurely prempting on computing node. Before exiting, it will save checkpoint and re-enqueue node. |

We use [submitit](https://github.com/facebookincubator/submitit) to submit and manage jobs on a SLURM cluster. If you need specific SLURM options, use `launcher.arg_name` in your configuration file. Here is a small curated list of the most commonly used arguments:

| arg_name                               | Default value     | Description                                                             |
| -------------------------------------- | ----------------- | ----------------------------------------------------------------------- |
| `launcher.cpus_per_task`               | 1                 | Number of CPUs to use. Used for dataloader multi-threading.             |
| `launcher.gpus_per_node`               | 1                 | Number of GPUs to use.                                                  |
| `launcher.nodes`                       | 1                 | Number of nodes to use.                                                 |
| `launcher.tasks_per_node`              | 1                 | Number of tasks per node.                                               |
| `launcher.timeout_min`                 | 60                | Maximum duration of the job in minutes. Used for SLURM `--time` option. |
| `launcher.slurm_signal_delay_s`        | 60                | Delay before exiting after receiving a signal. Used for requeuing jobs. |
| `launcher.slurm_additional_parameters` | dict, _Optional_  | Additional SLURM parameters to pass, e.g. `hint=nomultithread`.         |
| `launcher.slurm_setup`                 | array, _Optional_ | Additional commands to pass **before** `srun`, e.g. module loading.     |
| `launcher.slurm_srun_args`             | array, _Optional_ | Additional arguments to pass to `srun`.                                 |
