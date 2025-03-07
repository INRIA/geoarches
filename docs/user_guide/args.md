# Hydra Parameters

Full list of Hydra arguments that can be either modified in Hydra config files or overridden in CLI.

CLI Usage:
```sh
python -m geoarches.main_hydra ++{arg_name}={arg_value}
```

The two arguments you absolutely should know are:

1. `mode` chooses between running training (mode=`train`) or evaluation (mode=`test`).
2. `name` is the unique run id. Must be updated for each new run. Tip: make it readable.

## Pipeline args

| arg_name                       | Default value        | Description  |
| ------------------------------ | -------------------- | ------------ |
| `mode`                         | 'train'              | `train` to run training ie. runs `LightningModule.fit()`<br/>`test` to run evaluation ie. runs`LightningModule.test()`|
| `accumulate_grad_batches`      | 1                    | Accumulates gradients over k batches before stepping the optimizer. Used by [Lightning API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#accumulate-grad-batches). |
| `batchsize`                    | 1                    | Batch size of dataloaders for train, val, and test. |
| `limit_train_batches`<br/>`limit_val_batches`<br/>`limit_test_batches` | Optional. | Limit batches loaded in dataloaders in [Lightning API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#limit-train-batches). |
| `log_freq`                     | 100                  | Frequency to log metrics. |
| `max_steps`                    | 300000               | Max steps to run training. |
| `seed`                         | 0                    | Seed lightning with `L.seed_everything(cfg.seed)` |

## Args to save and load checkpoints

| arg_name                       | Default value        | Description  |
| ------------------------------ | -------------------- | ------------ |
| `exp_dir`                      | 'modelstore/${name}' | During training, folder to store model checkpoints and hydra config used. If run already exists, pipeline will try to resume training instead.<br/>During evaluation, folder to load checkpoint and config from.<br/>By default, chooses latest checkpoint in dir (unless `ckpt_filename_match` specified). Recommendation: do not change this arg and change `name` for each new run. |
| `name`                         | 'default-run'        | Default `exp_dir` will use `name` to set checkpoint folder to `modelstore/${name}/checkpoints/`. This is also the Wandb run name. Unique display name, update every time you launch a new training run. |
| `resume`                       | `True`               | Set `True` to resume training from a checkpoint when mode=`train`. |
| `ckpt_filename_match`          | Optional             | Set to substring to match checkpoints files under `exp_dir`/checkpoints/ if resuming checkpoint to train or running evaluation. Pipeline will choose latest checkpoint under `exp_dir/checkpoints/` that contains the substring `ckpt_filename_match`. |
| `load_ckpt`                    | Optional             | Path to load Pytorch lightning module checkpoint from but not resume run. Not compatible with `ckpt_filename_match`. Will load checkpoint, but not resume training. |
| `save_step_frequency`          | 50000                | Save checkpoint every N steps. |
## Logging args

Currently only supports logging to WandB. See [User Guide](../user_guide/index.md#weights-and-biases-wandb) for more info.

| arg_name                       | Default value        | Description  |
| ------------------------------ | -------------------- | ------------ |
| `log`                          | `False`              | Set `True` to log metrics. |
| `cluster.wandb_mode`           | 'offline'            | `online` allows machine with internet connection to log directly to wandb.<br/> `offline` mode logs locally and requires a separate step to sync with wandb. |
| `entity`                       | Optional             | WandB [entity](https://docs.wandb.ai/ref/python/init/). If not set, WandB assumes username. |
| `project`                      | `False`              | WandB [project](https://docs.wandb.ai/ref/python/init/) to log run under. |
| `name`                         | 'default-run'        | Wandb run [name](https://docs.wandb.ai/ref/python/init/). Unique display name, update every time you launch a new training run. Note: Default `exp_dir` will also use `name` to set checkpoint folder to `modelstore/${name}/checkpoints/`. If run already exists, pipeline (if in `train` mode) will try to resume training from a checkpoint instead. |

## Module args

Check module class.

## Dataloader args

Check dataloader and backbone class.

## Cluster args

| arg_name                       | Default value        | Description  |
| ------------------------------ | -------------------- | ------------ |
| `cluster.cpus`                 | 1                    | Number of cpus running. Used for dataloader multi-threading. |
| `cluster.precision`            | '16-mixed'           |  Lightning [precision](https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision) |
| `cluster.use_custom_requeue`   | `False`              |  Set `True` to handle job prematurely prempting on computing node. Before exiting, it will save checkpoint and re-enqueue node. |