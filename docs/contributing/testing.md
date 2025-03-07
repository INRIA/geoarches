# Integration testing

Since we do not have CI/CD pipeline set up for integration testing, here are some recommended commands to run and try out before submitting a pull request:

## Test 1: deterministic forecast module

### Train

```sh
python -m geoarches.main_hydra \
module=archesweather dataloader=era5 ++name=test_deterministic \
++log=True ++cluster.wandb_mode=online \
limit_train_batches=200 limit_val_batches=200 save_step_frequency=100 \
max_steps=1200
```

Checks:
- metrics are logged and look decent on Wandb.
- checkpoint and config dumped to `modelstore/test_deterministic/`.

### Evaluate

```sh
HYDRA_FULL_ERROR=1 python -m geoarches.main_hydra \
++name=test_deterministic \
mode=test \
++module.inference.save_test_outputs=True \
++module.inference.rollout_iterations=10 \
++dataloader.test_args.multistep=10 \
++limit_test_batches=0.1
```

Checks:
- predictions and metrics are dumped to `evalstore/test_deterministic/`.

## Test 2: diffusion module

### Train
```sh
HYDRA_FULL_ERROR=1 python -m geoarches.main_hydra \
module=archesweathergen dataloader=era5 ++name=test_diffusion \
++module.module.load_deterministic_model=[archesweather-m-seed0,archesweather-m-seed1,archesweather-m-skip-seed0,archesweather-m-skip-seed1] \
++log=True ++cluster.wandb_mode=online \
++limit_train_batches=200 ++limit_val_batches=200 save_step_frequency=100
```

Checks:
- metrics are logged and look decent on Wandb.
- checkpoint and config dumped to `modelstore/test_diffusion/`.

### Evaluate

```sh
HYDRA_FULL_ERROR=1 python -m geoarches.main_hydra \
++name=test_diffusion \
mode=test \
++module.inference.save_test_outputs=True \
++module.inference.rollout_iterations=10 \
++dataloader.test_args.multistep=10 \
++limit_test_batches=0.1
```

Checks:
- predictions and metrics are dumped to `evalstore/test_diffusion/`.
