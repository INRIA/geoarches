# Train ArchesWeather and ArchesWeatherGen

This section provides the **full** training pipeline to retrain **ArchesWeatherGen** from scratch. It assumes that you have already installed the `geoarches` package and downloaded the necessary data.

!!! info "Before you start"

    You can define the following aliases in your shell config (e.g. `.bashrc` or `.zshrc`) to simplify the commands:

    ```sh
    alias train=python -m geoarches.main_hydra ++log=True
    alias test=python -m geoarches.main_hydra ++mode=test
    alias strain=python -m geoarches.submit
    ```

## Step 1. Training ArchesWeather

First, we train four deterministic versions of ArchesWeather on ERA5 data:

```sh
for i in {0..3}; do
    train dataloader=era5 module=archesweather ++name=archesweather-m-seed$i
done
```

!!! success

    Each model will save its configuration to `modelstore/archesweather-m-seed{i}/config.yam`l and checkpoints under `modelstore/archesweather-m-seed$i/checkpoints/`.

!!! info

    In the released checkpoints, two models include skip connections, but that should not really matter.

!!! note

    ArchesWeatherGen **does not** require multistep fine-tuning.

## Step 2. Compute residuals on the ERA5 dataset

Since ArchesWeatherGen models residuals, we can pre-compute them on the full dataset to speed up training:

```sh
python -m geoarches.inference.encode_dataset \
    --uids archesweather-m-seed0,archesweather-m-seed1,archesweather-m-seed2,archesweather-m-seed3 \
    --output-path data/outputs/deterministic/archesweather-m4/
```

## Step 3. Training ArchesWeatherGen

Once residuals are computed, we train the flow matching model:

```sh
M4ARGS="++dataloader.dataset.pred_path=data/outputs/deterministic/archesweather-m4 \
++module.module.load_deterministic_model=[archesweather-m-seed0,archesweather-m-seed1,archesweather-m-seed2,archesweather-m-seed3] "

train module=archesweathergen dataloader=era5pred \
    ++limit_val_batches=10 \
    ++max_steps=200000 \
    ++name=archesweathergen-s \
    $M4ARGS \
    ++seed=0
```

## Step 4. Fine-tuning ArchesWeatherGen

In the paper, we fine-tune the model on 2019 data to overcome overfitting of the deterministic models. See the paper for more details.

```sh
train module=archesweathergen dataloader=era5pred \
    ++limit_val_batches=10 ++max_steps=60000 \
    "++name=archesweathergen-s-ft" \
    $M4ARGS \
    "++load_ckpt=modelstore/archesweathergen-s" \
    "++ckpt_filename_match=200000" \ # for loading the checkpoint at 200k steps
    "++dataloader.dataset.domain=val" \ # fine-tune on validation
    "++module.module.lr=1e-4" \
    "++module.module.num_warmup_steps=500" \
    "++module.module.betas=[0.95, 0.99]" \
    "++save_step_frequency=20000"
```

## Step 5. Evaluation

Finally, we can evaluate the saved model:

```sh
multistep=10

test ++name=archesweathergen-s-ft \
    ++limit_test_batches=0.1 \ # optional, for running on fewer members
    ++dataloader.test_args.multistep=$multistep \
    ++module.inference.save_test_outputs=True \ # can be set to False to not save forecasts \
    ++module.inference.rollout_iterations=$multistep \
    ++module.inference.num_steps=25 \ # number of diffusion steps
    ++module.inference.num_members=50 \
    ++module.inference.scale_input_noise=1.05  # to use noise scaling
```
