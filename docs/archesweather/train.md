# Train ArchesWeather and ArchesWeatherGen

This is the full training pipeline if you want to re-train ArchesWeatherGen from scratch. We assume that you have installed the geoarches package and downloaded the data with the script.

First, a few aliases to save space

```sh
alias train=python -m geoarches.main_hydra ++log=True
alias test=python -m geoarches.main_hydra ++mode=test

alias strain=python -m geoarches.submit
```

## Step 1: train deterministic models
Then, you need to train four deterministic version of ArchesWeather

```sh
for i in {0..3}; do
    train dataloader=era5 module=forecast-geoarchesweather ++name=archesweather-m-seed$i
done
```
This will start training for 4 deterministic models `ArchesWeather` on ERA5 data.
The model config will be saved to `modelstore/archesweather-m-seed$i/config.yaml` and the model checkpoints will be saved to `modelstore/archesweather-m-seed$i/checkpoints`.

In the released checkpoints, two models have a skip connection, but that should not really matter.

note that for ArchesWeatherGen, we don't need to do multistep fine-tuning.

## Step 2: compute residuals on the ERA5 dataset

Since ArchesWeatherGen models residuals, we can pre-compute them on the full dataset to save time during training:

```sh
python -m geoarches.inference.encode_dataset \
--uids archesweather-m-seed0,archesweather-m-seed1,archesweather-m-seed2,archesweather-m-seed3
--output-path data/outputs/deterministic/archesweather-m4/
```


## Step 3: ArchesWeatherGen main training

Now that the residuals are stored, we can run the main flow matching training

```sh
M4ARGS="++dataloader.dataset.pred_path=data/outputs/deterministic/archesweather-m4 \
++module.module.load_deterministic_model=[archesweather-m-seed0,archesweather-m-seed1,archesweather-m-seed2,archesweather-m-seed3] "

train module=archesweathergen dataloader=era5pred \
++limit_val_batches=10 ++max_steps=200000 \
++name=archesweathergen-s \
$M4ARGS \
++seed=0
```

## Step 4: ArchesWeatherGen fine-tuning

In the paper, we fine-tune the model in 2019, to overcome overfitting of the deterministic model. See the paper for more details.

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
"++save_step_frequency=20000" \
```

## Step 5: evaluation

Finally, we can evaluate the saved model:

```sh
multistep=10
test ++name=archesweathergen-s-ft
++limit_test_batches=0.1 \ # optional, for running on fewer members 
++dataloader.test_args.multistep=$multistep \ 
++module.inference.save_test_outputs=True \ # can be set to False to not save forecasts \
++module.inference.rollout_iterations=$multistep \
++module.inference.num_steps=25 \ # number of diffusion steps 
++module.inference.num_members=50 \
++module.inference.scale_input_noise=1.05 \ # to use noise scaling 
```