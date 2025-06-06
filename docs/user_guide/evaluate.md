# Run and evaluate models with the CLI

## Run inference and metrics

To evaluate a trained model (e.g. `ArchesWeather`) on the test set (year 2020), run:

```sh
MODEL=archesweather-m-seed0
python -m geoarches.main_hydra ++mode=test ++name=$MODEL
```

This command:

- Automatically loads the config from `modelstore/$MODEL/config.yaml`
- Automatically loads the latest checkpoint from `modelstore/$MODEL/checkpoints/`
- Runs the appropriate metrics (deterministic or generative depending on the model)

!!! warning

    No error will be raised if the model does not exist, so make sure to check the `modelstore/` directory for the correct model name.

### Useful options for testing

```sh
python -m geoarches.main_hydra ++mode=test ++name=$MODEL \
    ++ckpt_filename_match=100000 \ # (1)!                  # Load checkpoint containing this substring
    ++limit_test_batches=0.1 \ # (2)!                      # Run on a fraction of the test set (for debugging)
    ++module.inference.rollout_iterations=10 \ # (3)!      # Number of autoregressive steps
    ++dataloader.test_args.multistep=10 # (4)!             # Match rollout length on dataloader side
```

1. Loads the model checkpoint containing `100000` in its filename.
2. Runs inference on 10% of the test set (useful for debugging).
3. Sets the number of autoregressive steps to 10.
4. Matches the rollout length on the dataloader side to ensure consistency.

Additional options for generative models:

```sh
    ++module.inference.num_steps=25 # (1)!   # Number of diffusion steps
    ++module.inference.num_members=50 # (2)!   # Number of ensemble members to generate
```

1. Sets the number of diffusion steps for generative models.
2. Sets the number of ensemble members to generate during inference.

Refer to the [Pipeline API](api.md#pipeline) for a full list of arguments.

---

## Compute model outputs and metrics separately

You can decouple inference and metric computation. First, run inference and save the outputs:

```sh
python -m geoarches.main_hydra \
    ++mode=test \
    ++name=$MODEL \
    ++module.inference.save_test_outputs=True
```

!!! info

    Predictions will be saved to: `evalstore/$MODEL/`

Then, compute metrics using `evaluation/eval_multistep.py`:

```sh
python -m geoarches.evaluation.eval_multistep \
    --pred_path evalstore/$MODEL/ \
    --output_dir evalstore/$MODEL/ \
    --groundtruth_path data/era5/ \
    --multistep 10 \
    --metrics era5_ensemble_metrics \
    --num_workers 2
```

This reads the inference outputs from Xarray files, computes the specified metrics, and writes the results to `output_dir`.

!!! note

    Make sure metrics are registered in `evaluation/metric_registry.py` using `register_metric`. You can find examples in the codebase, such as:

    ```python
    register_metric(
        "era5_ensemble_metrics",
        Era5EnsembleMetrics,
        save_memory=True,
    )
    ```

---

## Plot (WIP)

!!! info "Work in progress!"

You can visualize and compare metrics across models using the `plot.py` script. Be sure to specify where metrics are stored (either `nc` files or `pt` files).

!!! example

    ```sh
    python -m geoarches.evaluation.plot \
        --output_dir plots/ \
        --metric_paths evalstore/modelx/metrics.nc evalstore/modely/metrics.nc \
        --model_names_for_legend ModelX ModelY \
        --metrics rankhist \
        --rankhist_prediction_timedeltas 1 7 \
        --figsize 10 4 \
        --vars Z500 Q700 T850 U850 V850
    ```
