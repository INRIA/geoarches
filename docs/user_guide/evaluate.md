# Run and evaluate models with the CLI

There are 2 options to evaluate models:

<ol type="A">
 <li>Run inference and metrics together.</li>
 <li>Run interence and metrics separately.</li>
</ol>

## A. Run inference and metrics

To evaluate a trained model (e.g. `ArchesWeather`) on the test set (year 2020), run:

```sh
MODEL=archesweather-m-seed0
python -m geoarches.main_hydra ++mode=test ++name=$MODEL
```

This command:

- Automatically loads the hydra config from `modelstore/$MODEL/config.yaml`
- Automatically loads the latest checkpoint from `modelstore/$MODEL/checkpoints/`
- Loads the data split defined in the hydra config under `dataloader.test_args.domain`.
- Runs the appropriate metrics (deterministic or generative depending on the model) defined in the hydra config under `module.inference.metrics`.
- Saves metrics under `evalstore/$MODEL/`.

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

Additional options for the diffusion module:

```sh
    ++module.inference.num_steps=25 # (1)!   # Number of diffusion steps
    ++module.inference.num_members=50 # (2)!   # Number of ensemble members to generate
```

1. Sets the number of diffusion steps for generative models.
2. Sets the number of ensemble members to generate during inference.

Refer to the [Pipeline API](api.md#pipeline) for a full list of arguments.

---

## B. Compute model outputs and metrics separately

You can decouple inference and metric computation. First, run inference and save the outputs:

```sh
python -m geoarches.main_hydra \
    ++mode=test \
    ++name=$MODEL ++module.name=$MODEL \
    ++module.inference.save_test_outputs=True
```

!!! info

    Predictions will be saved to: `evalstore/$MODEL/`

Then, compute metrics using `evaluation/eval_multistep.py`:

```sh
python -m geoarches.evaluation.eval_multistep \
    --pred_path evalstore/$MODEL/ \
    --output_dir evalstore/${MODEL}_metrics/ \
    --groundtruth_path data/era5_240/full/ \
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
        --model_colors orange blue \
        --metrics rankhist \
        --rankhist_prediction_timedeltas 1 7 \
        --figsize 10 4 \
        --vars Z500:geopotential:level:500 T850:temperature:level:850 Q700:specific_humidity:level:700 U850:u_component_of_wind:level:850 V850:v_component_of_wind:level:850 \
    ```