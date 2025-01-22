# Evaluate models with CLI

## Run inference and metrics

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
```

For testing the generative models, you can also use the following options:
```sh
++module.inference.num_steps=25 \ # num diffusion steps in generation
++module.inference.num_members=50 \ # num members in ensemble
++module.inference.rollout_iterations=10 \ # number of auto-regressive steps, 10 days by default.
```

See [Pipeline API](args.md) for full list of arguments.

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

## Plot (WIP)

You can plot metrics for several models using the script `plot.py`. Just specify where the computed metrics are stored (either .nc or .pt files). Example:

```sh
python -m geoarches.evaluation.plot --output_dir plots/ \ --metric_paths /evalstore/modelx/...nc /evalstore/modely/...nc --model_names_for_legend ModelX ModelY \ 
--metrics rankhist --rankhist_prediction_timedeltas 1 7 \ --figsize 10 4 --vars Z500 Q700 T850 U850 V850
```