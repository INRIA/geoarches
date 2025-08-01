# Evaluate

Set the model run name (used with the Hydra argument `++name=NAME`):

```sh
MODEL=archesweathergen
```

## Compute metrics

### Rank histogram

```sh
python -m geoarches.evaluation.eval_multistep  \
    --pred_path evalstore/${MODEL}/ \
    --output_dir evalstore/${MODEL}_metrics/ \
    --groundtruth_path data/era5_240/full/ \
    --multistep 10 --num_workers 4 \
    --metrics era5_rank_histogram_50_members
```

## Plot (WIP)

### Rank histogram

```sh
python -m geoarches.evaluation.plot --output_dir plots/
    --metric_paths evalstore/${MODEL}_metrics/test-multistep=10-era5_rank_histogram_50_members.nc \
    --model_names ArchesWeatherGen \
    --model_colors red \
    --metrics rankhist \
    --vars Z500:geopotential:level:500 T850:temperature:level:850 Q700:specific_humidity:level:700 U850:u_component_of_wind:level:850 V850:v_component_of_wind:level:850 \
    --rankhist_prediction_timedeltas 1 7 \
    --figsize 10 4
```
