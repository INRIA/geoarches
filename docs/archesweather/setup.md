# Setup

### 1. Install the package

To get started, follow the [installation guide](../getting_started/installation.md) to install the package and all required dependencies.

!!! tip

    If you plan to modify the codebase, it's recommended to fork the repository first. You’ll find relevant setup steps in the [contributing section](../contributing/index.md).

### 2. Download pretrained models

The following script downloads four deterministic models (`archesweather-m-seed*`) and one generative model (`archesweathergen`) from Hugging Face:

```sh
src="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"
MODELS=("archesweather-m-seed0" "archesweather-m-seed1" "archesweather-m-skip-seed0" "archesweather-m-skip-seed1" "archesweathergen")

for MOD in "${MODELS[@]}"; do
    mkdir -p "modelstore/$MOD/checkpoints"
    wget -O "modelstore/$MOD/checkpoints/checkpoint.ckpt" "$src/${MOD}_checkpoint.ckpt"
    wget -O "modelstore/$MOD/config.yaml" "$src/${MOD}_config.yaml"
done
```

You can then follow the [notebook tutorial](./run.ipynb) to load the models and run inference. For training, refer to the [train section](./train.md).

### 3. Download ERA5 quantile statistics

ERA5 quantiles are required to compute Brier scores and are used during both inference and training. Download them with:

```sh
src="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"
wget -O geoarches/stats/era5-quantiles-2016_2022.nc $src/era5-quantiles-2016_2022.nc
```
