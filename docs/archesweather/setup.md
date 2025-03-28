# Setup

### 1. Install package

Follow [installation](../getting_started/installation.md) instructions to install the package and its dependencies.

???+ tip

    If you want to make modifications on top, you can fork the repo (follow setup in the [contributing](../contributing/contribute.md#setup) section)

### 2. Download saved models 

Use following the script to download the 4 deterministic models (archesweather-m-...) and generative model (archesweathergen).

Use following the script to download the 4 deterministic models (archesweather-m-...) and generative model (archesweathergen).

```sh
src="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"
MODELS="archesweather-m-seed0 archesweather-m-seed1 archesweather-m-skip-seed0 archesweather-m-skip-seed1 archesweathergen"
for MOD in $MODELS; do
    mkdir -p modelstore/$MOD/checkpoints
    wget -O modelstore/$MOD/checkpoints/checkpoint.ckpt $src/${MOD}_checkpoint.ckpt
    wget -O modelstore/$MOD/config.yaml $src/${MOD}_config.yaml 
done
```
You can follow instructions in [`archesweather/tutorial.ipynb`](archesweather/tutorial.ipynb) to load the models and run inference with them. See [`archesweathergen/pipeline.md`](archesweather/pipeline.md) to run training.

### 3. Download ERA5 statistics

To compute brier score on ERA5 (needed to instantiate ArchesWeather models for inference or training), you will need to download ERA5 quantiles:
```sh
src="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"
wget -O geoarches/stats/era5-quantiles-2016_2022.nc $src/era5-quantiles-2016_2022.nc
```