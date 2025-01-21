# Installation

## Install poetry

We use poetry for package dependencies. Use pipx to install poetry:

```sh
brew install pipx
pipx install poetry
```

## Environment

Create an environment or activate the environment you are already using.

```sh
conda create --name weather python=3.11
conda activate weather
```

Move into the git repo and install dependencies:
```sh
cd geoarches
poetry install
```

Poetry, by default, installs the geoarches package in editable mode.
Editable mode allows you to make changes to the geoarches code locally, and these changes will automatically be reflected in your code that depends on it.

## Useful directories

We recommend making the following symlinks in the codebase folder:
```sh
ln -s /path/to/data/ data             # Store data for training and evaluation.
ln -s /path/to/models/ modelstore     # Store model checkpoints and model hydra configs.
ln -s /path/to/evaluation/ evalstore  # Store intermediate model outputs for computing metrics.
ln -s /path/to/wandb/ wandblogs       # Store Wandb logs.
```
If you want to store models and data in your working directory, or can also simply create regular folders.

## Downloading ArchesWeather and ArchesWeatherGen
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

## Downloading ERA5 statistics
To compute brier score on ERA5 (needed to instantiate ArchesWeather models for inference or training), you will need to download ERA5 quantiles:
```sh
src="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"
wget -O geoarches/stats/era5-quantiles-2016_2022.nc $src/era5-quantiles-2016_2022.nc
```