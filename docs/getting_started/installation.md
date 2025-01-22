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

Check [ArchesWeather](../archesweather/index.md) section.