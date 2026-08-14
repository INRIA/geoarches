# Installation

## Install from PyPI

`geoarches` supports Python 3.11 through 3.14.0. We recommend installing it in a
virtual environment.

### pip

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install geoarches
```

### uv

Install [`uv`](https://docs.astral.sh/uv/) by following its
[installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install geoarches
```

You can also activate an existing Conda environment and run `python -m pip install
geoarches` in it.

Verify the installation with:

```sh
python -c "import geoarches; print(geoarches.__version__)"
```

## Install from source

Clone the repository only if you want to contribute to `geoarches` or use unreleased
changes:

```sh
git clone git@github.com:INRIA/geoarches.git
cd geoarches
```

Install [`uv`](https://docs.astral.sh/uv/) and run:

```sh
uv sync
```

Alternatively, install [`Poetry`](https://python-poetry.org/docs/) 2.2 or later and run
in a virtual environment:

```sh
poetry install
```

Both source-installation methods install `geoarches` in editable mode and include the
development dependencies. See the [Contributing Guide](../contributing/index.md) for the
complete development workflow.

!!! tip "Building the documentation locally"

    Documentation dependencies are opt-in. Install them with `uv sync --group docs` or
    `poetry install --with docs` from a source checkout.

## Useful directories

In the working directory for your project, we recommend creating these directories or
symlinks:

```sh
ln -s /path/to/data/ data # (1)!
ln -s /path/to/models/ modelstore # (2)!
ln -s /path/to/evaluation/ evalstore # (3)!
ln -s /path/to/wandb/ wandblogs # (4)!
```

1. `data/`: stores all datasets used for training and evaluation.
2. `modelstore/`: stores model checkpoints and Hydra configs.
3. `evalstore/`: stores intermediate model outputs used for evaluation metrics.
4. `wandblogs/`: stores Weights & Biases logs.

You can create regular directories instead. Missing directories are created in the current
working directory when needed.

## Downloading data

The `download/` folder contains scripts to download data.
To download the full ERA5 dataset from WeatherBench for training and evaluation, run:

```sh
python -m geoarches.download.dl_era --folder /path/to/data/era5_240/full/
```

If you only want to run evaluations, download the years 2019 to 2021:

```sh
python -m geoarches.download.dl_era --folder data/era5_240/full/ --years 2019 2020 2021
```

## Working with ArchesWeather and ArchesWeatherGen

To use ArchesWeather or ArchesWeatherGen, follow the
[ArchesWeather setup instructions](../archesweather/setup.md).
