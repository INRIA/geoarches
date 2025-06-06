# Installation

## Install Poetry

We use [`poetry`](https://python-poetry.org/docs/) for package dependencies. If `poetry` is not installed, you can follow the [official installation instructions](https://python-poetry.org/docs/#installation).

## Set up Python environment

Use your preferred way to manage environments, e.g. `conda`, `virtualenv`, or `pyenv`.

!!! example "Using Conda"

    The following code snippet will create a new conda environment named `geoarches` with Python 3.12:

    ```sh
    conda create --name geoarches python=3.12
    conda activate geoarches
    ```

## Install the package

Once your environment is activated, clone the repository and install the package with Poetry:

```sh
git clone git@github.com:INRIA/geoarches.git
cd geoarches
poetry install
```

!!! note
    
    By default, Poetry will install `geoarches` in **editable mode**. This allows you to make changes to the package locally, meaning any local changes will **automatically be reflected** to the code in your environment.

## Useful directories

We recommend creating symlinks in the root the codebase:

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

You can also choose to create regular folders instead of symlinks. If none of these directories exist, they will be created automatically in the current working directory when needed.

## Downloading ArchesWeather and ArchesWeatherGen

To download pretrained models and statistics, follow the instructions in the [ArchesWeather section](../archesweather/index.md).
