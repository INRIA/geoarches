# Installation

There are two supported ways to install `geoarches`, both reading the same dependencies from `pyproject.toml`. Pick one:

- **Python env + Poetry** — manage the environment yourself (Conda, virtualenv, …), then install with [`poetry`](https://python-poetry.org/docs/) (this is the original workflow).
- **uv** — let [`uv`](https://docs.astral.sh/uv/) manage both the environment and the install in one step.

First, clone the repository:

```sh
git clone git@github.com:INRIA/geoarches.git
cd geoarches
```

## Option 1: Python environment + Poetry

We support [`poetry`](https://python-poetry.org/docs/) **>=2.2** (for shared [PEP 735](https://peps.python.org/pep-0735/) dependency groups). If `poetry` is not installed, follow the [official installation instructions](https://python-poetry.org/docs/#installation).

Use your preferred way to manage environments, e.g. `conda`, `virtualenv`, or `pyenv`. Poetry manages its own virtual environment, so this step is only about providing a compatible Python (`>=3.11,<3.13`) — Poetry selects an existing interpreter but does not install one (while `uv` can download a matching Python itself).

!!! example "Using Conda"

    The following code snippet will create a new conda environment named `geoarches` with Python 3.12:

    ```sh
    conda create --name geoarches python=3.12
    conda activate geoarches
    ```

Once your environment is activated, install the package with Poetry:

```sh
poetry install
```

## Option 2: uv

Install [`uv`](https://docs.astral.sh/uv/) by following the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv sync
```

`uv` creates a `.venv` for you (with a Python version matching `requires-python`) and installs the package into it. Run project commands via `uv run` (e.g. `uv run geoarches-main ...`), or activate the environment with `source .venv/bin/activate`.

!!! note

    Both options install `geoarches` in **editable mode**. This allows you to make changes to the package locally, meaning any local changes will **automatically be reflected** to the code in your environment. Both also install the `dev` dependency group by default.

!!! tip "Building the documentation locally"

    The documentation dependencies live in the optional `docs` group, which is not installed by default. Install it with `poetry install --with docs` (Poetry) or `uv sync --group docs` (uv).

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
