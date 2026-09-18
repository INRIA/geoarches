# Installation

There are 2 options to install geoarches:
1. [Recommended for first time users] Install as a package from PyPI if you don't intend to make any modifications to the code.
2. Install from source if you intend to make modifications to the code.

## Option 1: Install from PyPI

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

## Option 2: Install from source

Clone the repository only if you want to contribute to `geoarches` or use unreleased
changes:

```sh
git clone https://github.com/INRIA/geoarches.git
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

## Working with Specific Models & Protocols

`geoarches` supports multiple specialized research applications and model configurations:

- **Medium-range weather forecasting**: For deterministic and generative medium-range weather forecasting using ArchesWeather and ArchesWeatherGen, see the [ArchesWeather project page](../archesweather/index.md).
- **Historical AMIP-style climate projections**: For multi-decadal climate simulations following the AI Model Intercomparison Project (AIMIP) Phase 1 protocol, see the [AIMIP project page](../aimip/index.md).
