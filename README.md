<h1 align="center">
  <a href="http://www.geoarches.readthedocs.io">
    <img src="./docs/assets/logo.png" alt="geoarches logo" width="120" height="120">
  </a>
  <br/>
  geoarches
</h1>

<p align="center"><strong>Machine learning framework for geospatial data, mainly climate and weather.</strong></p>

<p align="center">
  <a href="https://geoarches.readthedocs.io/">Documentation</a>
</p>

## What is geoarches?

**geoarches** is a machine learning library for training, running and evaluating models on weather and climate data, developed by Guillaume Couairon and Renu Singh in the ARCHES team at INRIA (Paris, France).

geoarches's building blocks can easily be integrated into any research ML pipelines.
It can also be used to run the [ArchesWeather and ArchesWeatherGen](https://arxiv.org/abs/2412.12971) weather models.

geoarches builds on PyTorch, PyTorch Lightning and Hydra. Once installed, you can use its modules inside your own project, or use the main training and evaluating workflows.
To develop your own models or modify existing ones, the intended usage is to work in your own working directory and create your own configurations files and Lightning modules. See the [User Guide](https://geoarches.readthedocs.io/en/latest/user_guide/custom_models/) for the full details.

## Code Overview

geoarches is meant to jumpstart your ML pipeline with building blocks for data handling, model training, and evaluation. This is an effort to share engineering tools and research knowledge across projects.

### Data

- `download/`: Parallelized dataset download scripts with support for chunking to speed up read access.
- `dataloaders/`: PyTorch datasets for loading and preprocessing NetCDF files into ML-ready tensors.

### Model training

- `backbones/`: Network architectures that plug into Lightning modules.
- `lightning_modules/`: Training and inference wrappers that are agnostic to the backbone but specific to the ML task — handle losses, optimizers, and metrics.

### Evaluation

- `metrics/`: Tested suite of efficient, memory-friendly metrics.
- `evaluation/`: End-to-end scripts to benchmark model predictions and generate plots.

### Pipeline

- `main_hydra.py`: Entry point for training or inference using Hydra configurations.
- `docs/archesweather/`: Quickstart code for training and inference.

## Installation

Clone the repository and install the package with Poetry:

```sh
git clone git@github.com:INRIA/geoarches.git
cd geoarches
poetry install
```

> [!NOTE]
> By default, Poetry will install `geoarches` in editable mode. This allows you to make changes to the package locally, meaning any local changes will automatically be reflected to the code in your environment.

For full installation instructions, see the [documentation](https://geoarches.readthedocs.io/en/latest/getting_started/installation/).

## Contributing

The project welcomes contributions and suggestions. If you want to contribute, please read the [Contributing Guide](https://geoarches.readthedocs.io/en/latest/contributing/contribute/).

## External Resources

Many thanks to the authors of WeatherLearn for adapting the Pangu-Weather pseudocode to PyTorch. The code for our model is mostly based on their codebase.

- [WeatherBench](https://sites.research.google/weatherbench/)
- [WeatherLearn](https://github.com/lizhuoq/WeatherLearn/tree/master)

## License

geoarches is available under the [BSD 3-Clause License](LICENSE).
