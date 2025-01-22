<h1 align="center">
  <a href="http://www.geoarches.readthedocs.io">
    <img src="https://geoarches.readthedocs.io/en/latest/img/logo.png" alt="geoarches Logo" width="120" height="120">
  </a>
  <br/>
  geoarches
</h1>

<p align="center"><strong>ML Framework for geospatial data, mainly climate and weather.</strong></p>

<p align="center">
  <a href="https://geoarches.readthedocs.io/">Documentation</a>
</p>


## What is geoarches?

geoarches is a machine learning package for training, running and evaluating ML models on weather and climate data, developed by Guillaume Couairon and Renu Singh in the ARCHES team at INRIA (Paris, France).

geoarches's building blocks can be easily integrated into research ML pipelines.
It can also be used to run the ArchesWeather and ArchesWeatherGen weather models.

geoarches is based on pytorch, pytorch-lightning and hydra for configuration. After the package is installed, you can use its modules in python code, but you can also call the main training and evaluating scripts of geoarches. 

To develop your own models or make modifications to the existing ones, the intended usage is to write configurations files and pytorch lightning classes in your own working directory. Hydra will then discover your custom ``configs`` folder, and you can point to your custom classes from your custom config files.

To access the full documentation of the project, follow the link [https://geoarches.readthedocs.io/](https://geoarches.readthedocs.io/)

## Code Overview

geoarches is meant to jumpstart your ML pipeline with building blocks for data handling, model training, and evaluation. This is an effort to share engineering tools and research knowledge across projects.

Data:
- `download/`: scripts that parallelize downloads and show how to use chunking to speed up read access.
- `dataloaders/`: PyTorch datasets that read netcdf data and prepare tensors to feed into model training.

Model training:
- `backbones/`: network architecture that can be plugged into lightning modules.
- `lightning_modules/`: wrapper around backbone modules to handle loss computation, optimizer, etc for training and inference (agnostic to backbone but specific to ML task).

Evaluation:
- `metrics/`: tested suite of iterative metrics (memory efficient) for deterministic and generative models.
- `evaluation/`: scripts for running metrics over model predictions and plotting.

Pipeline:
- `main_hydra.py`: script to run training or inference with hydra configuration.
- `documentation/`: quickstart code for training and inference from a notebook.

## Installation

See [documentation](https://geoarches.readthedocs.io/en/latest/getting_started/installation/) for full installation instructions.

Move into the git repo and install dependencies with poetry:
```sh
cd geoarches
poetry install
```

Poetry, by default, installs the geoarches package in editable mode.
Editable mode allows you to make changes to the geoarches code locally, and these changes will automatically be reflected in your code that depends on it.

## Contributing to geoarches

If you want to contribute to geoarches, please see the [Contributing](https://geoarches.readthedocs.io/en/latest/contributing/contribute/) section.

## External resources

Many thanks to the authors of WeatherLearn for adapting the Pangu-Weather pseudocode to pytorch. The code for our model is mostly based on their codebase.

[WeatherBench](https://sites.research.google/weatherbench/)

[WeatherLearn](https://github.com/lizhuoq/WeatherLearn/tree/master)