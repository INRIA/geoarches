<div style="display: flex; align-items: flex-start;">
<img src="img/logo.png" style="height: 150px; width: 150px; margin-right: 20px;" alt="Logo">

<div>
    <h1>Geoarches Documentation</h1>
    <p>If you are a user, check <a href="getting_started/installation">Getting Started</a>, then <a href="user_guide">User Guide</a> for more information.</p>
    <p>If you want to contribute to the codebase, check the <a href="contributing/contribute">Contributing</a> section for developer setup and instructions.</p>
</div>
</div>

## What is geoarches ?

**geoarches** is a machine learning package for training, running and evaluating ML models on geospatial data, mainly weather and climate data.

geoarches's building blocks can be easily integrated into research ML pipelines.
It can also be used to run the **ArchesWeather** and **ArchesWeatherGen** weather models.

geoarches is based on pytorch, pytorch-lightning and hydra for configuration. After the package is installed, you can use its modules in python code, but you can also call the main training and evaluating scripts of geoarches.


## Overview

geoarches is meant to jumpstart your ML pipeline with building blocks for data handling, model training, and evaluation. This is an effort to share engineering tools and research knowledge across projects.

### Data
- `download/`: scripts that parallelize downloads and show how to use chunking to speed up read access.
- `dataloaders/`: PyTorch datasets that read netcdf data and prepare tensors to feed into model training.

### Model training
- `backbones/`: network architecture that can be plugged into lightning modules.
- `lightning_modules/`: wrapper around backbone modules to handle loss computation, optimizer, etc for training and inference (agnostic to backbone but specific to ML task).

### Evaluation
- `metrics/`: tested suite of iterative metrics (memory efficient) for deterministic and generative models.
- `evaluation/`: scripts for running metrics over model predictions and plotting.

### Pipeline
- `main_hydra.py`: script to run training or inference with hydra configuration.
- `docs/archesweather/`: quickstart code for training and inference from a notebook.