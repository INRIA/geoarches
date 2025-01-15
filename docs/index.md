---
title: Home
hide:
  - navigation
---

<div style="display: flex; align-items: center;">
  <img src="assets/logo.png" style="height: 150px; width: 150px; margin-left: 1em, margin-right: 1em" alt="Logo">
  <div style="margin-left: 1em">
    <h1 style="margin-bottom: 0.8em;">geoarches documentation</h1>
    <p>If you're a user, start with the <a href="getting_started/installation">Getting Started</a> guide, then explore the <a href="user_guide">User Guide</a> for more detailed instructions and tips.<br>
    If you're interested in contributing to the project, check out the <a href="contributing">Contributing</a> for developer setup and guidelines.</p>
  </div>
</div>

## What is geoarches?

**geoarches** is a research-friendly machine learning library for training, running, and evaluating models on **geospatial data**, mainly weather and climate data.

Built on [PyTorch](https://pytorch.org), [Pytorch Lightning](https://lightning.ai), and [Hydra](https://hydra.cc), geoarches offers a clean, modular structure for developing and scaling ML pipelines. Once installed, you can use its modules inside your own project, or use the main training and evaluating workflows.

??? tip "geoarches powers _ArchesWeather_ and _ArchesWeatherGen_ models."

    See [ArchesWeather section](./archesweather/index.md) for more details.

## Overview

geoarches is meant to jumpstart your ML pipeline with building blocks for data handling, model training, and evaluation. This is an ongoing effort to share engineering tools and research knowledge across projects.

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

## Next steps

- **[Install geoarches](./getting_started/installation.md)**
- **[Explore the User Guide](./user_guide/index.md)**
- **[Contribute to development](./contributing/index.md)**
