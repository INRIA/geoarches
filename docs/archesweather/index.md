# ArchesWeather and ArchesWeatherGen

[![arXiv](https://img.shields.io/badge/arXiv-2412.12971-b31b1b.svg)](https://arxiv.org/abs/2412.12971)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/INRIA/geoarches/blob/main/LICENSE)

## Overview

This section provides documentation, pretrained model checkpoints, and code examples for running, evaluating, and training **ArchesWeather** and **ArchesWeatherGen** for efficient machine learning weather forecasting.

Trained on $1.5^\circ \times 1.5^\circ$ ERA5 data, `geoarches` supports both deterministic and generative forecasting models:
- **ArchesWeather**: A fast, highly accurate deterministic neural network designed for efficient medium-range weather forecasting.
- **ArchesWeatherGen**: A probabilistic (generative) forecasting model built on flow matching, designed to capture atmospheric uncertainty and produce reliable weather ensembles.

---

## Paper & Citation

If you use ArchesWeather or ArchesWeatherGen in your research or project, please cite the following paper:

```bibtex
@misc{couairon2024archesweather,
      title={ArchesWeather & ArchesWeatherGen: a deterministic and generative model for efficient ML weather forecasting}, 
      author={Guillaume Couairon and Renu Singh and Anastase Charantonis and Christian Lessig and Claire Monteleoni},
      year={2024},
      eprint={2412.12971},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.12971}, 
}
```

---

## Available Models

We provide pretrained checkpoints and Hydra configurations for:
- **Deterministic Ensemble**: `archesweather-m-seed0`, `archesweather-m-seed1`, `archesweather-m-skip-seed0`, `archesweather-m-skip-seed1`
- **Generative Model**: `archesweathergen`

---

## Getting Started

- **[Setup Guide](./setup.md)**: Instructions for installing dependencies and downloading pretrained checkpoints.
- **[Run Tutorial](./run.ipynb)**: Interactive Jupyter notebook demonstrating inference and evaluation.
- **[Evaluation](./evaluate.md)** & **[Reproduce](./reproduce.md)**: Benchmarking and training instructions.

