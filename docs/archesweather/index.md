# ArchesWeather and ArchesWeatherGen

[![arXiv](https://img.shields.io/badge/arXiv-2412.12971-b31b1b.svg)](https://arxiv.org/abs/2412.12971)
[![Science Advances](https://img.shields.io/badge/Science%20Advances-adx2372-b31b1b.svg)](https://www.science.org/doi/10.1126/sciadv.adx2372)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/INRIA/geoarches/blob/main/LICENSE)

## Overview

This section provides documentation, pretrained model checkpoints, and code examples for running, evaluating, and training **ArchesWeather** and **ArchesWeatherGen** for efficient machine learning weather forecasting.

Trained on 1.5° × 1.5° ERA5 data, `geoarches` supports both deterministic and generative forecasting models:
- **ArchesWeather**: A fast, highly accurate deterministic neural network designed for efficient medium-range weather forecasting.
- **ArchesWeatherGen**: A probabilistic (generative) forecasting model built on flow matching, designed to capture atmospheric uncertainty and produce reliable weather ensembles.

---

## Paper & Citation

If you use ArchesWeather or ArchesWeatherGen in your research or project, please cite the following paper:

```bibtex
@article{couairon2026archesweathergen,
  title={ArchesWeatherGen: Skillful and compute-efficient probabilistic weather forecasting with machine learning},
  author={Couairon, Guillaume and Singh, Renu and Charantonis, Anastase and Lessig, Christian and Monteleoni, Claire},
  journal={Science Advances},
  volume={12},
  number={17},
  pages={eadx2372},
  year={2026},
  publisher={American Association for the Advancement of Science}
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

