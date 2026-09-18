# AIMIP: ArchesWeather and ArchesWeatherGen for Climate Simulations

[![arXiv:2605.29976](https://img.shields.io/badge/arXiv-2605.29976-b31b1b.svg)](https://arxiv.org/abs/2605.29976)
[![arXiv:2605.06944](https://img.shields.io/badge/arXiv-2605.06944-b31b1b.svg)](https://arxiv.org/abs/2605.06944)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/INRIA/geoarches/blob/main/LICENSE)

## Overview

This section provides documentation, pretrained model checkpoints, and code examples for running **ArchesWeather** and **ArchesWeatherGen** adapted for multi-decadal climate simulations under the **AI Model Intercomparison Project (AIMIP) Phase 1** protocol.

The AIMIP protocol establishes a standardized framework for systematically evaluating and comparing AI-based weather and climate models against other AI systems and traditional physics-based models.

While ArchesWeather and ArchesWeatherGen were originally designed for short-term weather forecasting (up to 10-day lead times), these models are adapted for climate simulations by training them from scratch on 1° × 1° ERA5 data conditioned on monthly forcings (sea surface temperature and sea ice cover). They are then rolled out from October 1979 to January 2025 using prescribed monthly forcings from ERA5.

---

## Paper & Citation

If you use the versions of ArchesWeather or ArchesWeatherGen adapted for AIMIP climate projections in your research or project, please cite our paper:

```bibtex
@article{singh2026archesweather,
  title={Evaluating Skill and Stability of ArchesWeather and ArchesWeatherGen under Multi-Decadal Climate Simulations},
  author={Singh, Renu and Brunstein, Robert and Jost, Antonia and Rackow, Thomas and Monteleoni, Claire and Hasson, Yana and Lessig, Christian and Couairon, Guillaume},
  journal={arXiv preprint arXiv:2605.29976},
  year={2026}
}
```

Additionally, please cite the general AIMIP Phase 1 protocol paper:

```bibtex
@article{henn2026aimip,
  title={AIMIP Phase 1: Systematic Evaluations of AI Weather and Climate Models},
  author={Henn, Brian and Bretherton, Christopher S. and Koldunov, Nikolay and Lessig, Christian and Molina, Maria J. and Arcomano, Troy and Watt-Meyer, Oliver and Couairon, Guillaume and Singh, Renu and Brunstein, Robert and Hasson, Yana and Jost, Antonia and Brenowitz, Noah and Manshausen, Peter and Cresswell-Clay, Nathaniel and Durran, Dale and Hall, Kyle Joseph Chen and Yuval, Janni and Kochkov, Dmitrii and Hoyer, Stephan and Lopez-Gomez, Ignacio},
  journal={arXiv preprint arXiv:2605.06944},
  year={2026}
}
```

---

## Available Models

We provide pretrained checkpoints and Hydra configurations for:
- **Deterministic Ensemble**: `aimip-archesweather-m-seed0`, `aimip-archesweather-m-seed1`, `aimip-archesweather-m-seed2`, `aimip-archesweather-m-seed3`
- **Generative Model**: `aimip-archesweathergen`

---

## Getting Started

- **[Setup Guide](./setup.md)**: Instructions for installing dependencies and downloading pretrained checkpoints.
- **[Run Tutorial](./run.ipynb)**: Interactive Jupyter notebook demonstrating model loading, forcing configuration, and multi-decadal inference.