# Setup

### 1. Install the package

To get started, follow the [installation guide](../getting_started/installation.md) to install the package and all required dependencies.

!!! tip

    If you plan to modify the codebase, it's recommended to fork the repository first. You’ll find relevant setup steps in the [contributing section](../contributing/index.md).

### 2. Download pretrained models

The following script downloads four deterministic models (`archesweather-m-seed*`) and one generative model (`archesweathergen`) from Hugging Face:
geoarches/download/dl_aw_models.sh

For each model, we download the pytorch checkpoint as well as the hydra config needed to evaluate the model.

You can then follow the [notebook tutorial](./run.ipynb) to load the models and run inference. For training, refer to the [train section](./train.md).

### 3. Download ERA5 quantile statistics

ERA5 quantiles are required to compute Brier scores and are used during both inference and training. Download them with:

```sh
src="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"
wget -O geoarches/stats/era5-quantiles-2016_2022.nc $src/era5-quantiles-2016_2022.nc
```
