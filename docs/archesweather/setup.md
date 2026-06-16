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

### 3. Quantile statistics

ERA5 and HRES quantiles are required only when computing their corresponding Brier skill
scores. Geoarches downloads the requested file from a pinned revision of
[ArchesWeather on Hugging Face](https://huggingface.co/gcouairon/ArchesWeather) on first use
and reuses the Hugging Face cache afterward.

To prepare for offline use, download the files in the cache while internet access is available:

```sh
python - <<'PY'
from geoarches.stats import QUANTILE_FILENAMES, resolve_quantiles_file

for filename in QUANTILE_FILENAMES:
    print(resolve_quantiles_file(filename))
PY
```

Alternatively, download only the file you need to a custom location:

```sh
curl --fail --location \
  --output /shared/path/era5-quantiles-2016_2022.nc \
  https://huggingface.co/gcouairon/ArchesWeather/resolve/main/era5-quantiles-2016_2022.nc
```

Replace `era5` with `hres` in both filenames for HRES quantiles. Then pass the downloaded
path to the metric:

```python
from geoarches.metrics.brier_skill_score import Era5BrierSkillScore

metric = Era5BrierSkillScore(
    quantiles_filepath="/shared/path/era5-quantiles-2016_2022.nc"
)
```

An existing custom path is used directly and does not require internet access.
