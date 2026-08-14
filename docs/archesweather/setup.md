# Setup

### 1. Install the package

To get started, if not already done, follow the [installation guide](../getting_started/installation.md) to install the package with all required dependencies and download the data.

!!! tip

    If you plan to modify the codebase, it's recommended to fork the repository first. You’ll find relevant setup steps in the [contributing section](../contributing/index.md).

### 2. Download pretrained models

From your project directory, the following command downloads four deterministic models
(`archesweather-m-seed*`) and one generative model (`archesweathergen`) from Hugging Face into
`./modelstore/`:

```sh
python -m geoarches.download.dl_aw_models --output-directory ./modelstore
```

This works with both PyPI and source installations. For each model, it downloads the PyTorch
checkpoint, adds the metadata required by PyTorch Lightning when needed, and installs the
version-matched Hydra config used for evaluation. Existing files are reused. The five
checkpoints require approximately 3.3 GB of disk space.

You can then follow the [notebook tutorial](./run.ipynb) to load the models and run inference. To train the models from scratch, refer to the [reproduce section](./reproduce.md).

### 3. Quantile statistics

ERA5 and HRES quantiles are required only when computing their corresponding Brier skill
scores. Geoarches downloads the requested file from a pinned revision of
[ArchesWeather on Hugging Face](https://huggingface.co/gcouairon/ArchesWeather) on first use
and reuses the Hugging Face cache afterward.

To prepare for offline use, run this command while internet access is available. It downloads
both files into the Hugging Face cache and prints their local paths:

```sh
python -c 'from geoarches.stats import QUANTILE_FILENAMES, resolve_quantiles_file; print(*(resolve_quantiles_file(filename) for filename in QUANTILE_FILENAMES), sep="\n")'
```

Alternatively, download only the file you need to a custom location:

```sh
curl --fail --location \
  --output /shared/path/era5-quantiles-2016_2022.nc \
  https://huggingface.co/gcouairon/ArchesWeather/resolve/main/era5-quantiles-2016_2022.nc
```

Replace `era5` with `hres` in both filenames for HRES quantiles. Then, in the Python code,
pass the downloaded path to the metric:

```python
from geoarches.metrics.brier_skill_score import Era5BrierSkillScore

metric = Era5BrierSkillScore(
    quantiles_filepath="/shared/path/era5-quantiles-2016_2022.nc"
)
```

An existing custom path is used directly and does not require internet access.
