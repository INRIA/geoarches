# Setup

### 1. Install the package

To get started, follow the [installation guide](../getting_started/installation.md) to install the package and all required dependencies.

!!! tip

    If you plan to modify the codebase, it's recommended to fork the repository first. You’ll find relevant setup steps in the [contributing section](../contributing/index.md).

### 2. Download pretrained models

The following script downloads four deterministic models (`archesweather-m-seed*`) and one generative model (`archesweathergen`) from Hugging Face:

```sh
src="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"
MODELS=("archesweather-m-seed0" "archesweather-m-seed1" "archesweather-m-skip-seed0" "archesweather-m-skip-seed1" "archesweathergen")

for MOD in "${MODELS[@]}"; do
    mkdir -p "modelstore/$MOD/checkpoints"
    wget -O "modelstore/$MOD/checkpoints/checkpoint.ckpt" "$src/${MOD}_checkpoint.ckpt"
    wget -O "modelstore/$MOD/config.yaml" "$src/${MOD}_config.yaml"
done
```

You can then follow the [notebook tutorial](./run.ipynb) to load the models and run inference. For training, refer to the [train section](./train.md).

### 3. Quantile statistics

ERA5 and HRES quantiles are required only when computing their corresponding Brier skill
scores. Geoarches downloads the requested file from a pinned revision of
[ArchesWeather on Hugging Face](https://huggingface.co/gcouairon/ArchesWeather) on first use
and reuses the Hugging Face cache afterward.

On compute nodes without network access, populate the cache before submitting a job:

```sh
python - <<'PY'
from geoarches.stats import QUANTILE_FILENAMES, resolve_quantiles_file

for filename in QUANTILE_FILENAMES:
    print(resolve_quantiles_file(filename))
PY
```

You can also pass an existing local file through the `quantiles_filepath` argument of
`Era5BrierSkillScore`.
