# Setup

### 1. Install the package

To get started, if not already done, follow the [installation guide](../getting_started/installation.md) to install the package with all required dependencies and download the data.

!!! tip

    If you plan to modify the codebase, it's recommended to fork the repository first. You’ll find relevant setup steps in the [contributing section](../contributing/index.md).

### 2. Download pretrained models

From your project directory, the following command downloads four deterministic models
(`aimip-archesweather-m-seed*`) and one generative model (`aimip-archesweathergen`) from Zenodo into
`./modelstore/`:

```sh
python -m geoarches.download.dl_aimip_models --output-directory ./modelstore
```

This works with both PyPI and source installations. For each model, it downloads the PyTorch
checkpoint, adds the metadata required by PyTorch Lightning when needed, and installs the
version-matched Hydra config used for evaluation. Existing files are reused. The five
checkpoints require approximately 2 GB of disk space.

You can then follow the [notebook tutorial](./run.ipynb) to load the models and run inference.