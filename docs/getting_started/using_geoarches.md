# Using geoarches in your project

After [installing `geoarches` from PyPI](./installation.md), create a working directory for
your scripts, data, models, and outputs. A source checkout is not required:

```text
my_project/
├── data/
├── modelstore/
├── evalstore/
├── wandblogs/
└── my_experiment.py
```

## Recommended usage

Use `geoarches` as an installed Python package from your own project, scripts, or notebooks.
Keeping your work separate from the library makes upgrading to new releases straightforward:

```sh
python -m pip install --upgrade geoarches
```

Import library modules directly in your code. For example, this creates the dataset for the
ERA5 weather forecasting task:

```python
from geoarches.dataloaders.era5 import Era5Forecast

ds = Era5Forecast(
    path="path/to/era5",
    load_prev=True,
    norm_file='pangu_norm_stats.nc',
)
```

Commands such as training and evaluation use paths relative to the current working directory
by default, so run them from `my_project/`.

For more information on how to use the library, explore the [User Guide](../user_guide/index.md).
For ArchesWeather-specific instructions, see the [ArchesWeather section](../archesweather/index.md).

## Editing the library

If you need to modify `geoarches`, install it from source and work on a development branch as
described in the [Contributing Guide](../contributing/index.md). Source installation is not
needed for normal use.
