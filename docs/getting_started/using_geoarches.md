# Using geoarches in your project

After completing the [installation](./installation.md), your working directory should look like this:

```
├── geoarches/       # Cloned repo
│   ├── geoarches/
│   ├── ...
└── my_project/      # Your own scripts, notebooks, experiments...
    ├── ...
```

## Recommended usage

We recommend using `geoarches` as a Python package from your own working directory or scripts, **without modifying the library code directly**. This allows you to cleanly separate your own work from the library itself, and makes it easier to update `geoarches` when new features or bug fixes are released.

You can import the library modules directly in your scripts or notebooks:

!!! example

    Thiw will import the dataset for the ERA5 weather forecast task:

    ```python
    from geoarches.dataloaders.era5 import Era5Forecast

    ds = Era5Forecast(
        path='path/to/era5',
        load_prev=True,
        norm_scheme='pangu',
    )
    ```

## Editing the library

If you do need to modify the library (e.g. to experiment with architectural changes), we recommend working on a **development branch**.
This makes it easier to rebase or merge when upstream updates are available. See the [Contributing Guide](../contributing/index.md) for details.

---

For more information on how to use the library, explore the [User Guide](../user_guide/index.md).
