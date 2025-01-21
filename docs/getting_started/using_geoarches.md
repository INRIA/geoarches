# Using geoarches modules in python

Your directory structure should look like this after installation:
```
├── geoarches
│   ├── geoarches
│   │   ├── ...
└── your_own_project
    ├── ...
```

The recommended way to use the package is to depend on the package inside your own working directory, by importing them in your project code e.g.

```python
from geoarches.dataloaders.era5 import Era5Forecast
ds = Era5Foreacast(path='data/era5_240/full',
                   load_prev=True,
                   norm_scheme='pangu')
```

Making edits directly in the geoarches package will make updates more difficult, but if you prefer this option, you can create a development branch so as to rebase it on future updates of geoarches. (See [Contributing](contributing.md) section).

See [User Guide](../user_guide.md) for detailed information.