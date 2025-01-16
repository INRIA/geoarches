import torch

lat_coeffs_equi = torch.tensor(
    [torch.cos(x) for x in torch.arange(-torch.pi / 2, torch.pi / 2, torch.pi / 120)]
)
lat_coeffs_equi = (lat_coeffs_equi / lat_coeffs_equi.mean())[None, None, :, None]


def acc(x, y, z=0):
    """Anomaly correlation coefficient.

    Expects inputs of shape: [..., lat, lon]

    Args:
        x: predictions
        y: targets
        z: climatology
    """
    assert x.shape[-2] == 120, "Wrong shape for ACC computation"
    coeffs = lat_coeffs_equi.to(x.device)[None]
    x = x - z
    y = y - z
    norm1 = (x * x).mul(coeffs).mean((-2, -1)) ** 0.5
    norm2 = (y * y).mul(coeffs).mean((-2, -1)) ** 0.5
    mean_acc = (x * y).mul(coeffs).mean((-2, -1)) / norm1 / norm2
    return mean_acc


def wrmse(x, y):
    """Weighted root mean square error.

    Expects inputs of shape: [..., lat, lon]

    Args:
        x: predictions
        y: targets
    """
    assert x.shape[-2] == 120, "Wrong shape for WRMSE computation"
    coeffs = lat_coeffs_equi.to(x.device)
    err = (x - y).pow(2).mul(coeffs).mean((-2, -1)).sqrt()
    return err


def headline_wrmse(pred, batch, prefix=""):
    """RMSE for the top variables in WeatherBench.

    Input shape should be (batch, leadtime, var, level, lat, lon)

    Args:
        pred: Dictionary with surface and level tensors for predictions.
        batch: Dictionary with surface and level tensors for targets.
        prefix: string prefix for the keys of the surface and level tensors in `pred` and `batch`.

    """
    assert prefix + "_level" in batch, prefix + "_level not in batch"
    assert prefix + "_surface" in batch, prefix + "_surface not in batch"

    surface_wrmse = wrmse(pred[prefix + "_surface"], batch[prefix + "_surface"])
    level_wrmse = wrmse(pred[prefix + "_level"], batch[prefix + "_level"])

    metrics = dict(
        T2m=surface_wrmse[..., 2, 0],
        SP=surface_wrmse[..., 3, 0],
        U10=surface_wrmse[..., 0, 0],
        V10=surface_wrmse[..., 1, 0],
        Z500=level_wrmse[..., 0, 7],
        T850=level_wrmse[..., 3, 10],
        Q700=1000 * level_wrmse[..., 4, 9],
        U850=level_wrmse[..., 1, 10],
        V850=level_wrmse[..., 2, 10],
    )

    return metrics
