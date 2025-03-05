import torch

lat_coeffs_equi = torch.tensor(
    [torch.cos(x) for x in torch.arange(-torch.pi / 2, torch.pi / 2 + 1e-6, torch.pi / 120)]
)
lat_coeffs_equi = (lat_coeffs_equi / lat_coeffs_equi.mean())[None, None, :, None]


def wrmse(pred, gt, weights=None):
    """Weighted root mean square error.

    Expects inputs of shape: [..., lat, lon]

    Args:
        pred: predictions
        gt: targets
        weights: weights for the latitudes
    """
    if weights is None:
        weights = lat_coeffs_equi.to(pred.device)

    err = (pred - gt).pow(2).mul(weights).nanmean((-2, -1)).sqrt()
    return err


def headline_wrmse(pred, gt, denormalize_function=None):
    """RMSE for the top variables in WeatherBench.

    Input shape should be (batch, leadtime, var, level, lat, lon)

    Args:
        pred: TensorDict with surface and level tensors for predictions.
        batch: TensorDict with surface and level tensors for targets.
        prefix: string prefix for the keys of the surface and level tensors in `pred` and `batch`.

    """
    pred = denormalize_function(pred)
    gt = denormalize_function(gt)

    surface_wrmse = wrmse(pred["surface"], gt["surface"])
    level_wrmse = wrmse(pred["level"], gt["level"])

    metrics = dict(
        T2m=surface_wrmse[..., 2, 0],
        SP=surface_wrmse[..., 3, 0],
        U10m=surface_wrmse[..., 0, 0],
        V10m=surface_wrmse[..., 1, 0],
        Z500=level_wrmse[..., 0, 7],
        T850=level_wrmse[..., 3, 10],
        Q700=1000 * level_wrmse[..., 4, 9],
        U850=level_wrmse[..., 1, 10],
        V850=level_wrmse[..., 2, 10],
    )

    return metrics
