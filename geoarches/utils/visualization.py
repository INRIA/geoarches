import torch


def bw_to_bwr(bw_tensor, m=None, M=None):  # noqa N803
    x = bw_tensor
    if m is None:
        x = x - x.min()
        x = x / x.max()
    else:
        x = (x - m) / (M - m)
    red = torch.tensor([1, 0, 0])[:, None, None].to(x.device)
    white = torch.tensor([1, 1, 1])[:, None, None].to(x.device)
    blue = torch.tensor([0, 0, 1])[:, None, None].to(x.device)
    x_blue = blue + 2 * x * (white - blue)
    x_red = white + (2 * x - 1) * (red - white)
    x_bwr = x_blue * (x < 0.5) + x_red * (x >= 0.5)
    x_bwr = (x_bwr * 255).int().permute((1, 2, 0))
    x_bwr = x_bwr.cpu().numpy().astype("uint8")
    return x_bwr
