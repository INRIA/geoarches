# adapted from https://github.com/lizhuoq/WeatherLearn/blob/master/weatherlearn/models/pangu/pangu.py

import torch
from timm.models.layers import DropPath, trunc_normal_
from torch import nn

from .weatherlearn_utils.crop import crop3d
from .weatherlearn_utils.earth_position_index import get_earth_position_index
from .weatherlearn_utils.pad import get_pad3d
from .weatherlearn_utils.shift_window_mask import (
    window_partition,
    window_reverse,
)


class Conv3dSimple(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        """
        a simple reimplementation of 3d convolution to be compatible with mps
        """
        super().__init__()
        self.kernel_size = kernel_size
        assert stride == kernel_size, "stride must be equal to kernel size"
        scale = (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]) ** -0.5 / 2
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size) * scale)
        self.bias = nn.Parameter(torch.randn(out_channels) * scale)

    def forward(self, x):
        """
        reshape x to be able to apply 3d convolution based on reshape and matmul
        """
        x = x.view(
            x.shape[0],
            x.shape[1],
            x.shape[2] // self.kernel_size[0],
            self.kernel_size[0],
            x.shape[3] // self.kernel_size[1],
            self.kernel_size[1],
            x.shape[4] // self.kernel_size[2],
            self.kernel_size[2],
        )
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)[..., None]  # matmul will append on dim -2
        x = (x * self.weight.permute(2, 3, 4, 1, 0)).sum(dim=(-5, -4, -3, -2))
        x = x + self.bias
        x = x.movedim(-1, 1)
        return x


# first, here is an implementation for subpixel convolution


# borrowed from
# https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
# def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
#     "tensor: the 2-dimensional Tensor or more"
#     upscale_factor_squared = upscale_factor * upscale_factor
#     assert tensor.shape[0] % upscale_factor_squared == 0, (
#         "The size of the first dimension: "
#         f"tensor.shape[0] = {tensor.shape[0]}"
#         " is not divisible by square of upscale_factor: "
#         f"upscale_factor = {upscale_factor}"
#     )
#     sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared, *tensor.shape[1:])
#     sub_kernel = initializer(sub_kernel, *args, **kwargs)
#     return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)


def ICNR_init(tensor, initializer, upscale_factor=2, *args, **kwargs):  # noqa N802
    "tensor: the 2-dimensional Tensor or more"
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, (
        "The size of the first dimension: "
        f"tensor.shape[0] = {tensor.shape[0]}"
        " is not divisible by square of upscale_factor: "
        f"upscale_factor = {upscale_factor}"
    )
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared, *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    new_tensor = sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)
    tensor.data.copy_(new_tensor)


class UpSample(nn.Module):
    """
    Up-sampling operation.
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, N, C)
        """
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[:, :out_pl, pad_top : 2 * in_lat - pad_bottom, pad_left : 2 * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x


class DownSample(nn.Module):
    """
    Down-sampling operation
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        pad_front = pad_back = 0

        self.pad = nn.ZeroPad3d((pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)

        # Padding the input to facilitate downsampling
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)

        x = self.norm(x)
        x = self.linear(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        **kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EarthAttention3D(nn.Module):
    """
    3D window attention with earth position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        window_size (tuple[int]): [pressure levels, latitude, longitude]
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wpl, Wlat, Wlon
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.type_of_windows = (input_resolution[0] // window_size[0]) * (
            input_resolution[1] // window_size[1]
        )

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1),
                self.type_of_windows,
                num_heads,
            )
        )  # Wpl**2 * Wlat**2 * Wlon*2-1, Npl//Wpl * Nlat//Wlat, nH

        earth_position_index = get_earth_position_index(
            window_size
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.earth_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: input features with shape of (B * num_lon, num_pl*num_lat, N, C)
            mask: (0/-inf) mask with shape of (num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon)
        """
        B_, nW_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows,
            -1,
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon, num_pl*num_lat, nH
        earth_position_bias = earth_position_bias.permute(
            3, 2, 0, 1
        ).contiguous()  # nH, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        attn = attn + earth_position_bias.unsqueeze(0)

        if mask is not None:
            nLon = mask.shape[0]
            attn = attn.view(B_ // nLon, nLon, self.num_heads, nW_, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EarthSpecificBlock(nn.Module):
    """
    3D Transformer Block
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=None,
        shift_size=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        axis_attn=False,
        roll_type=0,
        act_layer=nn.GELU,
        mlp_layer=Mlp,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 5) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.roll_type = roll_type

        self.norm1 = norm_layer(dim)
        padding = get_pad3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[-1] + padding[-2]
        pad_resolution[1] += padding[2] + padding[3]
        pad_resolution[2] += padding[0] + padding[1]

        self.attn = EarthAttention3D(
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            # latlon_features=input_resolution[1]*input_resolution[2],
            act_layer=act_layer,
            drop=drop,
        )

        shift_pl, shift_lat, shift_lon = self.shift_size
        # self.roll = shift_pl and shift_lon and shift_lat
        self.roll = roll_type > 0

        # if self.roll:
        #    attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
        # else:
        attn_mask = None

        if axis_attn:
            from axial_attention import AxialAttention, AxialPositionalEmbedding

            self.axis_pos = AxialPositionalEmbedding(dim=dim, shape=(8,), emb_dim_index=-1)
            self.axis_attn = AxialAttention(
                dim=dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=8,  # number of heads for multi-head attention
                num_dimensions=1,  # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out=True,  # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None, dt=1):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        assert L == Pl * Lat * Lon, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        # first modulation
        if c is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)
            x = x * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]

        x = x.view(B, Pl, Lat, Lon, C)

        # start pad
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        shift_pl, shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C

        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(
            x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C
        )
        # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = attn_windows.view(
            attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C
        )

        if self.roll:
            shifted_x = window_reverse(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            # B * Pl * Lat * Lon * C
            x = torch.roll(shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            x = shifted_x

        # crop, end pad
        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)

        x = x.reshape(B, Pl * Lat * Lon, C)

        # try axial attention here ?
        if hasattr(self, "axis_attn"):
            x2 = x.reshape(B, Pl, Lat * Lon, C).movedim(2, 1).flatten(0, 1)  # B*Lat*Lon, Pl, C
            x2 = self.axis_pos(x2)
            x2 = self.axis_attn(x2)
            x2 = x2.reshape(B, Lat * Lon, Pl, C).movedim(1, 2).flatten(1, 2)  # B, Pl*Lat*Lon, C

        if isinstance(dt, torch.Tensor):
            dt = dt[:, :, None]
        if c is None:
            x = shortcut + dt * self.drop_path(x)
            if hasattr(self, "axis_attn"):
                x = x + dt * self.drop_path(x2)
            x = x + dt * self.drop_path(self.mlp(self.norm2(x)))
        else:
            # already computed, just here for reference
            # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)
            if hasattr(self, "axis_attn"):
                x = x + dt * self.drop_path(x2)
            x = shortcut + gate_msa[:, None, :] * self.drop_path(x)
            mlp_input = self.norm2(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
            x = x + self.drop_path(gate_mlp[:, None, :] * self.mlp(mlp_input))
        return x


class BasicLayer(nn.Module):
    """A basic 3D Transformer layer for one stage

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                EarthSpecificBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    roll_type=(i % 2),  # 1 or 3
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    norm_layer=norm_layer,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, *args, **kwargs):
        for blk in self.blocks:
            x = blk(x, *args, **kwargs)
        return x


# conditional basic layer


class CondBasicLayer(BasicLayer):
    def __init__(self, *args, dim=192, cond_dim=32, **kwargs):
        super().__init__(*args, dim=dim, **kwargs)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        # init the modulation

    def forward(self, x, cond_emb=None):
        c = self.adaLN_modulation(cond_emb)
        return super().forward(x, c)


class LinVert(nn.Module):
    """
    a modification of Mlp that takes the full column
    """

    def __init__(self, in_features, drop=0.0, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(8 * in_features, 8 * in_features)

    def forward(self, x: torch.Tensor):
        shortcut = x
        x2 = (
            shortcut.reshape((shortcut.shape[0], 8, -1, shortcut.shape[-1]))
            .movedim(1, -2)
            .flatten(-2, -1)
        )  # B, lat*lon, 8*C
        x2 = self.fc1(x2)
        x2 = (
            x2.reshape((x2.shape[0], -1, 8, shortcut.shape[-1])).movedim(-2, 1).flatten(1, 2)
        )  # B, 8*lat*lon, C

        return shortcut + x2
