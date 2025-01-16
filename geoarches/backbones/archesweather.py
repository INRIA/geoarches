import importlib

import geoarches.stats as geoarches_stats
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torch.utils.checkpoint as gradient_checkpoint
from geoarches.backbones.archesweather_layers import ICNR_init
from tensordict.tensordict import TensorDict
from timm.layers.mlp import SwiGLU

from .archesweather_layers import (
    CondBasicLayer,
    DownSample,
    LinVert,
    Mlp,
    UpSample,
)


class WeatherEncodeDecodeLayer(nn.Module):
    """
    gathers layers for the encoder and decoder
    """

    def __init__(
        self,
        img_size=(13, 121, 240),
        emb_dim=192,
        out_emb_dim=2 * 192,  # because of skip
        patch_size=(2, 2, 2),
        surface_ch=4,
        level_ch=6,
        n_concatenated_states=0,
        final_interpolation=False,
    ) -> None:
        super().__init__()
        self.__dict__.update(locals())

        geoarches_stats_path = importlib.resources.files(geoarches_stats)
        self.constant_masks = torch.load(
            geoarches_stats_path / "archesweather_constant_masks.pt", weights_only=True
        )
        constant_dims = self.constant_masks.shape[0]

        surface_ch_in = constant_dims + surface_ch + n_concatenated_states * surface_ch
        level_ch_in = level_ch + n_concatenated_states * level_ch

        if torch.backends.mps.is_available():
            from .archesweather_layers import Conv3dSimple

            self.level_proj = Conv3dSimple(
                level_ch_in, emb_dim, kernel_size=patch_size, stride=patch_size
            )
        else:
            self.level_proj = nn.Conv3d(
                level_ch_in, emb_dim, kernel_size=patch_size, stride=patch_size
            )
        self.surface_proj = nn.Conv2d(
            surface_ch_in, emb_dim, kernel_size=patch_size[1:], stride=patch_size[1:]
        )

        l_pad = patch_size[0] - img_size[0] % patch_size[0]
        level_pads = [l_pad // 2, l_pad - l_pad // 2]

        self.level_padder = nn.ZeroPad3d((0, 0, 0, 0, *level_pads))

        # decode layers

        self.surface_deconv = nn.Conv2d(
            out_emb_dim,
            surface_ch * patch_size[-1] ** 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=0,
        )
        self.level_deconv = nn.Conv2d(
            out_emb_dim // 2,
            level_ch * patch_size[-1] ** 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=0,
        )
        self.pixelshuffle = nn.PixelShuffle(patch_size[-1])
        ICNR_init(
            self.surface_deconv.weight,
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_size[-1],
        )
        ICNR_init(
            self.level_deconv.weight,
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_size[-1],
        )

    def encode(self, state: TensorDict, cond_state: TensorDict = None):
        """
        cond state is a condition state that is concatenated to the main state
        """
        bs = state.shape[0]
        device = state.device
        if self.constant_masks.device != device:
            self.constant_masks = self.constant_masks.to(device)

        # embed
        level = state["level"]
        surface = state["surface"].squeeze(-3)

        # remove south pole if necessary
        if surface.shape[-2] % 2:
            surface = surface[..., :-1, :]
            level = level[..., :-1, :]

        constant = self.constant_masks[None, :, 0].expand((bs, -1, -1, -1))

        surface = torch.cat([surface, constant], dim=1)

        if cond_state is not None:
            cond_surface = cond_state["surface"].squeeze(-3)
            cond_level = cond_state["level"]

            if cond_state["surface"].shape[-2] % 2:
                cond_surface = cond_surface[..., :-1, :]
                cond_level = cond_level[..., :-1, :]
            surface = torch.cat([surface, cond_surface], dim=1)
            level = torch.cat([level, cond_level], dim=1)

        surface = self.surface_proj(surface)
        level = self.level_proj(self.level_padder(level))

        x = torch.concat([surface.unsqueeze(2), level], dim=2)
        return x

    def decode(self, x):
        surface, level = x[:, :, 0], x[:, :, 1:]

        output_surface = self.surface_deconv(surface)
        output_surface = self.pixelshuffle(output_surface)
        output_surface = output_surface.unsqueeze(-3)

        level = level.reshape(level.shape[0], level.shape[1] // 2, 2, *level.shape[2:]).flatten(
            2, 3
        )[:, :, 1:]
        level = level.movedim(-3, 1).flatten(0, 1)

        output_level = self.level_deconv(level)
        output_level = self.pixelshuffle(output_level)
        output_level = output_level.reshape(-1, self.img_size[0], *output_level.shape[1:]).movedim(
            1, -3
        )

        if self.final_interpolation:
            bs = output_surface.shape[0]
            output_level = F.interpolate(
                output_level.flatten(0, 1), size=self.img_size[1:], mode="bilinear"
            )
            output_level = output_level.reshape(bs, -1, *output_level.shape[1:])
            output_surface = F.interpolate(
                output_surface.flatten(0, 1), size=self.img_size[1:], mode="bilinear"
            )
            output_surface = output_surface.reshape(bs, -1, *output_surface.shape[1:])
        else:
            # put back fake south pole
            output_surface = torch.cat([output_surface, output_surface[..., -1:, :]], dim=-2)
            output_level = torch.cat([output_level, output_level[..., -1:, :]], dim=-2)

        return TensorDict(
            surface=output_surface,
            level=output_level,
            batch_size=output_surface.shape[0],
        ).to(x.device)


class ArchesWeatherCondBackbone(nn.Module):
    def __init__(
        self,
        tensor_size=(8, 60, 120),
        emb_dim=192,
        cond_dim=256,  # dim of the conditioning
        num_heads=(6, 12, 12, 6),
        window_size=(1, 6, 10),
        droppath_coeff=0.2,
        depth_multiplier=2,
        dropout=0.0,
        mlp_ratio=4.0,
        use_skip=True,
        first_interaction_layer="linear",
        gradient_checkpointing=False,
        mlp_layer="mlp",
        **kwargs,
    ):
        super().__init__()
        self.__dict__.update(locals())
        drop_path = np.linspace(
            0, droppath_coeff / depth_multiplier, 8 * depth_multiplier
        ).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        self.zdim = tensor_size[0]

        self.layer1_shape = tensor_size[1:]

        self.layer2_shape = (self.layer1_shape[0] // 2, self.layer1_shape[1] // 2)

        if first_interaction_layer == "linear":
            self.interaction_layer = LinVert(in_features=emb_dim)

        layer_args = dict(
            cond_dim=cond_dim,
            window_size=window_size,
            act_layer=nn.GELU,
            drop=dropout,
            mlp_layer=Mlp,
            mlp_ratio=mlp_ratio,
        )

        if mlp_layer == "swiglu":
            layer_args["mlp_ratio"] = mlp_ratio * 2 / 3
            layer_args["mlp_layer"] = SwiGLU

        self.layer1 = CondBasicLayer(
            dim=emb_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2 * depth_multiplier,
            num_heads=num_heads[0],
            drop_path=drop_path[: 2 * depth_multiplier],
            **layer_args,
            **kwargs,
        )
        self.downsample = DownSample(
            in_dim=emb_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            output_resolution=(self.zdim, *self.layer2_shape),
        )
        self.layer2 = CondBasicLayer(
            dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6 * depth_multiplier,
            num_heads=num_heads[1],
            drop_path=drop_path[2 * depth_multiplier :],
            **layer_args,
            **kwargs,
        )
        self.layer3 = CondBasicLayer(
            dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6 * depth_multiplier,
            num_heads=num_heads[2],
            drop_path=drop_path[2 * depth_multiplier :],
            **layer_args,
            **kwargs,
        )
        self.upsample = UpSample(
            emb_dim * 2, emb_dim, (self.zdim, *self.layer2_shape), (self.zdim, *self.layer1_shape)
        )
        out_dim = emb_dim if not self.use_skip else 2 * emb_dim
        self.layer4 = CondBasicLayer(
            dim=out_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2 * depth_multiplier,
            num_heads=num_heads[3],
            drop_path=drop_path[: 2 * depth_multiplier],
            **layer_args,
            **kwargs,
        )

    def forward(self, x, cond_emb, **kwargs):
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        if self.first_interaction_layer:
            x = self.interaction_layer(x)

        x = self.layer1(x, cond_emb)

        skip = x
        x = self.downsample(x)

        x = self.layer2(x, cond_emb)

        if self.gradient_checkpointing:
            x = gradient_checkpoint.checkpoint(self.layer3, x, cond_emb, use_reentrant=False)
        else:
            x = self.layer3(x, cond_emb)

        x = self.upsample(x)
        if self.use_skip and skip is not None:
            x = torch.concat([x, skip], dim=-1)
        x = self.layer4(x, cond_emb)

        output = x
        output = output.transpose(1, 2).reshape(output.shape[0], -1, 8, *self.layer1_shape)

        return output
