# ------------------------------------------------------------------
"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces
https://github.com/state-spaces/mamba

Built upon Video Swin Transformer
https://github.com/SwinTransformer/Video-Swin-Transformer

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from functools import reduce
from operator import mul
from einops import rearrange

from mamba_ssm import Mamba as Mamba_v1

# --------------------------------------------------------


class Mlp(nn.Module):
    """ Multilayer perceptron """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class MambaBlock(nn.Module):
    def __init__(self, dim, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., d_state=1, expand=1, d_conv=3, dt_min=0.01, dt_max=0.1,
                 drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_min = dt_min
        self.dt_max = dt_max

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim, elementwise_affine=False)

        self.ssm = Mamba_v1(
            d_model=dim,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            dt_min=dt_min,
            dt_max=dt_max
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, elementwise_affine=False)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x):

        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = x.view(B, D * H * W, C)
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.ssm(x_windows)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x)
        else:
            x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        pad_D = (D % 2 == 1) and D != 1
        if pad_D:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, D % 2))
        if D == 1:
            x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
            x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
            x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
            x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        else:
            x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
            x1 = x[:, 1::2, 1::2, 0::2, :]  # B D/2 H/2 W/2 C
            x2 = x[:, 0::2, 0::2, 1::2, :]  # B D/2 H/2 W/2 C
            x3 = x[:, 1::2, 1::2, 1::2, :]  # B D/2 H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Mamba layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 in_dim,
                 patch_size,
                 dim,
                 depth,
                 d_state=1,
                 expand=1,
                 d_conv=3,
                 dt_min=.01,
                 dt_max=0.1,
                 window_size=(4, 4, 4),
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 patch_norm=None,
                 use_checkpoint=False):
        super().__init__()

        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.in_dim = in_dim

        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_min = dt_min
        self.dt_max = dt_max

        # build blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=dim,
                window_size=window_size,  # if i != 2 else (8, 1, 1),
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
                dt_min=dt_min,
                dt_max=dt_max,
            )
            for i in range(depth)])

        if in_dim != dim or patch_size != (1, 1, 1):
            self.downsample = downsample(patch_size=patch_size, in_chans=in_dim, embed_dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        if self.downsample is not None:
            x = self.downsample(x)

        B, C, D, H, W = x.shape

        x = rearrange(x, 'b c d h w -> b d h w c')
        for blk in self.blocks:
            x = blk(x)

        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(2, 4, 4), in_chans=16, embed_dim=64, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim, elementwise_affine=False)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww

        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        return x


class Mamba(nn.Module):
    """ Simple replacement of Swin Transformer backbone with Mamba
        Mamba: Linear-Time Sequence Modeling with Selective State Spaces
        https://github.com/state-spaces/mamba
    """
    def __init__(self,
                 in_vars: int = 6,
                 in_chans: int = 1,
                 embed_dim: list = None,
                 window_size: list = None,
                 depths: list = None,
                 mlp_ratio: int = 4,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 patch_size: tuple = (1, 1, 1),
                 patch_norm: bool = False,
                 use_checkpoint: bool = False,
                 d_state: list = None,
                 d_conv: list = None,
                 expand: list = None,
                 dt_min: float = 0.01,
                 dt_max: float = 0.1
                 ):
        """
        Args:
            in_vars (int, optional): number of input variables. Defaults to 6
            in_chans (int, optional): number of input channels. Defaults to 1
            embed_dim (list, optional): embedding dimension per layer. Defaults to None
            window_size (list, optional): window size per layer. Defaults to None
            depths (list, optional): number of blocks per layer. Defaults to None
            mlp_ratio (int, optional): ratio of mlp hidden dim to embedding dim. Defaults to 4
            drop_rate (float, optional): dropout rate. Defaults to 0.
            drop_path_rate (float, optional): dropout rate. Defaults to 0.
            patch_size (tuple, optional): patch token size. Defaults to (1, 1, 1)
            patch_norm (bool, optional): whether to use patch norm. Defaults to False
            use_checkpoint (bool, optional): whether to use checkpoint. Defaults to False
            d_state (list, optional): SSM state expansion factor per layer. Defaults to None
            d_conv (list, optional): SSM local convolution width per layer. Defaults to None
            expand (list, optional): SSM d_inner expansion factor per layer. Defaults to None
            dt_min (float, optional): SSM dt_min. Defaults to 0.01
            dt_max (float, optional): SSM dt_max. Defaults to 0.1
        """
        super().__init__()

        self.in_vars = in_vars
        self.in_chans = in_chans
        self.embed_dim = embed_dim if embed_dim is not None else [16, 16]
        self.num_layers = len(self.embed_dim)
        self.window_size = window_size if window_size is not None else [(2, 4, 4), (8, 1, 1)]
        self.depths = depths if depths is not None else [2, 1]
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.patch_norm = patch_norm
        self.use_checkpoint = use_checkpoint
        self.d_state = d_state if d_state is not None else [1, 1]
        self.d_conv = d_conv if d_conv is not None else [3, 3]
        self.expand = expand if expand is not None else [1, 1]
        self.dt_min = dt_min
        self.dt_max = dt_max

        norm_layer = nn.LayerNorm  # fixed norm layer

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        # build layers
        self.layers_var, self.proj_var = nn.ModuleList(), nn.ModuleList()

        for _ in range(self.in_vars):
            layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(
                    in_dim=self.embed_dim[i_layer - 1] if i_layer > 0 else self.in_chans,
                    patch_size=self.patch_size if i_layer == 0 else (1, 1, 1),
                    dim=self.embed_dim[i_layer],
                    depth=self.depths[i_layer],
                    d_state=self.d_state[i_layer],
                    d_conv=self.d_conv[i_layer],
                    expand=self.expand[i_layer],
                    dt_min=self.dt_min,
                    dt_max=self.dt_max,
                    window_size=self.window_size[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=self.drop_rate,
                    drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchEmbed3D,  # if i_layer != 0 else None,
                    patch_norm=norm_layer if self.patch_norm and i_layer == 0 else None,
                    use_checkpoint=self.use_checkpoint)

                layers.append(layer)

            self.layers_var.append(layers)

            self.proj_var.append(nn.Sequential(nn.Conv3d(self.embed_dim[-1], self.embed_dim[-1], kernel_size=3, stride=1,
                                                         padding=1, padding_mode='replicate', bias=True),
                                               nn.ReLU(),
                                               nn.Conv3d(self.embed_dim[-1], self.embed_dim[-1], kernel_size=3, stride=1,
                                                         padding=1, padding_mode='replicate', bias=True),
                                               )
                                 )

        self.init_weights()

    def init_weights(self):
        """ Initialize the weights in backbone """

        def _init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv3d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """
        Forward function
        Args:
            x (torch.tensor): input dynamic variables [N, V, C, D, H, W]
        Returns:
            x_all (torch.tensor): output dynamic variables [N, V, out_channels[-1], D, H, W]
        """

        x_all = []

        for var in range(self.in_vars):
            x_v = x[:, var, :, :, :, :]

            for layer in range(len(self.layers_var[var])):
                x_v = self.layers_var[var][layer](x_v)

            x_v = self.proj_var[var](x_v)
            x_all.append(x_v.unsqueeze(1))

        x_all = torch.cat(x_all, dim=1)

        return x_all


if __name__ == '__main__':

    device = 'cuda'

    test_dynamic = torch.randn((1, 6, 1, 8, 100, 100), device=device)
    model = Mamba().to(device)
    print(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    for i in range(1):
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        test = model(test_dynamic)
        print(test.shape)



