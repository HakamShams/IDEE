# ------------------------------------------------------------------
"""
3D CNN Encoder

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

# ------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """ Video to patch embedding https://github.com/SwinTransformer/Video-Swin-Transformer """

    def __init__(self, patch_size: tuple = (2, 4, 4),
                 in_chans: int = 16,
                 embed_dim: int = 64,
                 norm_layer: nn.Module = None):
        """
        Args:
            patch_size (int, optional): patch token size. Default: (1, 1, 1)
            in_chans (int, optional): number of input video channels. Default: 16
            embed_dim (int, optional): number of linear projection output channels. Default: 64
            norm_layer (nn.Module, optional): normalization layer. Default: None
        """

        super().__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim, elementwise_affine=False)
        else:
            self.norm = None

    def forward(self, x):
        """
        Forward function
        Args:
            x (torch.Tensor): input tensor of shape [N, C, D, H, W]
        Returns:
            x (torch.Tensor): output tensor of shape [N, embed_dim, D // patch_size, H // patch_size, W // patch_size]
        """

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


class conv_block(nn.Module):

    """ Residual 3D CNN Block """

    def __init__(self, in_channels: int = 96,
                 out_channels: int = 96,
                 kernel_size: tuple = (3, 3, 3),
                 drop_rate: float = 0.,
                 drop_path: float = 0.):
        super(conv_block, self).__init__()

        """
        Args:
            in_channels (int, optional): number of input channels. Default: 96
            out_channels (int, optional): number of output channels. Default: 96
            kernel_size (tuple, optional): convolution kernel size. Default: (3, 3, 3)
            drop_rate (float, optional): dropout rate. Default: 0.
            drop_path (float, optional): dropout path rate. Default: 0.        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_rate = drop_rate
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.conv1 = nn.Conv3d(self.out_channels, self.out_channels,
                               kernel_size=kernel_size, stride=(1, 1, 1),
                               padding=(1, 1, 1), padding_mode='replicate', bias=False)
        self.norm1 = nn.LayerNorm(self.out_channels)

        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels,
                               kernel_size=kernel_size, stride=(1, 1, 1),
                               padding=(1, 1, 1), padding_mode='replicate', bias=False)

        self.norm2 = nn.LayerNorm(self.out_channels)

        self.act = nn.ReLU(inplace=True)

        if self.in_channels != self.out_channels:  #or patch_size != (1, 1, 1):
            self.downsample = PatchEmbed3D(patch_size=(1, 1, 1),
                                           in_chans=self.in_channels,
                                           embed_dim=self.out_channels,
                                           norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

    def forward(self, x):
        """
        Args:
            x (torch.tensor): input tensor [N, C, D, H, W]
        Returns:
            x (torch.tensor): output tensor [N, C, D, H, W]
        """

        if self.downsample is not None:
            x = self.downsample(x)

        B, C, D, H, W = x.shape

        shortcut = x

        x = self.conv1(x)
        x = x.view(B, C, D * H * W).permute(0, 2, 1)
        x = self.norm1(x)
        x = x.permute(0, 2, 1).view(B, C, D, H, W)
        x = self.act(x)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.act(self.norm2(self.conv2(x).view(B, C, D * H * W).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, D, H, W)))

        return x


class CNN_3D(nn.Module):
    """ 2D CNN Encoder """
    def __init__(self, in_vars: int = 6, in_channels: int = 1, out_channels: list = None,
                 drop_path_rate: float = 0., drop_rate: float = 0.,):
        """
        Args:
            in_vars (int, optional): number of input variables. Defaults to 6
            in_channels (int, optional): number of input channels. Defaults to 1
            out_channels (list, optional): number of output channels. Defaults to None
            drop_path_rate (float, optional): dropout rate. Defaults to 0.
            drop_rate (float, optional): dropout rate. Defaults to 0.
        """

        super(CNN_3D, self).__init__()

        self.in_vars = in_vars
        self.out_channels = out_channels if out_channels is not None else [16, 16]
        self.n_layers = len(self.out_channels)

        self.in_channels = [in_channels]

        for i in range(self.n_layers - 1):
            self.in_channels.append(self.out_channels[i])

        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate

        # build layers
        self.layers_var, self.proj_var = nn.ModuleList(), nn.ModuleList()

        for _ in range(self.in_vars):
            layers = nn.ModuleList()
            for layer in range(self.n_layers):
                layers.append(conv_block(self.in_channels[layer], self.out_channels[layer], (3, 3, 3),
                                         self.drop_rate, self.drop_path_rate))

            self.layers_var.append(layers)

            self.proj_var.append(
                nn.Sequential(nn.Conv3d(self.out_channels[-1], self.out_channels[-1], kernel_size=3, stride=1,
                                        padding=1, padding_mode='replicate', bias=True),
                              nn.ReLU(),
                              nn.Conv3d(self.out_channels[-1], self.out_channels[-1], kernel_size=3, stride=1,
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

    def forward(self, x: torch.tensor):
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

    test_dynamic = torch.randn((1, 6, 1, 8, 512//2, 832//2), device=device)

    model = CNN_3D().to(device)
    print(model)

    n_parameters = sum(p.numel() for p in model.proj_var.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    test = model(test_dynamic)
    print(test.shape)

