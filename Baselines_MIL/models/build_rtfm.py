# ------------------------------------------------------------
"""
This script includes the main class to import and build the RTFM model
Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning
https://arxiv.org/abs/2101.10030

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------

import torch
import torch.nn as nn
import importlib

# ------------------------------------------------------------

def import_class(en_de, name):
    """ helper function to import a class from a string"""
    module = importlib.import_module("Baselines_MIL.models." + en_de + '.' + name)
    return getattr(module, name)


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class Aggregate(nn.Module):
    def __init__(self, len_feature=16, dim=32):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm2d
        self.len_feature = len_feature
        self.dim = dim
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=len_feature, out_channels=dim, kernel_size=3,
                      stride=1, dilation=1, padding=1),
            nn.ReLU(),
            bn(dim)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=len_feature, out_channels=dim, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(dim)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=len_feature, out_channels=dim, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(dim)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=len_feature, out_channels=dim, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=dim * 4, out_channels=len_feature, kernel_size=3,
                      stride=1, padding=1, bias=False),  # should we keep the bias?
            nn.ReLU(),
            bn(len_feature),
            # nn.dropout(0.7)
        )

        # self.non_local = NONLocalBlock1D(dim, sub_sample=False, bn_layer=False)  # it is not feasable to do attention for HxW

    def forward(self, x):
        # x: (B, T, F)
        # z: B, V, C, H, W

        # B, V, C, H, W = x.shape
        B, V, C, T, H, W = x.shape

        # out = x.view(B*V, C, H, W)
        out = x.permute(0, 1, 3, 2, 4, 5).reshape(B * V * T, C, H, W)
        residual = out

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)

        out3 = self.conv_3(out)
        out_d = torch.cat((out1, out2, out3), dim=1)
        out = self.conv_4(out)

        # out = out.view(B*V, self.dim, H*W)
        # out = self.non_local(out)
        # out = out.view(B*V, self.dim, H, W)

        out = torch.cat((out_d, out), dim=1)

        out = self.conv_5(out)  # fuse all the features together
        out = out + residual

        # out = out.view(B, V, C, H, W)
        out = out.view(B, V, T, C, H, W).permute(0, 1, 3, 2, 4, 5)
        # out: (B, T, 1)

        return out


class MIL_model(nn.Module):
    """ Main MIL model including the encoder and classifier """

    def __init__(self, config):
        super(MIL_model, self).__init__()

        """
        Args:
        -----
        config (argparse): configuration file from config.py
        """

        if config.encoder == "CNN_3D":
            self.encoder = import_class('encoder', config.encoder)(
                in_vars=config.in_channels_dynamic,
                in_channels=config.in_channels,
                out_channels=config.en_embed_dim,
                drop_path_rate=config.en_drop_path_rate,
                drop_rate=config.en_drop_rate
            )
        elif config.encoder == "Swin_3D":
            self.encoder = import_class('encoder', config.encoder)(
                in_vars=config.in_channels_dynamic,
                in_chans=config.in_channels,
                embed_dim=config.en_embed_dim,
                window_size=config.en_window_size,
                depths=config.en_depths,
                num_heads=config.en_n_heads,
                mlp_ratio=config.en_mlp_ratio,
                drop_rate=config.en_drop_rate,
                attn_drop_rate=config.en_attn_drop_rate,
                drop_path_rate=config.en_drop_path_rate,
                qkv_bias=config.en_qkv_bias,
                qk_scale=config.en_qk_scale,
                patch_size=config.en_patch_size,
                patch_norm=config.en_patch_norm,
                use_checkpoint=config.en_use_checkpoint
            )
        elif config.encoder == "Mamba":
            self.encoder = import_class('encoder', config.encoder)(
                in_vars=config.in_channels_dynamic,
                in_chans=config.in_channels,
                embed_dim=config.en_embed_dim,
                window_size=config.en_window_size,
                depths=config.en_depths,
                mlp_ratio=config.en_mlp_ratio,
                drop_rate=config.en_drop_rate,
                drop_path_rate=config.en_drop_path_rate,
                patch_size=config.en_patch_size,
                patch_norm=config.en_patch_norm,
                use_checkpoint=config.en_use_checkpoint,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                dt_min=config.dt_min,
                dt_max=config.dt_max
            )
        else:
            raise NotImplementedError(f"Encoder {config.encoder} not implemented")

        self.agent = import_class('agent', config.agent)(
            in_vars=config.in_channels_dynamic,
            in_chans=config.en_embed_dim[-1],
            embed_dim=config.agent_embed_dim,
            window_size=config.agent_window_size,
            depths=config.agent_depths,
            num_heads=config.agent_n_heads,
            mlp_ratio=config.agent_mlp_ratio,
            drop_rate=config.agent_drop_rate,
            attn_drop_rate=config.agent_attn_drop_rate,
            drop_path_rate=config.agent_drop_path_rate,
            qkv_bias=config.agent_qkv_bias,
            qk_scale=config.agent_qk_scale,
            patch_size=config.agent_patch_size,
            patch_norm=config.agent_patch_norm,
            use_checkpoint=config.agent_use_checkpoint,
        )

        self.Aggregate = Aggregate(len_feature=config.en_embed_dim[-1], dim=config.dim_mtn_rtfm)

        self.classifier = import_class('classifier', config.classifier)(embed_dim=config.en_embed_dim[-1],
                                                                        dim=config.cls_dim,
                                                                        drop_rate=config.cls_drop_rate)
        self.pretrained = config.en_de_pretrained
        self._init_weights()

    def _init_weights(self, init_type='normal', gain=.02):  # gain=nn.init.calculate_gain('leaky_relu', 0.2)

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm3d') != -1 or classname.find(
                    'LayerNorm') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    # torch.nn.init.normal_(m.weight.data, 1.0, gain)
                    torch.nn.init.constant_(m.weight.data, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        if self.pretrained:
            print('initialize weights from pretrained model {} ...'.format(self.pretrained))
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            self.load_state_dict(state_dict, strict=True)
            del state_dict, checkpoint
            torch.cuda.empty_cache()

    def forward(self, x_d, mask_extreme):

        z = self.encoder(x_d)
        z = self.agent(z)  # B, V, C, T, H, W

        # B, V, C, T, H, W = z.shape
        # z = z.permute(1, 2, 3, 0, 4, 5)  # V, C, T, B, H, W
        # z = torch.mean(z, dim=3, keepdim=False)

        z = self.Aggregate(z)

        # z = z.unsqueeze(3)
        z = z.permute(1, 2, 3, 0, 4, 5)  # V, C, T, B, H, W

        z_n, z_p = [], []
        z_n_feature, z_p_feature = [], []

        for i in range(len(mask_extreme)):
            #z_n.append(z[:, :, :, mask_extreme[i] == 0])  # V, C, T, Nn
            #z_p.append(z[:, :, :, mask_extreme[i] != 0])  # V, C, T, Np

            # z.shape   V, C, T, N, H, W

            z_n_i = z[:, :, :, i, mask_extreme[i] == 0]  # V, C, T, Nn
            z_p_i = z[:, :, :, i, mask_extreme[i] != 0]  # V, C, T, Np

            z_n_i = z_n_i.permute(3, 0, 2, 1)  # Nn, V, T, C
            z_p_i = z_p_i.permute(3, 0, 2, 1)  # Np, V, T, C

            z_n_feature_i, z_n_i = self.classifier(z_n_i)
            z_p_feature_i, z_p_i = self.classifier(z_p_i)

            z_n.append(z_n_i)  # Nn, V, T, 1
            z_p.append(z_p_i)  # Np, V, T, 1

            z_n_feature.append(z_n_feature_i)  # Np, V, T, C
            z_p_feature.append(z_p_feature_i)  # Np, V, T, C

        # z_n = z[:, :, :, mask_extreme == 0]  # V, C, T, Nn
        # z_p = z[:, :, :, mask_extreme != 0]  # V, C, T, Np

        # z_n = z_n.permute(0, 3, 1, 2)  # V, Nn, C, T
        # z_p = z_p.permute(0, 3, 1, 2)  # V, Np, C, T

        # z_n = self.drop_instance(z_n)
        # z_p = self.drop_instance(z_p)

        #    z_n = z_n.permute(3, 0, 2, 1)  # Nn, V, T, C
        #    z_p = z_p.permute(3, 0, 2, 1)  # Np, V, T, C

        #    z_n = self.classifier(z_n)
        #    z_p = self.classifier(z_p)

        return z_n, z_p, z_n_feature, z_p_feature


if __name__ == '__main__':

    import Baselines_MIL.config as config
    import os

    config_file = config.read_arguments(train=True, print=False, save=False)

    if config_file.gpu_id != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config_file.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    test_dynamic = torch.randn((1, 6, 1, 8, 100, 100), device=device)
    test_mask_extreme = torch.randint(2, (1, 100, 100)).long().to(device)

    model = MIL_model(config_file).to(device)
    print(model)
    model.eval()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    z_n, z_p, _, _ = model(test_dynamic, test_mask_extreme)

    for i in range(len(z_n)):
        print(z_n[i].shape)
        print(z_p[i].shape)

