# ------------------------------------------------------------
"""
This script includes the main class to import and build the SimpleNet model
https://arxiv.org/abs/2303.15140

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
    module = importlib.import_module("Baselines_OneClass.models." + en_de + '.' + name)
    return getattr(module, name)


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc",
                                   torch.nn.Linear(_in, _out, bias=False))
            if i < n_layers - 1:
                #    if layer_type > 0:
                #       self.layers.add_module(f"{i}bn",
                #              torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):

        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x


class Backbone(nn.Module):
    """ Backbone model including the encoder """

    def __init__(self, config):
        super(Backbone, self).__init__()

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

        self.pretrained = config.en_de_pretrained
        self._init_weights()

    def _init_weights(self):  # gain=nn.init.calculate_gain('leaky_relu', 0.2)

        if self.pretrained:
            print('initialize weights from pretrained model {} ...'.format(self.pretrained))
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            state_dict = {k: v for k, v in state_dict.items() if "cls" not in k}
            state_dict = {k: v for k, v in state_dict.items() if "vq" not in k}
            state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
            state_dict = {k: v for k, v in state_dict.items() if "Aggregate" not in k}

            self.load_state_dict(state_dict, strict=False)
            del state_dict, checkpoint
            torch.cuda.empty_cache()


    def forward(self, x_d):
        """
        Args:
        -----
            x_d (torch.tensor): input dynamic variables [N, V, C, T, H, W]
        Returns:
        --------
            z (torch.tensor): output tensor of shape (N, V, embedding_dim, T, H, W]
        """
        z = self.encoder(x_d)

        return z


class SimpleNet(nn.Module):
    """ Encoder-Decoder U-Net """

    def __init__(self, config):
        super(SimpleNet, self).__init__()

        """
        Args:
        -----
        config (argparse): configuration file from config.py
        """

        self.pre_projection = Projection(in_planes=config.en_embed_dim[-1], out_planes=config.dim,
                                         n_layers=config.pre_proj, layer_type=config.proj_layer_type)

        self.discriminator = Discriminator(in_planes=config.dim, n_layers=config.dsc_layers, hidden=config.dsc_hidden)

        self.mix_noise = config.mix_noise
        self.noise_std = config.noise_std

        self.pretrained = config.model_pretrained
        self._init_weights()

    def _init_weights(self):

        def init_weight(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

        self.apply(init_weight)

        if self.pretrained:
            print('initialize weights from pretrained model {} ...'.format(self.pretrained))
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=True)
            del state_dict, checkpoint
            torch.cuda.empty_cache()

    def forward(self, z):
        """
        Args:
        -----
            z (torch.tensor): output tensor from the joint classification head of shape (N, V, embedding_dim, T, H, W]
        Returns:
        --------
            z_n_scores (torch.tensor): the classification for normal data of shape (N, V, T, H, W, 1)
            if training returns also:
                z_p_scores (torch.tensor): the classification for abnormal data of shape (N, V, T, H, W, 1)
       """

        # z = 0.038 + (z - 0.0644) * 0.0684 / 0.2880
        z = z * 0.01  # this was empirically better than using the row features
        B, V, C, T, H, W = z.shape

        z = z.permute(0, 1, 3, 4, 5, 2)  # B, V, T, H, W, C
        z = z.reshape(B * V * T * H * W, C)  # B*V*T*H*W, C

        z = self.pre_projection(z)
        # add noise
        if self.training:
            # slow as original implementation
            # noise_idxs = torch.randint(0, self.mix_noise, torch.Size([z.shape[0]]))
            #    noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(z.device)  # (N, K)
            #    noise = torch.stack([
            #        torch.normal(0, self.noise_std * 1.1 ** (k), z.shape)
            #        for k in range(self.mix_noise)], dim=1).to(z.device)  # (N, K, C)
            #    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
            noise = torch.normal(0, self.noise_std, z.shape, device=z.device)
            z_p = z + noise
            z_p_scores = self.discriminator(z_p)
            z_p_scores = z_p_scores.reshape(B, V, T, H, W, 1)  # B, V, T, H, W, C

        z_n_scores = self.discriminator(z)
        z_n_scores = z_n_scores.reshape(B, V, T, H, W, 1)  # B, V, T, H, W, C

        if self.training:
            return z_n_scores, z_p_scores
        else:
            return z_n_scores


if __name__ == '__main__':

    import Baselines_OneClass.config as config
    import os

    config_file = config.read_arguments(train=True, print=False, save=False)

    if config_file.gpu_id != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config_file.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    test_dynamic = torch.randn((1, 6, 1, 8, 200, 200), device=device)

    backbone = Backbone(config_file).to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    model = SimpleNet(config_file).to(device)
    print(model)
    model.eval()

    n_parameters = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in backbone.parameters())

    print(f"number of parameters: {n_parameters}")

    z_n_scores = model(backbone(test_dynamic.to(device)))

    print(z_n_scores.shape)

