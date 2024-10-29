# ------------------------------------------------------------
"""
This script includes the main class to import and build the DeepMIL model

Real-world Anomaly Detection in Surveillance Videos https://arxiv.org/abs/1801.04264

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

        self.classifier = import_class('classifier',  config.classifier)(embed_dim=config.en_embed_dim[-1],
                                                                         dim=config.cls_dim,
                                                                         drop_rate=config.cls_drop_rate)

        self.pretrained = config.en_de_pretrained
        self._init_weights()

    def _init_weights(self, init_type='normal', gain=.02):  # gain=nn.init.calculate_gain('leaky_relu', 0.2)

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm3d') != -1 or classname.find('LayerNorm') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    #torch.nn.init.normal_(m.weight.data, 1.0, gain)
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
        #z = self.agent(z)  # B, V, C, T, H, W

        #B, V, C, T, H, W = z.shape

        z = z.permute(1, 2, 3, 0, 4, 5)  # V, C, T, B, H, W

        #z = torch.mean(z, dim=2, keepdim=True)

        z_n, z_p = [], []

        for i in range(len(mask_extreme)):

            #z_n.append(z[:, :, :, mask_extreme[i] == 0])  # V, C, T, Nn
            #z_p.append(z[:, :, :, mask_extreme[i] != 0])  # V, C, T, Np
            z_n_i = z[:, :, :, i, mask_extreme[i] == 0]  # V, C, T, Nn
            z_p_i = z[:, :, :, i, mask_extreme[i] != 0]  # V, C, T, Np

            z_n_i = z_n_i.permute(3, 0, 2, 1)  # Nn, V, T, C
            z_p_i = z_p_i.permute(3, 0, 2, 1)  # Np, V, T, C

            z_n_i = self.classifier(z_n_i)  # Nn, V, T, 1
            z_p_i = self.classifier(z_p_i)  # Np, V, T, 1

            z_n.append(z_n_i)  # Nn, V, T, 1
            z_p.append(z_p_i)  # Np, V, T, 1

        return z_n, z_p


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

    test_dynamic = torch.randn((1, 6, 1, 8, 200, 200), device=device)
    test_mask_extreme = torch.randint(2, (1, 200, 200)).long().to(device)

    model = MIL_model(config_file).to(device)
    print(model)
    model.eval()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    z_n, z_p = model(test_dynamic, test_mask_extreme)

    for i in range(len(z_n)):
        print(z_n[i].shape)
        print(z_p[i].shape)

