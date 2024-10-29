# ------------------------------------------------------------
"""
This script includes the main class to import and build the models

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
    module = importlib.import_module("models." + en_de + '.' + name)
    return getattr(module, name)


class VQ_model(nn.Module):
    """ Main model including the encoder, quantizer and classifier """
    def __init__(self, config):
        super(VQ_model, self).__init__()
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

        self.cls = import_class('classifier', 'CNN_3D')(in_var=config.in_channels_dynamic,
                                                        embed_dim=config.codebook_dim,
                                                        dim=config.cls_dim,
                                                        drop_rate=config.cls_drop_rate)

        # TODO add different codebook options
        self.vq = import_class('codebook', 'LFQ')(dim=config.codebook_dim,
                                                  codebook_size=config.codebook_size,
                                                  entropy_loss_weight=config.lambda_entropy,
                                                  diversity_gamma=config.diversity_gamma,
                                                  commitment_loss_weight=config.lambda_commitment)

        self.pretrained = config.en_de_pretrained
        self._init_weights()

    def _init_weights(self, init_type='normal', gain=.02):  # gain=nn.init.calculate_gain('leaky_relu', 0.2)

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm3d') != -1 or classname.find(
                    'LayerNorm') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    # torch.nn.init.normal_(m.weight.data, 1.0, gain)
                    torch.nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.02, gain)
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


    def forward(self, x_d):
        """
        Args:
        -----
            x_d (torch.tensor): input dynamic variables [N, V, C, T, H, W]
        Returns:
        --------
            z (torch.tensor): output tensor from the joint classification head of shape (N, n_classes, H, W)
            y (list of torch.tensor): output tensors from the multi-head classifier of shape (N, n_classes, H, W)
                                      len(y) = V
            anomaly (torch.tensor): anomalous events from the quantization of shape (N, V, T, H, W)
            z_q (torch.tensor): output tensor from the quantization of shape (N, V, C, T, H, W)
            loss_z_q (torch.tensor): quantization loss
        """

        # extract feature with the encoder/backbone
        z = self.encoder(x_d)

        # quantize
        N, V, C, T, H, W = z.shape
        z = z.permute(0, 2, 1, 3, 4, 5).reshape(N, C, V * T * H * W).permute(0, 2, 1)
        z_q, anomaly, loss_z_q = self.vq(z)

        z_q = z_q.permute(0, 2, 1).view(N, C, V, T, H, W).permute(0, 2, 1, 3, 4, 5)
        anomaly = anomaly.view(N, V, T, H, W)

        # classify
        z, y = self.cls(z_q)

        return z, y, anomaly, z_q, loss_z_q.unsqueeze(0)


if __name__ == '__main__':

    import config
    import os

    config_file = config.read_arguments(train=True, print=False, save=False)

    if config_file.gpu_id != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config_file.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    test_dynamic = torch.randn((1, 6, 1, 8, 512 // 4, 832 // 4), device=device)

    model = VQ_model(config_file).to(device)
    print(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    z, y, anomaly, z_q, loss_z_q = model(test_dynamic)

    print(z.shape)
    print(y[0].shape)
    print(anomaly.shape)
    print(z_q.shape)
    print(loss_z_q.shape)

