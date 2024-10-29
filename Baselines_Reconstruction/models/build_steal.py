# ------------------------------------------------------------------
"""
This script includes the main class to import and build the STEALNET model
see: https://arxiv.org/abs/2110.09768
     https://github.com/aseuteurideu/STEAL

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn

# ------------------------------------------------------------


class Reconstruction3DEncoder(nn.Module):
    def __init__(self, chnum_in: int = 6, embed_dim=None):
        super(Reconstruction3DEncoder, self).__init__()

        # Dong Gong's paper code

        if embed_dim is None:
            embed_dim = [96, 128, 256]

        self.chnum_in = chnum_in
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Conv3d(chnum_in, embed_dim[0], (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(embed_dim[0], embed_dim[1], (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(embed_dim[1], embed_dim[2], (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(embed_dim[2], embed_dim[2], (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim[2]),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Reconstruction3DDecoder(nn.Module):
    def __init__(self, chnum_in: int = 6, embed_dim=None):
        super(Reconstruction3DDecoder, self).__init__()

        if embed_dim is None:
            embed_dim = [256, 128, 96]

        # Dong Gong's paper code + Tanh
        self.chnum_in = chnum_in
        self.embed_dim = embed_dim

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim[0], embed_dim[0], (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1),
                               output_padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(embed_dim[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(embed_dim[0], embed_dim[1], (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(embed_dim[1], embed_dim[2], (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(embed_dim[2], self.chnum_in, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                               output_padding=(0, 1, 1)),
            nn.Tanh()
        )

    def forward(self, x):

        #x = self.conv1(x)
        #x = F.interpolate(x, size=(2, 25, 25))
        x = self.decoder(x) * 10  # scale x, so it is bounded between -10, 10
        return x

class Rec_model(nn.Module):
    """ Autoencoder Encoder-Decoder """
    def __init__(self, config):
        super(Rec_model, self).__init__()

        """
        Args:
        -----
        config (argparse): configuration file from config.py
        """

        self.encoder = Reconstruction3DEncoder(chnum_in=config.in_channels_dynamic, embed_dim=config.en_embed_dim_steal)
        self.decoder = Reconstruction3DDecoder(chnum_in=config.in_channels_dynamic, embed_dim=config.de_embed_dim_steal)

        self.pretrained = config.en_de_pretrained
        self._init_weights()

    def _init_weights(self, init_type='normal', gain=.02):  # gain=nn.init.calculate_gain('leaky_relu', 0.2)

        """
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

       # self.apply(init_func)
        # initialization code is based on https://github.com/nvlabs/spade/
        #for m in self.children():
        #    if hasattr(m, '_init_weights'):
        #        m._init_weights()
        """

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
            x_d (torch.tensor): input dynamic variables [N, V, T, H, W]
        Returns:
        --------
            z (torch.tensor): output tensor of shape (N, V, T, H, W]
        """

        z = self.encoder(x_d)
        z = self.decoder(z)

        return z


if __name__ == '__main__':

    import Baselines_Reconstruction.config as config
    import os

    config_file = config.read_arguments(train=True, print=False, save=False)

    if config_file.gpu_id != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config_file.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    test_dynamic = torch.randn((1, 6, 8, 200, 200), device=device)

    model = Rec_model(config_file).to(device)
    print(model)

    n_parameters = sum(p.numel() for p in model.parameters() )#if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    model.eval()
    out = model(test_dynamic)

    print(out.shape)

