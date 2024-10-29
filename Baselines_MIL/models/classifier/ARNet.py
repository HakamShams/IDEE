# ------------------------------------------------------------------
"""
Weakly Supervised Video Anomaly Detection via Center-guided Discriminative Learning
ARNet https://arxiv.org/abs/2104.07268

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init as torch_init

# --------------------------------------------------------


class InstanceDropout(nn.Module):
    """  Dropout without rescaling """

    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate == 0:
            return x
        else:
            mask = torch.empty(x.shape, device=x.device, requires_grad=False).bernoulli_(1 - self.drop_rate)
            return x * mask


class ARNet(nn.Module):
    def __init__(self, embed_dim: int = 16, dim: list = None, drop_rate: float = 0.6):
        super(ARNet, self).__init__()

        self.embed_dim = embed_dim
        self.dim = dim
        self.drop_rate = drop_rate

        self.mlp_layers = nn.ModuleList()

        for i in range(len(dim)):

            self.mlp_layers.append(nn.Sequential(nn.Linear(embed_dim if i == 0 else dim[i-1], dim[i]),
                                                 nn.ReLU() if i != len(dim) - 1 else nn.Sigmoid(),
                                                 )
                                   )
        self.dropout = nn.Dropout(drop_rate)
        self.instance_dropout = InstanceDropout(drop_rate)

        self._init_weights()

    def _init_weights(self):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                torch_init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.apply(init_func)

    def forward(self, x):

        for i in range(len(self.mlp_layers)):
            x = self.mlp_layers[i](x)
            if i == 0:
                x_features = x
            if i != len(self.mlp_layers) - 1:
                x = self.dropout(x)
            #else:
            #    x = self.instance_dropout(x)

        return x_features, x


if __name__ == '__main__':

    device = 'cuda'

    x = torch.randn((2, 6, 8, 512//4, 832//4, 16), device=device)

    model = ARNet(embed_dim=16, dim=[512, 32, 1], drop_rate=0.6).to(device)

    print(model)

    test_feature, test = model(x)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    print(test_feature.shape)
    print(test.shape)

