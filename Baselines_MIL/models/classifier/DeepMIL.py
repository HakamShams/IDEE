# ------------------------------------------------------------------
"""
Real-world Anomaly Detection in Surveillance Videos
DeepMIL https://arxiv.org/abs/1801.04264

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init as torch_init

# --------------------------------------------------------


class DeepMIL(nn.Module):
    def __init__(self, embed_dim: int = 24, dim: list = None, drop_rate: float = 0.6):
        super(DeepMIL, self).__init__()

        self.embed_dim = embed_dim
        self.dim = dim
        self.drop_rate = drop_rate

        self.mlp_layers = nn.ModuleList()
        for i in range(len(dim)):

            self.mlp_layers.append(nn.Sequential(nn.Linear(embed_dim if i == 0 else dim[i-1], dim[i]),
                                                 nn.ReLU() if i != len(dim) - 1 else nn.Sigmoid(),
                                                 nn.Dropout(drop_rate) if i != len(dim) - 1 else nn.Identity()
                                                 )
                                   )

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

        return x


if __name__ == '__main__':

    device = 'cuda'

    x = torch.randn((2, 6, 8, 512//4, 832//4, 16), device=device)

    model = DeepMIL(embed_dim=16, dim=[512, 32, 1], drop_rate=0.6).to(device)

    print(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    test = model(x)
    print(test.shape)
