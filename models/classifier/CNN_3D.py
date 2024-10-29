# ------------------------------------------------------------------
"""
3D CNN Classifier

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

# ------------------------------------------------------------------

class Layer_v(nn.Module):
    """ 3D CNN Classification head used for each variable separately """
    def __init__(self, embed_dim: int = 16, dim: int = 16, n_classes: int = 1, drop_rate: int = 0.1):
        """
        Args:
            embed_dim (int, optional): input channel dimension. Defaults to 16
            dim (int, optional):hidden dimension. Defaults to 16
            n_classes (int, optional): number of classes. Defaults to 1
            drop_rate (float, optional): dropout rate. Defaults to 0.1
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.dim = dim
        self.n_classes = n_classes
        self.drop_rate = drop_rate

        # TODO add generic model for delta T

        self.conv1 = nn.Conv3d(embed_dim, dim, (2, 3, 3), (2, 1, 1), (0, 1, 1), bias=True)
        self.conv2 = nn.Conv3d(dim, dim, (2, 3, 3), (2, 1, 1), (0, 1, 1), bias=True)
        self.conv3 = nn.Conv3d(dim, n_classes, (2, 3, 3), (2, 1, 1), (0, 1, 1), bias=True)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop_rate, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function
        Args:
            x (torch.tensor): input tensor of shape (N, C, T, H, W)
        Returns
            x (torch.tensor): output tensor of shape (N, n_classes, H, W)
        """
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.squeeze(2)
        return x


class CNN_3D(nn.Module):
    """ 3D CNN Classifier """
    def __init__(self, in_var: int = 6, embed_dim: int = 16, dim: int = 16, n_classes: int = 1, drop_rate: int = 0.2):
        """
        Args:
            in_var (int, optional): number of input variables. Defaults to 6
            embed_dim (int, optional): input channel dimension for each variable. Defaults to 16
            dim (int, optional):hidden dimension. Defaults to 16
            n_classes (int, optional): number of classes. Defaults to 1
            drop_rate (float, optional): dropout rate. Defaults to 0.2
        """

        super().__init__()

        self.in_var = in_var
        self.embed_dim = embed_dim * in_var
        self.dim = dim * in_var
        self.n_classes = n_classes
        self.drop_rate = drop_rate

        # TODO add generic model for delta T
        # create a joint-head classifier
        self.conv1 = nn.Conv3d(self.embed_dim, self.dim, (2, 3, 3), (2, 1, 1), (0, 1, 1), bias=True)
        self.conv2 = nn.Conv3d(self.dim, self.dim, (2, 3, 3), (2, 1, 1), (0, 1, 1), bias=True)
        self.conv3 = nn.Conv3d(self.dim, self.n_classes, (2, 3, 3), (2, 1, 1), (0, 1, 1), bias=True)

        # create a multi-head classifier
        self.layers = nn.ModuleList()
        for i in range(in_var):
            self.layers.append(Layer_v(embed_dim=embed_dim, dim=dim, n_classes=1, drop_rate=drop_rate))

        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop_rate, inplace=False)

    def init_weights(self):
        """ Initialize the weights in the model """
        def _init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.04)  # 0.04
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
            x (torch.tensor): input tensor of shape (N, V, C, T, H, W)
        Returns
            x (torch.tensor): output tensor from the joint classification head of shape (N, n_classes, H, W)
            y (list of torch.tensor): output tensors from the multi-head classifier of shape (N, n_classes, H, W)
                                      len(y) = V
        """

        N, V, C, T, H, W = x.shape

        y = []
        for i in range(self.in_var):
            y.append(self.layers[i](x[:, i, ...]))

        x = x.reshape(N, V * C, T, H, W)

        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.squeeze(2)

        return x, y


if __name__ == '__main__':

    test_zq = torch.randn((1, 6, 16, 8, 512 // 4, 832 // 4), device='cuda')

    model = CNN_3D().to('cuda')
    print(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    test, test_var = model(test_zq)
    print(test.shape)
    print(test_var[1].shape)

    test = F.log_softmax(test, dim=1).exp()
    print(torch.sigmoid(test[0, 0, 60, 104]))

