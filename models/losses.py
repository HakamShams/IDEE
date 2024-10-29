# ------------------------------------------------------------------
""""
Loss functions

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn

# ------------------------------------------------------------------

class Anomaly_L1_loss(nn.Module):
    """
       Anomaly loss
       used to penalize the model if it predicts anomalies outside of regions where there were no extreme events reported
       also used to control the quantization assignment so that all variables are assigned to the same class in the Quantization layer
    """
    def __init__(self, n_dynamic: int = 6, delta_t: int = 8, dim: int = 24):
        """
        Args:
            n_dynamic (int, optional): number of input dynamic variables. Defaults to 3
            delta_t (int, optional): delta time or temporal resolution. Defaults to 8
            dim (int, optional): dimension of input tensors. Defaults to 24
        """
        super().__init__()

        self.loss = nn.L1Loss(reduction='none')
        self.n_dynamic = n_dynamic
        self.delta_t = delta_t
        self.dim = dim

    def forward(self, pred, mask_extreme, mask_valid, vq_0):
        """
        Forward function
            pred (torch.tensor): prediction tensor of shape (N, V, C, T, H, W)
            mask_extreme (torch.tensor): mask tensor of extreme events of shape (N, H, W)
            mask_valid (torch.tensor): mask tensor of valid pixels of shape (N, H, W)
            vq_0 (torch.tensor): tensor of shape (N, C) representing the quantization code for normal data
        Returns:
            loss (torch.tensor): loss tensor (scaler)
        """

        N, H, W = mask_extreme.shape

        mask_extreme = mask_extreme.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, self.n_dynamic, self.dim, self.delta_t, 1, 1)
        mask_valid = mask_valid.unsqueeze(1).unsqueeze(1).repeat(1, self.n_dynamic, self.dim, 1, 1, 1)
        mask_extreme = mask_extreme + mask_valid
        mask_extreme[mask_extreme > 1] = 1
        weights = 1 - mask_extreme.clone()

        vq_0 = vq_0.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5).repeat(N, self.n_dynamic, 1, self.delta_t, H, W)
        vq_0[mask_extreme == 1] = pred[mask_extreme == 1].clone().detach()
        vq_0 = vq_0.requires_grad_(False).float()

        loss = self.loss(pred, vq_0) * weights
        loss = (torch.sum(loss) / torch.sum(weights))

        return loss


class BCE_loss(nn.Module):
    """ Binary Cross Entropy Loss """
    def __init__(self):
        super(BCE_loss, self).__init__()

        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target, mask_valid):
        """
        Forward function
        Args:
            pred (torch.tensor): prediction tensor of shape (N, H, W)
            target (torch.tensor): target tensor of shape (N, H, W)
            mask_valid (torch.tensor): mask tensor of valid pixels of shape (N, H, W)
        Returns
            loss (torch.tensor): loss tensor (scaler)
        """
        # generate weights based on the inverse of frequency
        weights = torch.histc(target[mask_valid.bool()].float(), bins=2)
     #   weights = 1 - weights / torch.sum(weights)
        weights[torch.isinf(weights)] = 1
        weights = (weights / torch.sum(weights)) ** -0.5
        weights = torch.log(weights + 1.1)
        weights = weights[target.long()].requires_grad_(False)
        # weight will be zero for invalid pixels
        weights[mask_valid == 0] = 0

        # compute loss
        loss = self.loss(pred, target.float()) * weights
        loss = (torch.sum(loss) / torch.sum(mask_valid))

        return loss


class BCE_loss_synthetic(nn.Module):
    """ Binary Cross Entropy Loss for the synthetic data"""
    def __init__(self):
        super(BCE_loss_synthetic, self).__init__()

        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, target):
        """
        Forward function
        Args:
            pred (torch.tensor): prediction tensor of shape (N, C, H, W)
            target (torch.tensor): target tensor of shape (N, C, H, W)
        Returns
            loss (torch.tensor): loss tensor (scaler)
        """
        # generate weights based on the inverse of frequency
        weights = torch.histc(target, bins=2)
     #   weights = 1 - weights / torch.sum(weights)
        weights[torch.isinf(weights)] = 1
        weights = (weights / torch.sum(weights)) ** -0.5
        weights = torch.log(weights + 1.1)
        weights = weights[target.long()].requires_grad_(False)

        # compute loss
        loss = torch.mean(self.loss(pred, target) * weights)
        return loss


class Anomaly_L1_loss_synthetic(nn.Module):
    """
        Anomaly loss
        used to penalize the model if it predicts anomalies outside of regions where there were no extreme events reported
        also used to control the quantization assignment so that all variables are assigned to the same class in the Quantization layer
    """
    def __init__(self, n_dynamic: int = 3, delta_t: int = 8, dim: int = 24):
        """
        Args:
            n_dynamic (int, optional): number of input dynamic variables. Defaults to 3
            delta_t (int, optional): delta time or temporal resolution. Defaults to 8
            dim (int, optional): dimension of input tensors. Defaults to 24
        """
        super().__init__()

        self.loss = nn.L1Loss(reduction='none')
        self.n_dynamic = n_dynamic
        self.delta_t = delta_t
        self.dim = dim

    def forward(self, pred, mask_extreme, vq_0):
        """
        Forward function
            pred (torch.tensor): prediction tensor of shape (N, V, C, T, H, W)
            mask_extreme (torch.tensor): mask tensor of extreme events of shape (N, H, W)
            vq_0 (torch.tensor): tensor of shape (N, C) representing the quantization code for normal data
        Returns:
            loss (torch.tensor): loss tensor (scaler)
        """
        N, H, W = mask_extreme.shape

        mask_extreme = mask_extreme.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, self.n_dynamic, self.dim, self.delta_t, 1, 1)
        weights = 1 - mask_extreme.clone()

        vq_0 = vq_0.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5).repeat(N, self.n_dynamic, 1, self.delta_t, H, W)
        vq_0[mask_extreme == 1] = pred[mask_extreme == 1].clone().detach()
        vq_0 = vq_0.requires_grad_(False).float()

        loss = self.loss(pred, vq_0) * weights
        loss = (torch.sum(loss) / torch.sum(weights))

        return loss
