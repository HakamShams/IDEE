# ------------------------------------------------------------------
"""
MIL loss functions

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

# ------------------------------------------------------------------


class NoScaleDropout(nn.Module):
    """ Dropout without rescaling """

    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0:
            return x
        else:
            mask = torch.empty(x.shape[0], device=x.device, requires_grad=False).bernoulli_(1 - self.rate)
            return x * mask[:, None, None, None]


class RankingLoss_alt(_Loss):

    def __init__(self):
        super(RankingLoss_alt, self).__init__()
        self.loss = nn.MarginRankingLoss(margin=1., reduction='mean')

    def forward(self, z_p, z_n, y):
        loss = self.loss(z_p, z_n, y)
        return loss


class RankingLoss(_Loss):

    def __init__(self, drop_rate=0.5, k=100):
        super(RankingLoss, self).__init__()

        self.drop_rate = 1 - drop_rate
        self.loss = nn.MarginRankingLoss(margin=1., reduction='mean')
        self.k = k

        #  self.y = torch.tensor(1., requires_grad=False, device=device)

    def forward(self, z_p, z_n, is_training=False):

        if is_training:
            z_p_drop = z_p * torch.empty(z_p.shape, device=z_p.device, requires_grad=False).bernoulli_(self.drop_rate)
            z_n_drop = z_n * torch.empty(z_n.shape, device=z_n.device, requires_grad=False).bernoulli_(self.drop_rate)

            z_p_drop_topk, _ = torch.topk(z_p_drop, k=self.k, dim=0)
            z_n_drop_topk, _ = torch.topk(z_n_drop, k=self.k, dim=0)
            loss = F.relu(1 - z_p_drop_topk + z_n_drop_topk).mean()
            # loss = self.loss(torch.max(z_p_drop), torch.max(z_n_drop), self.y)
        # loss = self.loss(z_p_drop_topk,
        #                  z_n_drop_topk,
        #                  torch.ones(z_n_drop_topk.shape, device=z_p_drop_topk.device, requires_grad=False))
        else:
            # loss = F.relu(1 - torch.max(z_p) + torch.max(z_n))
            # loss = self.loss(torch.max(z_p), torch.max(z_n), self.y)

            z_p_topk, _ = torch.topk(z_p, k=self.k, dim=0)
            z_n_topk, _ = torch.topk(z_n, k=self.k, dim=0)
            #   loss = self.loss(z_p_topk, z_n_topk, torch.ones(z_p_topk.shape, device=z_p_topk.device, requires_grad=False))
            loss = F.relu(1 - z_p_topk + z_n_topk).mean()
        return loss


class SmoothL2Loss(_Loss):
    def __init__(self, lambda1=8e-5):
        super(SmoothL2Loss, self).__init__()
        self.lambda1 = lambda1

    def forward(self, z_p):
        # z_p  Np, V, T, C

        z_p_t = torch.zeros_like(z_p)
        z_p_t[:, :, :-1, :] = z_p[:, :, 1:, :]
        z_p_t[:, :, -1, :] = z_p[:, :, -1, :]

        return torch.sum((z_p - z_p_t) ** 2) * self.lambda1


class SparsityLoss(_Loss):
    def __init__(self, lambda2=8e-5):
        super(SparsityLoss, self).__init__()

        self.lambda2 = lambda2

    def forward(self, z_p):
        # z_p  Np, V, T, C
        return torch.sum(z_p) * self.lambda2


class DMIL_RankingLoss(_Loss):
    def __init__(self, drop_rate=0.5, alpha=400, t=40000):
        super().__init__()
        self.alpha = alpha
        self.k = t // alpha
        self.drop_rate = 1 - drop_rate

        self.loss = nn.BCELoss()

    def forward(self, z_p, z_n, is_training=False):

        if is_training:
            z_p_drop = z_p * torch.empty(z_p.shape, device=z_p.device, requires_grad=False).bernoulli_(self.drop_rate)
            z_n_drop = z_n * torch.empty(z_n.shape, device=z_n.device, requires_grad=False).bernoulli_(self.drop_rate)
        else:
            z_p_drop = z_p
            z_n_drop = z_n

        z_p_topk, _ = torch.topk(z_p_drop, k=self.k, dim=0)
        z_n_topk, _ = torch.topk(z_n_drop, k=self.k, dim=0)

        loss_z_p = self.loss(z_p_topk, torch.ones(z_p_topk.shape, requires_grad=False, device=z_p_topk.device))
        loss_z_n = self.loss(z_n_topk, torch.zeros(z_n_topk.shape, requires_grad=False, device=z_n_topk.device))

        return loss_z_p + loss_z_n


class CenterLoss(_Loss):
    def __init__(self, lambda_c=20):
        super().__init__()

        self.lambda_c = lambda_c
        self.loss = nn.MSELoss()

    def forward(self, z_n):
        return self.loss(z_n,
                         torch.empty(z_n.shape, requires_grad=False, device=z_n.device).fill_(
                             torch.mean(z_n))) * self.lambda_c


class RTFMLoss(_Loss):
    def __init__(self, drop_rate: float = 0.5, alpha: float = 0.0001, margin: float = 100., k: int = 100):
        super(RTFMLoss, self).__init__()

        self.alpha = alpha
        self.margin = margin
        self.k = k
        self.drop_rate = 1 - drop_rate

        self.loss = nn.BCELoss()

    #  self.p_gt = torch.tensor(1., requires_grad=False, device='cuda')
    #  self.n_gt = torch.tensor(0., requires_grad=False, device='cuda')

    def forward(self, z_p, z_n, z_p_features, z_n_features, is_training=False):

        # Nn, _, _, _ = z_n.shape
        # N, V, T, C = z_p_features.shape

        # z_p_drop = z_p.reshape(N*V*T, 1)
        # z_n_drop = z_n.reshape(Nn*V*T, 1)

        # if is_training:
        #    z_p_features_drop = z_p_features * torch.empty(z_p_features.shape[1], device=z_p_features.device, requires_grad=False).bernoulli_(0.7)[None, :, None, None]
        #    z_n_features_drop = z_n_features * torch.empty(z_n_features.shape[1], device=z_n_features.device, requires_grad=False).bernoulli_(0.7)[None, :, None, None]
        # else:
        #    z_p_features_drop = z_p_features
        #    z_n_features_drop = z_n_features

        # z_p_features_drop = z_p_features_drop.reshape(N*V*T, C)
        # z_n_features_drop = z_n_features_drop.reshape(Nn*V*T, C)

        # assert self.k <= len(z_n), "normal pixels are less than k"
        # assert self.k <= len(z_p), "abnormal pixels are less than k"

        if is_training:
            z_p_features_drop = z_p_features * torch.empty(z_p_features.shape[0], device=z_p_features.device,
                                                           requires_grad=False).bernoulli_(self.drop_rate)[:, None,
                                               None]
            z_n_features_drop = z_n_features * torch.empty(z_n_features.shape[0], device=z_n_features.device,
                                                           requires_grad=False).bernoulli_(self.drop_rate)[:, None,
                                               None]
        else:
            z_p_features_drop = z_p_features
            z_n_features_drop = z_n_features

        z_p_features_magnitude = torch.norm(z_p_features_drop, p=2, dim=-1)
        z_n_features_magnitude = torch.norm(z_n_features_drop, p=2, dim=-1)

        z_p_features_topk, z_p_indices = torch.topk(z_p_features_magnitude, k=self.k, dim=0)
        z_n_features_topk, z_n_indices = torch.topk(z_n_features_magnitude, k=self.k, dim=0)

        # loss_z_p = self.loss(z_p[z_p_indices].mean(), self.p_gt)
        # loss_z_n = self.loss(z_n[z_n_indices].mean(), self.n_gt)

        loss_z_p = self.loss(z_p[z_p_indices], torch.ones(z_p[z_p_indices].shape, requires_grad=False, device='cuda'))
        loss_z_n = self.loss(z_n[z_n_indices], torch.zeros(z_n[z_n_indices].shape, requires_grad=False, device='cuda'))

        z_p_features_mean = torch.norm(torch.mean(z_p_features_drop[z_p_indices], dim=0), p=2,
                                       dim=-1)  # follow the official code
        z_n_features_mean = torch.norm(torch.mean(z_n_features_drop[z_n_indices], dim=0), p=2,
                                       dim=-1)  # follow the official code

        loss_z_p_features = torch.abs(self.margin - z_p_features_mean)
        # loss_z_p_features = F.relu(self.margin - z_p_features_mean)
        loss_rtfm = torch.mean((loss_z_p_features + z_n_features_mean) ** 2)

        loss = loss_z_n + loss_z_p + self.alpha * loss_rtfm

        return loss


class RTFMLoss1(_Loss):
    def __init__(self, alpha: float = 0.0001, margin: float = 100., k: int = 100):
        super(RTFMLoss1, self).__init__()

        self.alpha = alpha
        self.margin = margin
        self.k = k

        self.loss = nn.BCELoss()

    # self.p_gt = torch.tensor(1., requires_grad=False, device='cuda')
    # self.n_gt = torch.tensor(0., requires_grad=False, device='cuda')

    def forward(self, z_p, z_n, z_p_features, z_n_features):
        assert self.k <= len(z_n), "normal pixels are less than k"
        assert self.k <= len(z_p), "abnormal pixels are less than k"

        z_p_features_magnitude = torch.norm(z_p_features, p=2, dim=-1)
        z_n_features_magnitude = torch.norm(z_n_features, p=2, dim=-1)

        z_p_features_topk, z_p_indices = torch.topk(z_p_features_magnitude, k=self.k, dim=0)
        z_n_features_topk, z_n_indices = torch.topk(z_n_features_magnitude, k=self.k, dim=0)

        # loss_z_p = self.loss(z_p[z_p_indices].mean(), self.p_gt)
        # loss_z_n = self.loss(z_n[z_n_indices].mean(), self.n_gt)

        loss_z_p = self.loss(z_p[z_p_indices], torch.ones(z_p[z_p_indices].shape, requires_grad=False, device='cuda'))
        loss_z_n = self.loss(z_n[z_n_indices], torch.zeros(z_n[z_n_indices].shape, requires_grad=False, device='cuda'))

        z_p_features_mean = torch.norm(torch.mean(z_p_features[z_p_indices], dim=0), p=2,
                                       dim=-1)  # follow the official code
        z_n_features_mean = torch.norm(torch.mean(z_n_features[z_n_indices], dim=0), p=2,
                                       dim=-1)  # follow the official code

        loss_z_p_features = torch.abs(self.margin - z_p_features_mean)
        loss_rtfm = torch.mean((loss_z_p_features + z_n_features_mean) ** 2)

        loss = loss_z_n + loss_z_p + self.alpha * loss_rtfm

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=100.):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.abs(self.margin - euclidean_distance),
                                                          2))  # torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class MGFNLoss_alt(_Loss):
    def __init__(self, k: int = 3, lambda_mgfn: float = 0.001, margin: float = 100.):
        super(MGFNLoss_alt, self).__init__()

        self.k = k
        self.lambda_mgfn = lambda_mgfn
        self.margin = margin
        self.loss = nn.BCELoss()
        self.contrastive = ContrastiveLoss(margin=margin)

        self.p_gt = torch.tensor(1., requires_grad=False, device='cuda')
        self.n_gt = torch.tensor(0., requires_grad=False, device='cuda')

    def forward(self, z_p, z_n, z_p_features, z_n_features):
        # z_p  [Np, 1, 1]     z_p_features  [Np, 1, 64]

        z_p_features_magnitude = torch.norm(z_p_features, p=2, dim=-1)
        z_n_features_magnitude = torch.norm(z_n_features, p=2, dim=-1)

        z_p_features_topk, z_p_indices = torch.topk(z_p_features_magnitude, k=self.k, dim=0)
        z_n_features_topk, z_n_indices = torch.topk(z_n_features_magnitude, k=self.k, dim=0)

        loss_z_p = self.loss(z_p[z_p_indices].mean(), self.p_gt)
        loss_z_n = self.loss(z_n[z_n_indices].mean(), self.n_gt)

        loss_cls = loss_z_n + loss_z_p

        # feature magnitude should be Np*crop, 3, 64
        # z_p_features_topk = z_p_features_topk.unsqueeze(0)
        # z_n_features_topk = z_n_features_topk.unsqueeze(0)
        # loss_con = self.contrastive(torch.norm(z_p_features_topk, p=1, dim=2), torch.norm(z_n_features_topk, p=1, dim=2),
        #                            1)  # try to separate normal and abnormal
        # loss_con_n = self.contrastive(torch.norm(z_n_features_topk, p=1, dim=2),
        #                              torch.norm(z_n_features_topk, p=1, dim=2),
        #                              0)  # try to cluster the same class
        # loss_con_a = self.contrastive(torch.norm(z_p_features_topk, p=1, dim=2),
        #                              torch.norm(z_p_features_topk, p=1, dim=2), 0)

        loss_con = self.contrastive(z_p_features_topk, z_n_features_topk, 1)  # try to separate normal and abnormal
        loss_con_n = self.contrastive(z_n_features_topk, z_n_features_topk, 0)  # try to cluster the same class
        loss_con_a = self.contrastive(z_p_features_topk, z_p_features_topk, 0)

        loss_total = loss_cls + self.lambda_mgfn * (self.lambda_mgfn * loss_con + loss_con_a + loss_con_n)

        return loss_total


class MGFNLoss(_Loss):
    def __init__(self, drop_rate: float = 0.5, k: int = 3, lambda_mgfn: float = 0.001, margin: float = 10.,
                 n_var: int = 6):
        super(MGFNLoss, self).__init__()

        self.k = k
        self.lambda_mgfn = lambda_mgfn
        self.margin = margin
        self.n_var = n_var
        self.loss = nn.BCELoss()

        self.drop_rate = 1 - drop_rate

        self.contrastive = ContrastiveLoss(margin=margin)

    #        self.p_gt = torch.tensor(1., requires_grad=False, device='cuda')
    #       self.n_gt = torch.tensor(0., requires_grad=False, device='cuda')

    def forward(self, z_p, z_n, z_p_features, z_n_features, is_training=False):

        # z_p  [Np, 1, 1]     z_p_features  [Np, 1, 64]

        loss_cls = torch.tensor(0., device='cuda')
        loss_con = torch.tensor(0., device='cuda')
        loss_con_n = torch.tensor(0., device='cuda')
        loss_con_a = torch.tensor(0., device='cuda')

        for v in range(self.n_var):

            z_p_features_topk_all, z_n_features_topk_all = None, None

            for n in range(len(z_n)):

                if is_training:
                    z_p_features_drop = z_p_features[n][:, v, ...] * torch.empty(z_p_features[n][:, v, ...].shape[0],
                                                                                 device=z_p_features[n][:, v,
                                                                                        ...].device,
                                                                                 requires_grad=False).bernoulli_(
                        self.drop_rate)[:, None, None]
                    z_n_features_drop = z_n_features[n][:, v, ...] * torch.empty(z_n_features[n][:, v, ...].shape[0],
                                                                                 device=z_n_features[n][:, v,
                                                                                        ...].device,
                                                                                 requires_grad=False).bernoulli_(
                        self.drop_rate)[:, None, None]
                else:
                    z_p_features_drop = z_p_features[n][:, v, ...]
                    z_n_features_drop = z_n_features[n][:, v, ...]

                z_p_features_magnitude = torch.norm(z_p_features_drop, p=2, dim=-1)
                z_n_features_magnitude = torch.norm(z_n_features_drop, p=2, dim=-1)

                _, z_p_indices = torch.topk(z_p_features_magnitude, k=self.k, dim=0)
                _, z_n_indices = torch.topk(z_n_features_magnitude, k=self.k, dim=0)

                # loss_z_p = self.loss(z_p[n][:, v, ...][z_p_indices].mean(), self.p_gt)
                # loss_z_n = self.loss(z_n[n][:, v, ...][z_n_indices].mean(), self.n_gt)

                loss_z_p = self.loss(z_p[n][:, v, ...][z_p_indices],
                                     torch.ones(z_p[n][:, v, ...][z_p_indices].shape, requires_grad=False,
                                                device='cuda'))
                loss_z_n = self.loss(z_n[n][:, v, ...][z_n_indices],
                                     torch.zeros(z_n[n][:, v, ...][z_n_indices].shape, requires_grad=False,
                                                 device='cuda'))

                loss_cls += loss_z_n + loss_z_p

                # z_p_features_topk = z_p_features_topk.unsqueeze(0)
                # z_n_features_topk = z_n_features_topk.unsqueeze(0)

                if z_p_features_topk_all is None:
                    z_p_features_topk_all = z_p_features[n][:, v, 0, :][z_p_indices[:, 0]].unsqueeze(0)
                    z_n_features_topk_all = z_n_features[n][:, v, 0, :][z_n_indices[:, 0]].unsqueeze(0)
                else:
                    z_p_features_topk_all = torch.cat((z_p_features_topk_all,
                                                       z_p_features[n][:, v, 0, :][z_p_indices[:, 0]].unsqueeze(0)),
                                                      dim=0)
                    z_n_features_topk_all = torch.cat((z_n_features_topk_all,
                                                       z_n_features[n][:, v, 0, :][z_n_indices[:, 0]].unsqueeze(0)),
                                                      dim=0)

            # feature magnitude should be Np*crop, 3, 64
            seperate = int(len(z_n_features_topk_all) / 2)

            loss_con += self.contrastive(torch.norm(z_p_features_topk_all, p=1, dim=2),
                                         torch.norm(z_n_features_topk_all, p=1, dim=2),
                                         1)  # try tp separate normal and abnormal
            if len(z_n_features_topk_all) % 2 == 0:
                seperate = int(len(z_n_features_topk_all) / 2)
                loss_con_n += self.contrastive(torch.norm(z_n_features_topk_all[:seperate], p=1, dim=2),
                                               torch.norm(z_n_features_topk_all[seperate:], p=1, dim=2),
                                               0)  # try to cluster the same class
                loss_con_a += self.contrastive(torch.norm(z_p_features_topk_all[:seperate], p=1, dim=2),
                                               torch.norm(z_p_features_topk_all[seperate:], p=1, dim=2),
                                               0)

        # loss_con = self.contrastive(z_p_features_topk, z_n_features_topk, 1)  # try to separate normal and abnormal
        # loss_con_n = self.contrastive(z_n_features_topk, z_n_features_topk, 0)  # try to cluster the same class
        # loss_con_a = self.contrastive(z_p_features_topk, z_p_features_topk, 0)

        loss_total = loss_cls / len(z_n) + self.lambda_mgfn * (loss_con + loss_con_a + loss_con_n)

        return loss_total

