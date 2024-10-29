# ------------------------------------------------------------------
""""
Loss functions

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
from torch.nn.modules.loss import _Loss

# ------------------------------------------------------------------


class SimpleLoss(_Loss):

    def __init__(self, th_n: float = 0.5, th_p: float = 0.5):
        super(SimpleLoss, self).__init__()

        self.th_n = th_n
        self.th_p = th_p

    def forward(self, z_n_scores, z_p_scores=None, is_training=False):

        if is_training:
            true_loss = torch.clip(self.th_n - z_n_scores, min=0)
            fake_loss = torch.clip(z_p_scores + self.th_p, min=0)
            loss = true_loss.mean() + fake_loss.mean()
        else:
            true_loss = torch.clip(self.th_n - z_n_scores, min=0)
            fake_loss = torch.clip(z_p_scores + self.th_p, min=0)
            loss = true_loss.sum() + fake_loss.sum()
            loss = loss/(torch.numel(true_loss) + torch.numel(fake_loss))
        return loss

