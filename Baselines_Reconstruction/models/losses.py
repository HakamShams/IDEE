# ------------------------------------------------------------------
"""
Loss functions

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch.nn as nn
from torch.nn.modules.loss import _Loss

# ------------------------------------------------------------------


class STEALLoss(_Loss):
    def __init__(self, n_dynamic: int = 6):
        super(STEALLoss, self).__init__()

        self.n_dynamic = n_dynamic
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, pred, target, mask):

        mask = mask.unsqueeze(1).repeat(1, self.n_dynamic, 1, 1, 1)

        ind_extreme = (mask != 0)
        loss_n = self.loss(pred[~ind_extreme], target[~ind_extreme])
        loss_p = -self.loss(pred[ind_extreme], target[ind_extreme])
        loss = loss_n + loss_p

        return loss


class UniADLoss(_Loss):
    def __init__(self):
        super(UniADLoss, self).__init__()

        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, pred, target):
        return self.loss(pred, target)

