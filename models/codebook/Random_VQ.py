# ------------------------------------------------------------------
"""
Random Projection Quantizer

Proposed in https://arxiv.org/abs/2202.01855
See: https://github.com/lucidrains/vector-quantize-pytorch
"""
# ------------------------------------------------------------------

import torch
from torch import nn, einsum
import torch.nn.functional as F
from models.codebook.VQ import VQ

from einops import rearrange, repeat, pack, unpack

def exists(val):
    return val is not None

class Random_VQ(nn.Module):
    """ https://arxiv.org/abs/2202.01855 """

    def __init__(
        self,
        *,
        dim=16,
        codebook_size=2,
        codebook_dim=16,
        num_codebooks=1,
        norm=False,
        **kwargs
    ):
        super().__init__()
        self.num_codebooks = num_codebooks

        rand_projs = torch.empty(num_codebooks, dim, codebook_dim)
        nn.init.xavier_normal_(rand_projs)

        self.register_buffer('rand_projs', rand_projs)

        # in section 3 of https://arxiv.org/abs/2202.01855
        # "The input data is normalized to have 0 mean and standard deviation of 1 ... to prevent collapse"

        self.norm = nn.LayerNorm(dim, elementwise_affine=False) if norm else nn.Identity()

        self.vq = VQ(
            dim=codebook_dim * num_codebooks,
            heads=num_codebooks,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            use_cosine_sim=False,
            learnable_codebook=False,
            separate_codebook_per_head=False,
            freeze_codebook=True,
            **kwargs
        )

    def forward(
        self,
        x,
        indices=None
    ):
        return_loss = exists(indices)

        x = self.norm(x)

        x = einsum('b n d, h d e -> b n h e', x, self.rand_projs)
        x, ps = pack([x], 'b n *')
        out = self.vq(x, indices=indices, freeze_codebook=True)

        if return_loss:
            _, ce_loss = out
            return ce_loss

        z_q, indices, loss_z_q = out
        return z_q, indices, loss_z_q



if __name__ == '__main__':

    test_z = torch.randn((2, 16, 6, 512 // 4, 832 // 4), device='cuda')

    model = Random_VQ(dim=16, codebook_size=2, codebook_dim=16).to('cuda')
    print(model)

    #opt = torch.optim.AdamW(model.parameters(), lr=0.001)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    test_z = test_z.view(2, 16, 6 * 512//4 * 832//4).permute(0, 2, 1)

    x, indices, _ = model(test_z)
    indices = indices.view(2, 6, 512//4, 832//4)


