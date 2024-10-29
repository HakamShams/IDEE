# ------------------------------------------------------------------
"""
Magnitude-Contrastive Glance-and-Focus Network for Weakly-Supervised Video Anomaly Detection
MGFN https://arxiv.org/abs/2211.15098

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange

# --------------------------------------------------------


class InstanceDropout(nn.Module):
    """ Dropout without rescaling """

    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate == 0:
            return x
        else:
            mask = torch.empty(x.shape, device=x.device, requires_grad=False).bernoulli_(1 - self.drop_rate)
            return x * mask


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


def attention(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


def FeedForward(dim, repe=4, dropout=0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv1d(dim, dim * repe, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(dim * repe, dim, 1)
    )


# MHRAs (multi-head relation aggregators)
class FOCUS(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            dim_head=16,
            local_aggr_kernel=5
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.BatchNorm1d(dim)
        self.to_v = nn.Conv1d(dim, inner_dim, 1, bias=False)
        self.rel_pos = nn.Conv1d(heads, heads, local_aggr_kernel, padding=local_aggr_kernel // 2, groups=heads)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)  # (b*crop,c,t)
        b, c, *_, h = *x.shape, self.heads
        v = self.to_v(x)  # (b*crop,c,t)
        v = rearrange(v, 'b (c h) ... -> (b c) h ...', h=h)  # (b*ten*64,c/64,32)
        out = self.rel_pos(v)
        out = rearrange(out, '(b c) h ... -> b (c h) ...', b=b)
        return self.to_out(out)


class GLANCE(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)
        self.attn = 0

    def forward(self, x):
        x = self.norm(x)
        shape, h = x.shape, self.heads
        x = rearrange(x, 'b c ... -> b c (...)')
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        self.attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', self.attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n', h=h)
        out = self.to_out(out)

        return out.view(*shape)


class Backbone(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            heads,
            mgfn_type='gb',
            kernel=5,
            dim_headnumber=64,
            ff_repe=4,
            dropout=0.,
            attention_dropout=0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mgfn_type == 'fb':
                attention = FOCUS(dim, heads=heads, dim_head=dim_headnumber, local_aggr_kernel=kernel)
            elif mgfn_type == 'gb':
                attention = GLANCE(dim, heads=heads, dim_head=dim_headnumber, dropout=attention_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv1d(dim, dim, 3, padding=1),
                attention,
                FeedForward(dim, repe=ff_repe, dropout=dropout),
            ]))

    def forward(self, x):
        for scc, attention, ff in self.layers:
            x = scc(x) + x
            x = attention(x) + x
            x = ff(x) + x

        return x


class MGFN(nn.Module):
    def __init__(self, embed_dim: int = 16,
                 dim: list = None,
                 drop_rate: float = 0.,
                 drop_rate_instance: float = 0.6,
                 alpha: float = 0.1,
                 depths: list = [2, 2],
                 mgfn_types: list = ['gb', 'fb'],
                 lokernel: int = 5,
                 ff_repe: int = 4,
                 dim_head: list = [16, 64],
                 attention_drop_rate: float = 0.
                 ):

        super(MGFN, self).__init__()

        self.embed_dim = embed_dim
        self.dim = dim
        self.dim_head = dim_head
        self.drop_rate = drop_rate
        self.attention_drop_rate = attention_drop_rate
        self.alpha = alpha
        self.ff_repe = ff_repe
        self.lokernel = lokernel

        self.to_mag = nn.Conv1d(1, embed_dim, kernel_size=3, stride=1, padding=1)

        mgfn_types = tuple(map(lambda t: t.lower(), mgfn_types))

        self.stages = nn.ModuleList([])

        for ind, (depth, mgfn_types) in enumerate(zip(depths, mgfn_types)):
            is_last = ind == len(depths) - 1
            stage_dim = self.dim[ind]
            heads = stage_dim // dim_head[ind]

            self.stages.append(nn.ModuleList([
                Backbone(
                    dim=stage_dim,
                    depth=depth,
                    heads=heads,
                    mgfn_type=mgfn_types,
                    ff_repe=ff_repe,
                    dropout=drop_rate,
                    attention_dropout=attention_drop_rate
                ),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv1d(stage_dim, self.dim[ind + 1], 1, stride=1),
                ) if not is_last else None
            ]))

        self.to_logits = nn.LayerNorm(dim[-2])
        self.fc = nn.Linear(dim[-2], 1)
        self.sigmoid = nn.Sigmoid()

   #     self.drop_out_instance = InstanceDropout(drop_rate_instance)

    def forward(self, x):

        #  x Np, V, 1, C

        N, V, T, C = x.shape

        x = x.reshape(N * V, T, C).permute(0, 2, 1)  # N*V, C, T

        x_m = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x + self.alpha * self.to_mag(x_m)

        for backbone, conv in self.stages:
            x = backbone(x)
            if conv is not None:
                x = conv(x)

        x = x.permute(0, 2, 1)

        x = self.to_logits(x)  # N*V, T, C
    #    x = self.drop_out_instance(x)
        scores = self.sigmoid(self.fc(x))  # N*V, T, 1

        x = x.view(N, V, T, -1)  # N, V, T, 64
        scores = scores.view(N, V, T, 1)  # N, V, T, 1

        return x, scores


def MSNSD(features, scores, bs, batch_size, drop_out, ncrops, k):
    # magnitude selection and score prediction
    features = features  # (B*10crop,32,1024)
    bc, t, f = features.size()

    scores = scores.view(bs, ncrops, -1).mean(1)  # (B,32)
    scores = scores.unsqueeze(dim=2)  # (B,32,1)

    normal_features = features[0:batch_size * ncrops]  # [b/2*ten,32,1024]
    normal_scores = scores[0:batch_size]  # [b/2, 32,1]

    abnormal_features = features[batch_size * ncrops:]
    abnormal_scores = scores[batch_size:]

    feat_magnitudes = torch.norm(features, p=2, dim=2)  # [b*ten,32]
    feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # [b,32]
    nfea_magnitudes = feat_magnitudes[0:batch_size]  # [b/2,32]  # normal feature magnitudes
    afea_magnitudes = feat_magnitudes[batch_size:]  # abnormal feature magnitudes
    n_size = nfea_magnitudes.shape[0]  # b/2

    if nfea_magnitudes.shape[0] == 1:  # this is for inference
        afea_magnitudes = nfea_magnitudes
        abnormal_scores = normal_scores
        abnormal_features = normal_features

    select_idx = torch.ones_like(afea_magnitudes).cuda()
    #select_idx = drop_out(select_idx)

    afea_magnitudes_drop = afea_magnitudes * select_idx
    idx_abn = torch.topk(afea_magnitudes_drop, k, dim=1)[1]
    idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

    abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
    abnormal_features = abnormal_features.permute(1, 0, 2, 3)

    total_select_abn_feature = torch.zeros(0).to('cuda')
    for abnormal_feature in abnormal_features:
        feat_select_abn = torch.gather(abnormal_feature, 1,
                                       idx_abn_feat)
        total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))  #

    idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])  #
    score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score),
                                dim=1)

    select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
    #select_idx_normal = drop_out(select_idx_normal)
    nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
    idx_normal = torch.topk(nfea_magnitudes_drop, k, dim=1)[1]
    idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

    normal_features = normal_features.view(n_size, ncrops, t, f)
    normal_features = normal_features.permute(1, 0, 2, 3)

    total_select_nor_feature = torch.zeros(0).to('cuda')
    for nor_fea in normal_features:
        feat_select_normal = torch.gather(nor_fea, 1,
                                          idx_normal_feat)
        total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

    idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
    score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

    abn_feamagnitude = total_select_abn_feature
    nor_feamagnitude = total_select_nor_feature

    return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores


if __name__ == '__main__':

    device = 'cuda'

    x = torch.randn((1 * 16 * 2, 10, 1, 16), device=device)

    model = MGFN(embed_dim=16,
                 dim=[16, 96, 1],
                 drop_rate=0.,
                 drop_rate_instance=0.6,
                 alpha=0.1,
                 dim_head=[16, 16],
                 depths=[2, 2],
                 mgfn_types=['gb', 'fb'],
                 lokernel=5,
                 ff_repe=4,
                 attention_drop_rate=0.
                 ).to(device)

    print(model)

    feature, test = model(x)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")
    print(feature.shape)
    print(test.shape)

    feature = feature.view(32*10, 1, 96).repeat(1, 32, 1)
    test = test.view(32*10, 1, 1).repeat(1, 32, 1)

    print(test.shape)

    score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = MSNSD(feature,test,32,1,0,10,3)
    print()
    print(score_normal.shape)
    print(score_abnormal.shape)
    print(abn_feamagnitude.shape)

    print(scores.shape)


