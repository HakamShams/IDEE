# ------------------------------------------------------------------
"""
This script includes the main class to import and build the UniAD model
see: https://github.com/zhiyuanyou/UniAD

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import copy
import math
import random
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
# from initializer import initialize_from_cfg
from torch import Tensor, nn

# ------------------------------------------------------------------


def init_weights_normal(module, std=0.01):
    for m in module.modules():
        if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear)
                or isinstance(m, nn.ConvTranspose2d)
        ):
            nn.init.normal_(m.weight.data, std=std)
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_xavier(module, method):
    for m in module.modules():
        if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear)
                or isinstance(m, nn.ConvTranspose2d)
        ):
            if "normal" in method:
                nn.init.xavier_normal_(m.weight.data)
            elif "uniform" in method:
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise NotImplementedError(f"{method} not supported")
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_msra(module, method):
    for m in module.modules():
        if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear)
                or isinstance(m, nn.ConvTranspose2d)
        ):
            if "normal" in method:
                nn.init.kaiming_normal_(m.weight.data, a=1)
            elif "uniform" in method:
                nn.init.kaiming_uniform_(m.weight.data, a=1)
            else:
                raise NotImplementedError(f"{method} not supported")
            if m.bias is not None:
                m.bias.data.zero_()


def initialize(model, method, **kwargs):
    # initialize BN, Conv, & FC with different methods
    # initialize BN
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # initialize Conv & FC
    if method == "normal":
        init_weights_normal(model, **kwargs)
    elif "msra" in method:
        init_weights_msra(model, method)
    elif "xavier" in method:
        init_weights_xavier(model, method)
    else:
        raise NotImplementedError(f"{method} not supported")


def initialize_from_cfg(model, method):
    if method is None:
        initialize(model, "normal", std=0.01)
        return

    # cfg = copy.deepcopy(cfg)
    # method = cfg.pop("method")
    initialize(model, method)


class Transformer(nn.Module):
    def __init__(
            self,
            hidden_dim,
            feature_size,
            neighbor_size,
            neighbor_mask,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.neighbor_size = neighbor_size
        self.neighbor_mask = neighbor_mask

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            feature_size,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.hidden_dim = hidden_dim
        self.nhead = nhead

    def generate_mask(self, feature_size, neighbor_size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .cuda()
        )
        return mask

    def forward(self, src, pos_embed):
        _, batch_size, _ = src.shape
        pos_embed = torch.cat(
            [pos_embed.unsqueeze(1)] * batch_size, dim=1
        )  # (H X W) x B x C

        if self.neighbor_mask:
            mask = self.generate_mask(
                self.feature_size, self.neighbor_size
            )
            mask_enc = mask if self.neighbor_mask[0] else None
            mask_dec1 = mask if self.neighbor_mask[1] else None
            mask_dec2 = mask if self.neighbor_mask[2] else None
        else:
            mask_enc = mask_dec1 = mask_dec2 = None

        output_encoder = self.encoder(
            src, mask=mask_enc, pos=pos_embed
        )  # (H X W) x B x C
        output_decoder = self.decoder(
            output_encoder,
            tgt_mask=mask_dec1,
            memory_mask=mask_dec2,
            pos=pos_embed,
        )  # (H X W) x B x C

        return output_decoder, output_encoder


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
            self,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        output = memory

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            hidden_dim,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            hidden_dim,
            feature_size,
            nhead,
            dim_feedforward,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()
        num_queries = feature_size[0] * feature_size[1]
        self.learned_embed = nn.Embedding(num_queries, hidden_dim)  # (H x W) x C

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            out,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
            self,
            out,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
            self,
            out,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                out,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
            )
        return self.forward_post(
            out,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
            self,
            feature_size,
            num_pos_feats=128,
            temperature=10000,
            normalize=False,
            scale=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))  # H x W
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return pos.to(tensor.device)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size  # H, W
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)  # W
        j = torch.arange(self.feature_size[0], device=tensor.device)  # H
        x_emb = self.col_embed(i)  # W x C // 2
        y_emb = self.row_embed(j)  # H x C // 2
        pos = torch.cat(
            [
                torch.cat(
                    [x_emb.unsqueeze(0)] * self.feature_size[0], dim=0
                ),  # H x W x C // 2
                torch.cat(
                    [y_emb.unsqueeze(1)] * self.feature_size[1], dim=1
                ),  # H x W x C // 2
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (H X W) X C
        return pos


def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    if pos_embed_type in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"):
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed


class UniAD(nn.Module):
    def __init__(
            self,
            inplanes=16 * 6,
            instrides=2,
            feature_size=(100, 100),
            feature_jitter_scale=1,
            feature_jitter_prob=1.,
            neighbor_size=(7, 7),
            neighbor_mask=[True, True, True],
            hidden_dim=16 * 6,
            pos_embed_type='learned',
            # save_recon=False,
            initializer='xavier_uniform',
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=16 * 6 * 2,
            dropout=0.1,
            activation='relu',
            normalize_before=False,
            return_intermediate_dec=False,
            en_de_pretrained=None
    ):
        super().__init__()
        #    assert isinstance(inplanes, list) and len(inplanes) == 1
        #    assert isinstance(instrides, list) and len(instrides) == 1
        self.inplanes = inplanes
        self.instrides = instrides
        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]
        self.feature_jitter_scale = feature_jitter_scale
        self.feature_jitter_prob = feature_jitter_prob

        self.pos_embed = build_position_embedding(
            pos_embed_type, feature_size, hidden_dim
        )
        # self.save_recon = save_recon

        self.transformer = Transformer(
            hidden_dim=hidden_dim, feature_size=feature_size, neighbor_size=neighbor_size, neighbor_mask=neighbor_mask,
            nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            normalize_before=normalize_before, return_intermediate_dec=return_intermediate_dec,
        )
        self.input_proj = nn.Linear(inplanes, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, inplanes)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides)

        initialize_from_cfg(self, initializer)

        self.pretrained = en_de_pretrained
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

    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                    feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda()
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens


    def forward(self, feature_align, mask_extreme_loss=None):

        # feature_align = input["feature_align"]  # B x C X H x W
        feature_align = F.interpolate(feature_align, scale_factor=1 / self.instrides, mode='bilinear')

        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"
        )  # (H x W) x B x C
        if self.training and self.feature_jitter_scale:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter_scale, self.feature_jitter_prob
            )

        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C
        pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C
        output_decoder, _ = self.transformer(
            feature_tokens, pos_embed
        )  # (H x W) x B x C
        feature_rec_tokens = self.output_proj(output_decoder)  # (H x W) x B x C
        feature_rec = rearrange(
            feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W

        """
        if not self.training and self.save_recon:
            clsnames = input["clsname"]
            filenames = input["filename"]
            for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
                filedir, filename = os.path.split(filename)
                _, defename = os.path.split(filedir)
                filename_, _ = os.path.splitext(filename)
                save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
                os.makedirs(save_dir, exist_ok=True)
                feature_rec_np = feat_rec.detach().cpu().numpy()
                np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)

        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W
        pred = self.upsample(pred)  # B x 1 x H x W
        return {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
        }
             """
        loss = (feature_rec - feature_align) ** 2

        if mask_extreme_loss is not None:
            loss = self.upsample(loss)
            loss = loss.permute(1, 0, 2, 3)
            #    mask_extreme_loss = F.interpolate(mask_extreme_loss, scale_factor=1/self.instrides, mode='bilinear')
            loss[:, mask_extreme_loss == 1] = loss[:, mask_extreme_loss == 1] * -1
            loss = loss.permute(1, 0, 2, 3)
        # loss = torch.mean(loss)

        # feature_rec = self.upsample(feature_rec)  # B x 1 x H x W

        # return feature_rec, loss
        return loss


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

    test_dynamic = torch.randn((1, 6, 200, 200), device=device)
    #test_mask_extreme = torch.randint(1, (1, 200, 200)).long().to(device)

    model = UniAD(inplanes=config_file.inplanes, instrides=config_file.instrides, feature_size=config_file.feature_size,
                  feature_jitter_scale=config_file.feature_jitter_scale, feature_jitter_prob=config_file.feature_jitter_prob,
                  neighbor_size=config_file.neighbor_size, neighbor_mask=config_file.neighbor_mask, hidden_dim=config_file.hidden_dim,
                  pos_embed_type=config_file.pos_embed_type, initializer=config_file.initializer,
                  nhead=config_file.nhead, num_encoder_layers=config_file.num_encoder_layers,
                  num_decoder_layers=config_file.num_decoder_layers, dim_feedforward=config_file.dim_feedforward,
                  dropout=config_file.dropout, activation=config_file.activation, normalize_before=config_file.normalize_before,
                  return_intermediate_dec=config_file.return_intermediate_dec, en_de_pretrained=config_file.en_de_pretrained
                  ).to(device)

    print(model)
    n_parameters = sum(p.numel() for p in model.parameters()) # if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    model.eval()

    p_scores = model(test_dynamic.to(device), None)
    print(p_scores.shape)

