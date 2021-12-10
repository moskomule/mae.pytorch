""" A PyTorch implementation of MaskedAutoEncoder.

Transformer is based on moskomule/simple_transformer and
some tricks can be attributed to lucidrains/vit-pytorch

"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional, Callable, Type

import torch
from torch import einsum, nn

LayerNorm = partial(nn.LayerNorm, eps=1e-6)


def _to_tuple(x, size=2) -> tuple:
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return tuple([x for _ in range(size)])


def dotproduct_self_attention(query: torch.Tensor,
                              key: torch.Tensor,
                              value: torch.Tensor,
                              dropout: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                              ) -> torch.Tensor:
    context = einsum("bnhk,bmhk->bhmn", query, key).div(math.sqrt(query.size(-1)))
    context = context.softmax(dim=-1)
    if dropout is not None:
        context = dropout(context)
    return einsum("bhmn,bnhv->bmhv", context, value)


class SelfAttention(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_heads: int,
                 attn_dropout_rate: float,
                 proj_dropout_rate: float,
                 qkv_bias: bool = True,
                 proj_bias: bool = True,
                 ):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=qkv_bias)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=proj_bias)
        self.attn_dropout = nn.Identity() if attn_dropout_rate == 0 else nn.Dropout(attn_dropout_rate)
        self.proj_dropout = nn.Identity() if proj_dropout_rate == 0 else nn.Dropout(proj_dropout_rate)
        self.attn_fn = dotproduct_self_attention

    def forward(self,
                input: torch.Tensor,
                ) -> torch.Tensor:
        # input: BxNxC
        b, n, c = input.size()
        # BxNx3C -> BxNxHxC'
        query, key, value = self.qkv(input).view(b, n, 3, self.num_heads, -1).unbind(2)
        attention = self.attn_fn(query, key, value, self.attn_dropout).reshape(b, -1, self.emb_dim)
        return self.proj_dropout(self.proj(attention))


class PatchEmbed2d(nn.Module):
    def __init__(self,
                 patch_size: int or tuple,
                 emb_dim: int,
                 in_channels: int
                 ):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        # input: BxCxHxW -> BxNxC'
        return self.proj(input).flatten(2).transpose(1, 2)


class Block(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 attention: SelfAttention,
                 dropout_rate: float,
                 widen_factor: int = 4,
                 activation: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.ln1 = LayerNorm(emb_dim)
        self.ln2 = LayerNorm(emb_dim)
        self.attention = attention
        self.mlp = nn.Sequential(nn.Linear(emb_dim, widen_factor * emb_dim),
                                 activation(),
                                 nn.Linear(widen_factor * emb_dim, emb_dim),
                                 nn.Dropout(dropout_rate))

    def forward(self,
                input: torch.Tensor,
                ) -> torch.Tensor:
        x = input
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class VisionTransformer(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 num_layers: int,
                 num_heads: int,
                 patch_size: int,
                 image_size: int,
                 num_classes: int,
                 dropout_rate: float = 0,
                 attn_dropout_rate: float = 0,
                 in_channels: int = 3,
                 mlp_widen_factor: int = 4,
                 mean_pooling: bool = False
                 ):
        super().__init__()

        self.emb_dim = emb_dim
        self.image_size = _to_tuple(image_size)
        self.patch_size = _to_tuple(patch_size)
        self.num_patches = math.prod(self.image_size) // math.prod(self.patch_size)
        self.mean_pooling = mean_pooling

        self.patch_emb = PatchEmbed2d(self.patch_size, emb_dim, in_channels)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches + int(not self.mean_pooling), emb_dim))

        if not self.mean_pooling:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.Sequential(
            *[Block(emb_dim,
                    SelfAttention(emb_dim, num_heads, attn_dropout_rate, dropout_rate), dropout_rate, mlp_widen_factor)
              for _ in range(num_layers)
              ])
        self.norm = LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)

    def add_cls_token(self,
                      x: torch.Tensor
                      ) -> torch.Tensor:
        b, n, c = x.size()
        # torch.cat([cls_token, x]), but can be efficient
        tmp = x.new_empty(b, n + 1, c)
        tmp[:, :1].copy_(self.cls_token)
        tmp[:, 1:].copy_(x)
        return tmp

    def forward(self,
                input: torch.Tensor
                ) -> torch.Tensor:
        x = self.patch_emb(input)  # BxNxC
        if not self.mean_pooling:
            x = self.add_cls_token(x)
        x = self.dropout(self.pos_emb + x)
        x = self.norm(self.blocks(x))
        x = x.mean(dim=1) if self.mean_pooling else x[:, 0]  # BxC
        return self.fc(x)  # Bx{num_classes}


def vit_b16(**kwargs) -> VisionTransformer:
    return VisionTransformer(768, 12, 12, 16, 224, 1_000, **kwargs)


class MaskedAutoEncoder(nn.Module):
    def __init__(self,
                 encoder: VisionTransformer,
                 dec_ebm_dim: int,
                 dec_depth: int,
                 dec_num_heads: int,
                 mask_ratio: float,
                 ):
        super().__init__()
        self.encoder = encoder
        self.enc_num_patches = self.encoder.num_patches
        self.decoder = nn.Sequential(*[Block(dec_ebm_dim,
                                             SelfAttention(dec_ebm_dim, dec_num_heads, 0, 0),
                                             0, widen_factor=4)
                                       for _ in range(dec_depth)])
        self.dec_pos_emb = nn.Embedding(self.enc_num_patches, dec_ebm_dim)
        self.mask_emb = nn.Parameter(torch.randn(1, dec_ebm_dim))
        self.enc_to_dec = nn.Linear(self.encoder.emb_dim, dec_ebm_dim)
        patch = self.encoder.patch_emb.proj
        self.unfold = nn.Unfold(patch.kernel_size, stride=patch.stride)
        self.pred = nn.Linear(dec_ebm_dim, 3 * math.prod(encoder.patch_size))
        self.mask_ratio = mask_ratio
        self.init_weights()

    def forward(self,
                input: torch.Tensor
                ) -> tuple[torch.Tensor, ...]:
        # input: BxCxHxW
        tokens = self.encoder.patch_emb(input)  # BxNxC
        if not self.encoder.mean_pooling:
            tokens = self.encoder.add_cls_token(tokens)
        tokens = tokens + self.encoder.pos_emb

        num_masked = int(self.mask_ratio * self.enc_num_patches)
        rand_idx = torch.rand(input.size(0), self.enc_num_patches, device=input.device).argsort(dim=1)
        # masked_idx: BxN''x1, unmasked_idx: BxN'x1
        masked_idx, unmasked_idx = rand_idx.split((num_masked, self.enc_num_patches - num_masked), dim=1)

        unmasked_tokens = tokens.take_along_dim(unmasked_idx[:, :, None], dim=1)  # BxN'xC
        unmasked_tokens = self.encoder.blocks(unmasked_tokens)  # BxN'xC
        unmasked_tokens = self.encoder.norm(unmasked_tokens)

        unmasked_tokens = self.dec_pos_emb(unmasked_idx) + self.enc_to_dec(unmasked_tokens)  # BxN'xC'
        masked_tokens = self.dec_pos_emb(masked_idx) + self.mask_emb  # BxN''xC'

        dec_tokens = torch.cat((masked_tokens, unmasked_tokens), dim=1)  # BxNxC'
        dec_tokens = self.decoder(dec_tokens)[:, :num_masked]  # BxN''xC'
        masked_pred = self.pred(dec_tokens)  # BxN''x{3xPxP}
        masked_gt = self.unfold(input).transpose(1, 2).take_along_dim(masked_idx[:, :, None], dim=1)  # BxN''x{3xPxP}

        return masked_pred, masked_gt

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        proj_w = self.encoder.patch_emb.proj
        fan_in = proj_w.in_channels * math.prod(proj_w.kernel_size)
        nn.init.trunc_normal_(proj_w.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(proj_w.bias)
        nn.init.trunc_normal_(self.encoder.pos_emb, std=0.02)
        if not self.encoder.mean_pooling:
            nn.init.trunc_normal_(self.encoder.cls_token, std=0.02)

    @property
    def param_groups(self):
        lns = [m for m in self.modules() if isinstance(m, nn.LayerNorm)]
        no_wd = {m.weight for m in lns} | {m.bias for m in lns}
        wd = {p for p in self.parameters() if p not in no_wd}
        return {'decay': list(wd), 'no_decay': list(no_wd)}
