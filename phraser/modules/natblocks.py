"""
Neighborhood Attention Transformer.
To appear in CVPR 2023.
https://arxiv.org/abs/2204.07143

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

SOURCE =============== https://github.com/SHI-Labs/Neighborhood-Attention-Transformer ============
"""
from typing import Optional

import torch  # noqa: F401
from natten.functional import neighborhood_attention_generic
from natten.types import DimensionTypeOrDed, CausalArgTypeOrDed
from natten.utils.checks import check_all_args
from torch import nn, Tensor

from phraser.modules.drop import DropPath
from phraser.modules.rope import RotaryEmbedding


class NeighborhoodAttentionGeneric(nn.Module):
    def __init__(
        self,
        na_dim: int,
        embed_dim: int,
        num_heads: int,
        kernel_size: DimensionTypeOrDed,
        stride: DimensionTypeOrDed = 1,
        dilation: DimensionTypeOrDed = 1,
        is_causal: CausalArgTypeOrDed = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        kernel_size, stride, dilation, is_causal = check_all_args(
            na_dim, kernel_size, stride, dilation, is_causal
        )

        if embed_dim % num_heads != 0:
            raise ValueError(
                "Number of attention heads must evenly divide embedding dimension, "
                f"got {embed_dim=}, {num_heads=}."
            )

        self.na_dim = na_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.is_causal = is_causal

        self.expected_input_tensor_rank = self.na_dim + 2  # batch, embedding dim

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, rope: RotaryEmbedding) -> Tensor:
        if x.dim() != self.expected_input_tensor_rank:
            raise ValueError(
                f"NeighborhoodAttention{self.na_dim}D expected a tensor with rank "
                f"{self.expected_input_tensor_rank} ({self.na_dim} for token layout, 1 for batch, "
                f"1 for embedding dimension), got {x.dim()=}."
            )

        B, *input_shape, C = x.shape

        if C != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embed_dim}, got {C} ({x.shape=})."
            )

        # 3, batch, *input_shape, heads, head_dim
        permutation = (
            [self.na_dim + 1, 0]
            + [x + 1 for x in range(self.na_dim)]
            + [self.na_dim + 2, self.na_dim + 3]
        )
        qkv = (
            self.qkv(x)
            .reshape(B, *input_shape, 3, self.num_heads, self.head_dim)
            .permute(*permutation)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = rope.rotate_queries_or_keys(q)
        k = rope.rotate_queries_or_keys(k)

        x = neighborhood_attention_generic(
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            is_causal=self.is_causal,
            scale=self.scale,
        )
        x = x.reshape(B, *input_shape, C)

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, "
            + f"stride={self.stride}, "
            + f"dilation={self.dilation}, "
            + f"is_causal={self.is_causal}"
        )


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NAT2DLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.RMSNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttentionGeneric(
            na_dim=2,
            embed_dim=dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_drop=drop

        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x, rope):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x, rope)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, rope)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NAT2DBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        drop_path=0.0,
        norm_layer=nn.RMSNorm,
        layer_scale=None,
        rope = None
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads

        self.blocks = nn.ModuleList(
            [
                NAT2DLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ]
        )

        if rope is None:
            self.rope = RotaryEmbedding(dim = min(int(self.dim / num_heads // 2), 32))
        else:
            self.rope = rope

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x, self.rope)

        return x



class NAT1DLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.RMSNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttentionGeneric(
            na_dim=1,
            embed_dim=dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_drop=drop

        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x, rope):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x, rope)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, rope)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NAT1DBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        drop_path=0.0,
        norm_layer=nn.RMSNorm,
        layer_scale=None,
        rope=None
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads

        self.blocks = nn.ModuleList(
            [
                NAT1DLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ]
        )

        if rope is None:
            self.rope = RotaryEmbedding(dim = min(int(self.dim / num_heads // 2), 32))
        else:
            self.rope = rope


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x, self.rope)

        return x
