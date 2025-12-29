import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Patch嵌入层"""

    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, embed_dim, H/patch_size, W/patch_size
        x = x.flatten(2).transpose(1, 2)  # B, N, embed_dim
        return x


class CrossAttention(nn.Module):
    """双向交叉注意力模块"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape

        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BidirectionalCrossAttention(nn.Module):
    """双向交叉注意力网络"""

    def __init__(self, dim, num_heads, patch_size=16):
        super(BidirectionalCrossAttention, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size, dim, dim)
        self.cross_attn_linear = CrossAttention(dim, num_heads)
        self.cross_attn_circular = CrossAttention(dim, num_heads)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, linear_feat, circular_feat):
        # Patch embedding
        linear_tokens = self.patch_embedding(linear_feat)
        circular_tokens = self.patch_embedding(circular_feat)

        # 正向注意力：linear → circular
        attn_linear = self.cross_attn_linear(
            self.norm1(linear_tokens),
            self.norm1(circular_tokens),
            self.norm1(circular_tokens)
        )
        linear_tokens = linear_tokens + attn_linear
        linear_tokens = linear_tokens + self.mlp(self.norm2(linear_tokens))

        # 逆向注意力：circular → linear
        attn_circular = self.cross_attn_circular(
            self.norm1(circular_tokens),
            self.norm1(linear_tokens),
            self.norm1(linear_tokens)
        )
        circular_tokens = circular_tokens + attn_circular
        circular_tokens = circular_tokens + self.mlp(self.norm2(circular_tokens))

        # 重构特征图
        B, N, C = linear_tokens.shape
        H = W = int(math.sqrt(N))

        linear_feat = linear_tokens.transpose(1, 2).reshape(B, C, H, W)
        circular_feat = circular_tokens.transpose(1, 2).reshape(B, C, H, W)

        return linear_feat, circular_feat

