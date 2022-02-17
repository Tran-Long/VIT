import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Attributes:
    ----------
    n_patches: number of patches
    proj: nn.Conv2D - convolution for patches splitting and embedding
    """

    def __init__ (self, img_size, patch_size, in_channels = 3, embed_dim = 768):
        """
        :param img_size: (int) size of the img (height = width)
        :param patch_size: (int) size of patch (height = width)
        :param in_channels: (int) number of channels
        :param embed_dim: (int) embedding dimension
        """
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels = in_channels,
            out_channels = embed_dim,
            kernel_size = patch_size,
            stride = patch_size)

    def forward (self, x):
        """
        :param x: batches of img (batch_size, N, H, W)
        :return: batches of embeds (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # => (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0,5)
        x = x.flatten(2)  # => (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # => (batch_size, n_patches, embed_dim)
        return x

class Attention(nn.Module):
    """
    Attributes:
    ----------
    scale: (float) Normalizing constant for the dot product
    qkv: Linear layers - matrices
    proj: Linear - W0, matrix for reshape concatenated matrices
    attn_drop, proj_drop: Dropout
    """
    def __init__(self, dim, n_heads, qkv_bias=True, attn_p=0., proj_p=0.):
        """
        :param dim: (int) dimension of input, output
        :param n_heads: (int) number of heads
        :param qkv_bias: (bool) bias to qkv matrices
        :param attn_p: dropout rate apply to the q, k, v tensor
        :param proj_p: dropout rate apply to output tensor
        """
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.qkv_bias = qkv_bias
        self.head_dim = dim // n_heads
        self.scale = self.self.head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)  # 3 matrices
        self.attn_drop = nn.Dropout(attn_p)
        self.proj_drop = nn.Dropout(proj_p)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        :param x: batches of patches (batch_size, n_patches+1, dim)
        :return: batches of attention patches (batch_size, n_patches+1, dim)
        +1 means adding class token
        """
        batch_size, n_tokens, dim = x.shape  # n_tokens = n_patches + 1
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # => (batch_size, n_patches+1, dim*3)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = torch.permute(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # => (batch_size, n_heads, n_tokens, head_dim)
        k_t = torch.transpose(k, -2, -1)  # => (batch_size, n_heads, head_dim, n_tokens)
        dp = torch.matmul(q, k_t) * self.scale  # => (batch_size, n_heads, n_tokens, n_tokens)
        attn = torch.softmax(dp, dim=-1)
        attn = self.attn_drop(attn)
        weighted_avg = torch.matmul(attn, v)  # => (batch_size, n_heads, n_tokens, head_dim)
        weighted_avg = torch.transpose(weighted_avg, 1, 2)  # => (batch_size, n_tokens, n_heads, head_dim)
        weighted_avg = torch.flatten(weighted_avg, 2)  # => (batch_size, n_tokens, n_heads * head_dim)
        x = self.proj(weighted_avg)  # => (batch_size, n_tokens, dim)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop_out_rate=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x  # => (batch_size, n_tokens, out_features)

class Block(nn.Module):
    """
    Attributes:
    ----------
    norm1, norm2: layer normalization
    attn: Attention module
    mlp: MLP module
    """
    def __init__(self, dim, n_heads, mlp_ratio = 4, qkv_bias = True, dropout = 0., attn_p = 0.):
        """
        :param dim: same as before
        :param n_heads: same as before
        :param mlp_ratio: = hidden_features / input_features (dim)
        :param qkv_bias: same as before
        :param dropout: same as before
        :param attn_p: same as before
        """
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps = 1e-6)
        self.attn = Attention(
            dim = dim,
            n_heads = n_heads,
            qkv_bias = qkv_bias,
            attn_p = attn_p,
            proj_p = dropout
        )
        self.norm2 = nn.LayerNorm(dim, eps = 1e-6)
        hidden_features = mlp_ratio * dim
        self.mlp = MLP(dim, hidden_features, dim, drop_out_rate = dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # => (batch_size, n_tokens, dim)
        x = x + self.mlp(self.norm2(x))  # => (batch_size, n_tokens, dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=384,
            patch_size=16,
            in_channels=3,
            n_classes=1000,
            embed_dim=768,
            depth=12, # number of block
            n_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            p=0.,
            attn_p=0.
    ):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p)
        self.block = [
            Block(
                dim = embed_dim,
                n_heads = n_heads,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                dropout = p,
                attn_p = attn_p
            )
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # => (batch_size, 1, embed_dim)
        x = torch.cat([cls_token, x], dim = 1)  # => (batch_size, 1+n_patches, embed_dim)
        x = + self.pos_embed
        for blk in self.block:
            x = blk(x)
        x = self.norm(x)
        cls_token_final = x[:, 0, :]
        x = self.head(cls_token_final)
        return x