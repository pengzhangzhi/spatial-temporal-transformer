'''=================================================

@Project -> File：ST-Transformer->ViT

@IDE：PyCharm

@coding: utf-8

@time:2021/7/24 7:11

@author:Pengzhangzhi

@Desc：
=================================================='''
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
from torch.nn import functional as F, init


# helpers
class iLayer(nn.Module):
    '''    elementwise multiplication
    '''

    def __init__(self, input_shape):
        '''
        input_shape: (,*,c,,h,w)
        self.weights shape: (,*,c,h,w)
        '''
        super(iLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(*input_shape))  # define the trainable parameter
        # init.xavier_uniform_(self.weights.data)

    def forward(self, x):
        '''
        x: (batch, c, h,w)
        self.weights shape: (c,h,w)
        output: (batch, c, h,w)
        '''
        return x * self.weights


def pair(t):
    return t if isinstance(t, list) else [t, t]


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Softmax(nn.Module):
    def __init__(self, normlization_scale):
        super(Softmax, self).__init__()
        self.normlization_scale = normlization_scale

    def forward(self, x):
        return F.softmax(x / (self.normlization_scale ** 0.5), dim=1)


class ViT(nn.Module):

    def __init__(self, *,
                 image_size,
                 patch_size,
                 num_classes,
                 dim, depth, heads,
                 mlp_dim,
                 pool='mean',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0., seq_pool=True):
        """

        args:
        :param image_size: input map size
        :param patch_size: patch size
        :param num_classes: output size
        :param dim: embedding dimension.
        :param depth: num Of Transformer Block
        :param heads: number of head
        :param mlp_dim: dimension of MLP in transformer.
        :param pool: the way of pooling transformer output.
        :param channels: number of channels in input tensor.
        :param dim_head: embedding dimension of a head.
        :param dropout: dropout probability
        :param emb_dropout: dropout probability in transformer
        :param seq_pool: weather to use sequence pooling after transformer block.
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.dim = dim
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        mlp_dim = dim
        # if seq_pool:
        #     mlp_dim = dim * num_patches
        # else:
        #     mlp_dim = dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, num_classes),
            # nn.ReLU(),
            # nn.Linear(num_classes, num_classes)
        )
        self.seq_pool = seq_pool
        if seq_pool:
            self.attention_pool = nn.Linear(dim, 1)
            self.atten_layer_norm = nn.LayerNorm([self.num_patches, dim])
            self.attention_pool = nn.Sequential(self.atten_layer_norm,self.attention_pool)
            self.softmax_scale = dim
            self.softmax = Softmax(normlization_scale=self.softmax_scale)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.seq_pool:
            x = torch.matmul(self.softmax(self.attention_pool(x)).transpose(-1, -2), x).squeeze(-2)
            # attention_sequence = self.softmax(self.attention_pool(x))
            # attention_sequence = repeat(attention_sequence, "b n d -> b n (d dim)", dim=self.dim)
            # x = x * attention_sequence
            # x = rearrange(x, "b n d -> b (n d)")


        else:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        return x


if __name__ == '__main__':
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=128,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        seq_pool=True
    )

    # import Recorder and wrap the ViT

    # forward pass now returns predictions and the attention maps

    img = torch.randn(1, 3, 256, 256)
    preds = v(img)
    print(preds.shape)
    # there is one extra patch due to the CLS token 64 + 1

    # from vit_pytorch.cct import CCT
