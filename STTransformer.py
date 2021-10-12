'''=================================================

@Project -> File：ST-Transformer->STTransformer

@IDE：PyCharm

@coding: utf-8

@time:2021/7/23 17:01

@author:Pengzhangzhi

@Desc：
=================================================='''
import os
import sys

import torch
import torch.nn as nn
from einops import rearrange, reduce
from torch.nn import init

from arg_convertor import arg_class2dict
from base_layers import BasicBlock
from help_funcs import summary, Logger
from vit import ViT


class Rc(nn.Module):
    def __init__(self, input_shape):
        super(Rc, self).__init__()
        self.nb_flow = input_shape[0]
        self.ilayer = iLayer(input_shape)

    def forward(self, x):
        """
            x: (*, c, h, w)
          out: (*, 2, h ,w)
        """
        # x = rearrange(x,"b (nb_flow c) h w -> b nb_flow c h w",nb_flow=self.nb_flow)
        # x = reduce(x,"b nb_flow c h w -> b nb_flow h w","sum")
        x = reduce(x, "b (c1 c) h w -> b c1 h w", "sum", c1=self.nb_flow)
        out = self.ilayer(x)
        return out


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
        init.xavier_uniform_(self.weights.data)

    def forward(self, x):
        '''
        x: (batch, c, h,w)
        self.weights shape: (c,h,w)
        output: (batch, c, h,w)
        '''
        return x * self.weights


class STTransformer(nn.Module):
    def __init__(self, map_height=32, map_width=32, patch_size=4,
                 close_channels=6, trend_channels=6, close_dim=1024, trend_dim=1024,
                 close_depth=4, trend_depth=4, close_head=2,
                 trend_head=2, close_mlp_dim=2048, trend_mlp_dim=2048, nb_flow=2,
                 seq_pool=True,
                 pre_conv=True,
                 shortcut=True,
                 conv_channels=64,
                 drop_prob=0.1,
                 conv3d=False,
                 time_class=48,
                 **kwargs):
        """

        :param seq_pool: whether to use sequence pooling.
        :param pre_conv: whether to use pre-conv
        :param conv_channels: number of channels inside pre_conv.
        :param map_height:
        :param map_width:
        :param patch_size:
        :param close_channels: number of channels in Xc,
        :param trend_channels: number of channels in Xc,
        :param close_dim: embedding dimension of closeness component.
        :param trend_dim: embedding dimension of trend component.
        :param close_depth: number of transformer in closeness component
        :param trend_depth: number of transformer in trend component
        :param close_head: number of head in closeness component
        :param trend_head: number of head in trend component
        :param close_mlp_dim: embedding dimension of a head in closeness component
        :param trend_mlp_dim: embedding dimension of a head in trend component
        :param nb_flow: number of flow.
        :param kwargs: filter out useless args.
        """
        super(STTransformer, self).__init__()
        self.map_height = map_height
        self.map_width = map_width
        self.nb_flow = nb_flow
        output_dim = nb_flow * map_height * map_width
        close_dim_head = int(close_dim / close_head)
        trend_dim_head = int(trend_dim / close_head)

        self.pre_conv = pre_conv
        self.conv3d = conv3d
        if pre_conv:
            self.pre_close_conv = nn.Sequential(
                BasicBlock(inplanes=close_channels, planes=conv_channels),
                # BasicBlock(inplanes=close_channels,planes=conv_channels),
            )
            self.pre_trend_conv = nn.Sequential(
                BasicBlock(inplanes=trend_channels, planes=conv_channels),
                # BasicBlock(inplanes=trend_channels,planes=conv_channels)
            )

        # close_channels, trend_channels = nb_flow * close_channels, nb_flow * trend_channels

        self.closeness_transformer = ViT(
            image_size=[map_height, map_width],
            patch_size=patch_size,
            num_classes=output_dim,
            dim=close_dim,
            depth=close_depth,
            heads=close_head,
            mlp_dim=close_mlp_dim,
            dropout=drop_prob,
            emb_dropout=drop_prob,
            channels=close_channels,
            dim_head=close_dim_head,
            seq_pool=seq_pool
        )
        self.trend_transformer = ViT(
            image_size=[map_height, map_width],
            patch_size=patch_size,
            num_classes=output_dim,
            dim=trend_dim,
            depth=trend_depth,
            heads=trend_head,
            mlp_dim=trend_mlp_dim,
            dropout=drop_prob,
            emb_dropout=drop_prob,
            channels=trend_channels,
            dim_head=trend_dim_head,
            seq_pool=seq_pool,

        )
        input_shape = (nb_flow, map_height, map_width)

        self.shortcut = shortcut
        if shortcut:
            self.Rc_Xc = Rc(input_shape)
            self.Rc_Xt = Rc(input_shape)
            # self.Rc_conv_Xc = Rc(input_shape)
            # self.Rc_conv_Xt = Rc(input_shape)

        self.close_ilayer = iLayer(input_shape=input_shape)
        self.trend_ilayer = iLayer(input_shape=input_shape)

        self.mlp = nn.Sequential(
            nn.Linear((close_dim + trend_dim), time_class)
        )

        self.mlp_head_close = nn.Sequential(
            nn.Linear(close_dim, output_dim),
        )
        self.mlp_head_trend = nn.Sequential(
            nn.Linear(trend_dim, output_dim),
        )

    def forward(self, xc, xt, x_ext=None):
        """

        :param xc: batch size, num_close,map_height,map_width
        :param xt: batch size, num_week,map_height,map_width
        :return:
        """
        if len(xc.shape) == 5:
            # reshape 5 dimensions to 4 dimensions.
            xc, xt = list(map(lambda x: rearrange(x, "b n l h w -> b (n l) h w"), [xc, xt]))
        batch_size = xc.shape[0]
        identity_xc, identity_xt = xc, xt
        if self.pre_conv:
            xc = self.pre_close_conv(xc)
            xt = self.pre_trend_conv(xt)

        close_out = self.closeness_transformer(xc)
        trend_out = self.trend_transformer(xt)
        identity_close_out, identity_trend_out = close_out, trend_out
        close_out, trend_out = self.mlp_head_close(close_out), self.mlp_head_trend(trend_out)

        # relu + linear

        close_out = close_out.reshape(batch_size, self.nb_flow, self.map_height, self.map_width)
        trend_out = trend_out.reshape(batch_size, self.nb_flow, self.map_height, self.map_width)

        close_out = self.close_ilayer(close_out)
        trend_out = self.trend_ilayer(trend_out)
        out = close_out + trend_out

        if self.shortcut:
            shortcut_out = self.Rc_Xc(identity_xc) + self.Rc_Xt(identity_xt)

            out += shortcut_out

        if not self.training:
            out = out.relu()


        # classification task
        # xc, xt: (Batch_size, Token_Dim)
        concat_x = torch.cat([identity_close_out, identity_trend_out], dim=-1)
        out_class = self.mlp(concat_x)
        out = (out, out_class)
        return out


def create_model(arg):
    """

    :param arg: arg class.
    :return:
    """
    device = arg.device
    arg_dict = arg_class2dict(arg)
    model = STTransformer(**arg_dict)
    # num_close,map_height,map_width
    xc_shape = (arg.close_channels, arg.map_height, arg.map_width)
    xt_shape = (arg.trend_channels, arg.map_height, arg.map_width)

    summary(model.to(device), [xc_shape, xt_shape])
    return model.to(device)


if __name__ == '__main__':
    shape = (1, 2, 6, 32, 32)
    # 1,12,32,32 -> 1,64,16*12
    xt = torch.randn(shape)
    xc = torch.randn(shape)
    transformer = STTransformer(close_channels=12, trend_channels=12, conv3d=True)

    pred = transformer(xc, xt)
    print(pred.shape)
    # todo: train,val,evaluate plot training curve,print test result.
