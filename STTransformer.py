"""=================================================

@Project -> File：ST-Transformer->STTransformer

@IDE：PyCharm

@coding: utf-8

@time:2021/7/23 17:01

@author:Pengzhangzhi

@Desc：
=================================================="""
import os
import re
import sys
from time import time

import torch
import torch.nn as nn
from einops import rearrange, reduce
from torch.nn import init

from arg_convertor import arg_class2dict
from base_layers import BasicBlock, PostConvBlock
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
    """elementwise multiplication"""

    def __init__(self, input_shape):
        """
        input_shape: (,*,c,,h,w)
        self.weights shape: (,*,c,h,w)
        """
        super(iLayer, self).__init__()
        self.weights = nn.Parameter(
            torch.randn(*input_shape)
        )  # define the trainable parameter
        init.xavier_uniform_(self.weights.data)

    def forward(self, x):
        """
        x: (batch, c, h,w)
        self.weights shape: (c,h,w)
        output: (batch, c, h,w)
        """
        return x * self.weights


class TemporalMLP(nn.Module):
    """transform a spatial-temporal embedding to temporal embedding.
    TODO: add layer norm.
    """

    def __init__(self, in_dim, hidden_dim, num_time_class):
        super(TemporalMLP, self).__init__()
        self.temporal_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_time_class),
        )

    def forward(self, embedding):
        return self.temporal_mlp(embedding)


def pair(t):
    return t if isinstance(t, list) else [t, t]


class InversePatchify(nn.Module):
    """transform bunch of tokens into 2D map by inverse patchify."""

    def __init__(self, map_size, patch_size, token_dim, num_channels):
        """the map size and the patch size should match.
        Args:
            map_size (int/list):
            patch_size (int/list):
            token_dim (int):
            num_channels (int): the num of channels of output 2d feature map.
        """
        super(InversePatchify, self).__init__()
        self.map_height, self.map_width = pair(map_size)
        self.patch_height, self.patch_width = pair(patch_size)
        self.num_heights = self.map_height // self.patch_height
        self.num_widths = self.map_width // self.patch_width
        assert (
            self.map_height % self.patch_height == 0
            and self.map_width % self.patch_width == 0
        ), "Map dimensions must be divisible by the patch size."
        self.linear = nn.Linear(
            token_dim, self.patch_height * self.patch_width * num_channels
        )

    def forward(self, st_embedding):
        """inverse the st_embedding back to the 2D map.

        Args:
            st_embedding (torch.Tensor): (batch size, num_tokens, token_dim)

        Returns:
            torch.Tensor: (batch_size, num_channels, map_height, map_width)
        """
        st_embedding = self.linear(
            st_embedding
        )  # (batch_size, num_tokens, patch_height* patch_width* num_channels)
        st_embedding = rearrange(
            st_embedding,
            "b n (ph pw c) -> b n ph pw c",
            ph=self.patch_height,
            pw=self.patch_width,
        )
        map_features = rearrange(
            st_embedding,
            "b (nh nw) ph pw c -> b c (nh ph) (nw pw)",
            nh=self.num_heights,
            nw=self.num_widths,
        )
        return map_features


class TemporalAttention(nn.Module):
    """temporal attention module that leverage time information of predicted interval to generate a 2D attention map."""

    def __init__(self, embedding_dim, map_size):
        """
        Args:
            embedding_dim (int): the dim of time embedding (num of time class).
            map_size (int/list): the map size of 2D flow map.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.map_height, self.map_width = pair(map_size)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, self.map_height * self.map_width),
        )

    def forward(self, time_embedding):
        """transform the time_embedding to temporal attention that are applied to 2D feature map.

        Args:
            time_embedding (torch.Tensor): the time embedding used for time prediction.(batch_size, embedding_dim)

        Returns:
            torch.Tensor: 2D attention map with temporal information. (batch_size, map_height,map_width)
        """
        time_attention = self.attention(time_embedding)
        time_attention = rearrange(
            time_attention, "b (h w) -> b 1 h w", h=self.map_height, w=self.map_width
        )
        return time_attention


class STTransformer(nn.Module):
    def __init__(
        self,
        map_height=32,
        map_width=32,
        patch_size=4,
        len_closeness=6,
        len_trend=6,
        close_dim=1024,
        trend_dim=1024,
        close_depth=4,
        trend_depth=4,
        close_head=2,
        trend_head=2,
        close_mlp_dim=2048,
        trend_mlp_dim=2048,
        nb_flow=2,
        pre_conv=True,
        shortcut=True,
        conv_channels=64,
        drop_prob=0.1,
        time_class=48,  # num_of_day_of_week(7) + num_of_hour_of_day(24/48)
        temporal_hidden_dim=2048,
        post_num_channels=10,
        ext_dim=24,
        **kwargs
    ):

        super(STTransformer, self).__init__()
        self.map_height = map_height
        self.map_width = map_width
        self.nb_flow = nb_flow
        output_dim = nb_flow * map_height * map_width
        close_dim_head = int(close_dim / close_head)
        trend_dim_head = int(trend_dim / close_head)

        self.pre_conv = pre_conv
        close_channels = len_closeness * nb_flow
        trend_channels = len_trend * nb_flow
        if pre_conv:
            self.pre_close_conv = nn.Sequential(
                BasicBlock(inplanes=close_channels, planes=conv_channels),
            )
            self.pre_trend_conv = nn.Sequential(
                BasicBlock(inplanes=trend_channels, planes=conv_channels),
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
        )
        combined_token_dim = close_dim + trend_dim
        self.inverse_patchify = InversePatchify(
            [map_height, map_width], patch_size, combined_token_dim, post_num_channels
        )
        self.temporal_mlp = TemporalMLP(
            combined_token_dim, temporal_hidden_dim, time_class
        )
        self.temporal_attention = TemporalAttention(time_class, [map_height, map_width])
        self.post_conv_block = PostConvBlock(
            inplanes=post_num_channels+1, planes=self.nb_flow
        ) # the inplanes is the number of channels of reshaped st_features + an one-channel external features
        self.ext_dim = ext_dim
        
        if ext_dim:
            self.ext_module = TemporalMLP(
                ext_dim, 10, map_height*map_width
            )
        
        input_shape = (nb_flow, map_height, map_width)
        self.shortcut = shortcut
        if shortcut:
            self.Rc_Xc = Rc(input_shape)
            self.Rc_Xt = Rc(input_shape)
            # self.Rc_conv_Xc = Rc(input_shape)
            # self.Rc_conv_Xt = Rc(input_shape)

    def forward(self, xc, xt, x_ext=None):
        """extract spatial-temporal patterns from historical data and
        predict the traffic flow at future interval and its time label (which day of a week).

        Args:
            xc (_type_): batch_size, nb_flow, length_c, height, width
            xt (_type_): batch_size, nb_flow, length_t, height, width
            x_ext (_type_, optional): _description_. Defaults to None.

        Returns:
            st_features: predictions of traffic flow at future interval.
            day_of_week: predicted which day of a week.
            hour_of_day: predicted which hour (half-hour) of a day.
        """
        if len(xc.shape) == 5:
            # reshape 5 dimensions to 4 dimensions.
            xc, xt = list(
                map(lambda x: rearrange(x, "b n l h w -> b (n l) h w"), [xc, xt])
            )
        identity_xc, identity_xt = torch.clone(xc), torch.clone(xt)
        # pre-conv Block
        if self.pre_conv:
            xc = self.pre_close_conv(xc)
            xt = self.pre_trend_conv(xt)
        # transformer block
        close_out, mean_close_out = self.closeness_transformer(xc)
        trend_out, mean_trend_out = self.trend_transformer(xt)

        # temporal prediction task
        # xc, xt: (Batch_size, token_dim)
        temporal_embedding = torch.cat([mean_close_out, mean_trend_out], dim=-1)
        time_class = self.temporal_mlp(temporal_embedding)

        st_embedding = torch.cat(
            [close_out, trend_out], dim=-1
        )  # (b, num_tokens, 2*token_dim)
        st_map = self.inverse_patchify(
            st_embedding
        )  # (b, n_channels, map_height,map_width)
        time_attention = self.temporal_attention(
            time_class
        )  # (b, 1, map_height,map_width)

        st_map = st_map * time_attention  # (b, n_channels, map_height,map_width)
        if x_ext is not None and self.ext_dim != 0:
            ext_features = self.ext_module(x_ext)
            ext_features = rearrange(ext_features,"b (h w) -> b 1 h w",h=self.map_height,w=self.map_width)
            st_map = torch.cat([st_map,ext_features],dim=1)
        st_map = self.post_conv_block(st_map)  # output: (b, n_flow, map_height,map_width)

        if self.shortcut:
            shortcut_out = self.Rc_Xc(identity_xc) + self.Rc_Xt(identity_xt)
            st_prediction = st_map + shortcut_out

        if not self.training:
            st_prediction = st_prediction.relu()

        day_of_week, hour_of_day = time_class[:, :7], time_class[:, 7:]
        return st_prediction, day_of_week, hour_of_day


def create_model(arg):
    """

    :param arg: arg class.
    :return:
    """
    device = arg.device
    arg_dict = arg_class2dict(arg)
    model = STTransformer(**arg_dict)
    # num_close,map_height,map_width
    xc_shape = (arg.nb_flow * arg.len_closeness, arg.map_height, arg.map_width)
    xt_shape = (arg.nb_flow * arg.len_closeness, arg.map_height, arg.map_width)

    # summary(model.to(device), [xc_shape, xt_shape])
    return model.to(device)


if __name__ == "__main__":
    batch_size, len_c, len_t, height, width = 88, 2, 2, 32, 32
    patch_size = [4, 8]

    sttransformer = STTransformer(
        map_height=height,
        map_width=width,
        patch_size=patch_size,
        len_closeness=len_c,
        len_trend=len_t,
        close_dim=1024,
        trend_dim=1024,
        close_depth=4,
        trend_depth=4,
        close_head=2,
        trend_head=2,
        close_mlp_dim=2048,
        trend_mlp_dim=2048,
        nb_flow=2,
        pre_conv=True,
        shortcut=True,
        conv_channels=64,
        drop_prob=0.1,
        time_class=7 + 48,  # num_of_day_of_week(7) + num_of_hour_of_day(24/48)
        temporal_hidden_dim=2048,
        post_num_channels=10,
        ext_dim = 24
    )
    xc = torch.randn((batch_size, 2, len_c, height, width))
    xt = torch.randn((batch_size, 2, len_t, height, width))
    x_ext = torch.randn((batch_size,24))
    flow_prediction, day_of_week_label, time_of_day_label = sttransformer(xc, xt,x_ext)
    print(flow_prediction.shape)
    print(day_of_week_label.shape)
    print(time_of_day_label.shape)
