"""=================================================

@Project -> File：新建文件夹->arg_convertor

@IDE：PyCharm

@coding: utf-8

@time:2021/7/24 16:57

@author:Pengzhangzhi

@Desc：
=================================================="""
import json
import os
import torch


def json2dict(json_path, verbose=True):
    """convert json file into class args."""
    with open(json_path) as f:
        arg_dict = json.load(f)
    if verbose:
        print("*********preprocessing hyper-parameters ***********")

        for key, value in arg_dict.items():
            print(f"{key}:{value}\t", end=" ")
        print()
        print("*" * 30)
    return arg_dict


def json2args(json_path, verbose=True):
    """
    json file to arg class.
    :param json_path:
    :param verbose:
    :return:
    """
    arg_dict = json2dict(json_path, verbose=verbose)
    return type("temp", (), arg_dict)


def arg_class2dict(arg_class):
    """convert arg class to dict."""
    arg_dict = arg_class.__dict__.copy()
    temp_dict = {}
    for key, values in arg_dict.items():
        if not key.startswith("_"):
            temp_dict[key] = values
    return temp_dict


def arg_class2json(arg_class, path):
    """generate args json file."""
    temp_dict = arg_class2dict(arg_class)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(temp_dict, file, ensure_ascii=False, indent=2)


def convert_TaxiBJ(
    c=4,
    t=1,
    depth=2,
    pretrain_epoch=1200,
    pretrain_way_="random",
    config_name="TaxiBJ",
    pretrain_times_=1,
):
    class arg:
        split = 0.1
        batch_size = 128
        lr = 0.001
        lrf = 0.01
        epochs = 1000
        device = "cuda" if torch.cuda.is_available() else "cpu"
        consider_external_info = True
        len_closeness = c
        len_period = 0
        len_trend = t
        T = 48
        nb_flow = 2
        days_test = 28
        map_height = 32
        map_width = 32
        m_factor = 1.0
        m_factor_2 = 1.0
        dataset = "TaxiBJ"
        prediction_offset = 0
        random_pick = True

        loss_weight = 1.0
        pretrain_epochs = pretrain_epoch
        pretrain_times = pretrain_times_
        pretrain_way = pretrain_way_
        experiment_name = config_name

        drop_prob = 0.1
        conv_channels = 64
        pre_conv = True
        seq_pool = True
        shortcut = True
        patch_size = 8
        close_channels = len_closeness * nb_flow
        trend_channels = len_trend * nb_flow
        close_dim = 128
        trend_dim = 128
        close_depth = depth
        trend_depth = depth
        close_head = 2
        trend_head = 2
        close_mlp_dim = 512
        trend_mlp_dim = 512

    arg_class2json(arg, os.path.join("config", f"{config_name}.json"))


def convert_BikeNYC(
    c=6,
    t=2,
    pt=1,
    pw="random",
    pc=True,
    sp=True,
    sc=True,
    depth=2,
    ps=4,
    config_name="BikeNYC",
):
    class arg:
        split = 0.1
        batch_size = 128
        lr = 0.001
        lrf = 0.01
        epochs = 1000
        device = "cuda"
        consider_external_info = True
        len_closeness = c
        len_period = 0
        len_trend = t
        T = 24
        nb_flow = 2
        days_test = 10
        map_height = 16
        map_width = 8
        m_factor = 1.2570787221094177
        m_factor_2 = 1.5802469135802468
        dataset = "BikeNYC"
        prediction_offset = 0

        loss_weight = 1.0
        random_pick = False
        pretrain_epochs = 600
        pretrain_times = pt
        pretrain_way = pw

        experiment_name = config_name

        drop_prob = 0.1
        conv_channels = 64
        pre_conv = pc
        seq_pool = sp
        shortcut = sc
        patch_size = ps
        close_channels = len_closeness * nb_flow
        trend_channels = len_trend * nb_flow
        close_dim = 128
        trend_dim = 128
        close_depth = depth
        trend_depth = depth
        close_head = 2
        trend_head = 2
        close_mlp_dim = 512
        trend_mlp_dim = 512

    arg_class2json(arg, os.path.join("config", f"{config_name}.json"))


def convert_TaxiNYC(
    c=6,
    t=2,
    pt=1,
    pw="random",
    pc=True,
    sp=True,
    sc=True,
    depth=2,
    ps=8,
    trans_dim=128,
    config_name="TaxiNYC",
):
    class arg:
        split = 0.1
        batch_size = 128
        lr = 0.005
        lrf = 0.01
        epochs = 800
        device = "cuda"
        consider_external_info = 1
        len_closeness = c
        len_period = 0
        len_trend = t
        T = 24
        nb_flow = 2
        days_test = 28
        map_height = 16
        map_width = 8
        m_factor = 1
        m_factor_2 = 1
        dataset = "TaxiNYC"
        prediction_offset = 0

        pretrain_epochs = 600
        pretrain_times = pt
        experiment_name = config_name

        drop_prob = 0.1
        conv_channels = 64
        pre_conv = pc
        shortcut = sc
        patch_size = ps
        close_dim = trans_dim
        trend_dim = trans_dim
        close_depth = depth
        trend_depth = depth
        close_head = 2
        trend_head = 2
        close_mlp_dim = 4 * close_dim
        trend_mlp_dim = 4 * trend_dim
        temporal_hidden_dim = close_dim + trend_dim
        post_num_channels = 10
        time_class = 7 + T
        alpha = 1.0
        beta = 1.0
        ext_dim = 24-8

    arg_class2json(arg, os.path.join("config", f"{config_name}.json"))


if __name__ == "__main__":
    convert_TaxiNYC()
