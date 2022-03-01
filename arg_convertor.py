'''=================================================

@Project -> File：新建文件夹->arg_convertor

@IDE：PyCharm

@coding: utf-8

@time:2021/7/24 16:57

@author:Pengzhangzhi

@Desc：
=================================================='''
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
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(temp_dict, file, ensure_ascii=False, indent=2)


def convert_TaxiBJ(c=4, t=1, depth=2, pretrain_epoch=600, sp=True, pretrain_way_="random", config_name="TaxiBJ",
                   pretrain_times_=1, ex=28):
    class arg:
        split = 0.1
        batch_size = 128
        lr = 0.001
        lrf = 0.01
        epochs = 600
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

        pretrain_epochs = pretrain_epoch
        pretrain_times = pretrain_times_
        pretrain_way = pretrain_way_
        experiment_name = config_name

        ext_dim = 28
        drop_prob = 0.1
        conv_channels = 64
        pre_conv = True
        seq_pool = False
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

    path = os.path.join("config",f"{config_name}.json")
    arg_class2json(arg, path)
    print("done!")


def convert_BikeNYC(c=6, t=2,
                    pt=0, pw="random",
                    pc=True, sp=True, sc=True,
                    depth=2,
                    ps=4,
                    config_name="BikeNYC", ):
    class arg:
        split = 0.1
        batch_size = 128
        lr = 0.001
        lrf = 0.01
        epochs = 800
        device = "cuda"
        consider_external_info = True
        len_closeness = c
        len_period = 0
        len_trend = t
        T = 24
        nb_flow = 2
        days_test = 10
        map_height = 20
        map_width = 10
        m_factor = 1
        m_factor_2 = 1
        # m_factor = 1.2570787221094177
        # m_factor_2 = 1.5802469135802468
        dataset = "BikeNYC"
        prediction_offset = 0

        random_pick = False
        pretrain_epochs = 600
        pretrain_times = pt
        pretrain_way = pw

        experiment_name = config_name

        ext_dim = 33
        drop_prob = 0.1
        conv_channels = 64
        pre_conv = pc
        seq_pool = False
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
    path = os.path.join("config",f"{config_name}.json")
    arg_class2json(arg, path)
    print("done!")

def convert_TaxiNYC(c=6, t=2,
                    pt=1, pw="random",
                    pc=True, sp=True, sc=True,
                    depth=2, ps=8, trans_dim=128,
                    config_name="TaxiNYC", ):
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

        random_pick = False
        pretrain_epochs = 600
        pretrain_times = pt
        pretrain_way = pw
        shuffle_mode = ""
        experiment_name = config_name

        ext_dim = 24
        drop_prob = 0.1
        conv_channels = 64
        pre_conv = pc
        seq_pool = False
        shortcut = sc
        patch_size = ps
        close_channels = len_closeness * nb_flow
        trend_channels = len_trend * nb_flow
        close_dim = trans_dim
        trend_dim = trans_dim
        close_depth = depth
        trend_depth = depth
        close_head = 2
        trend_head = 2
        close_mlp_dim = 512
        trend_mlp_dim = 512

    path = os.path.join("config",f"{config_name}.json")
    arg_class2json(arg, path)
    print("done!")


def generate_idx_depth_args():
    for depth in [2, 8]:
        for index in range(1, 8):
            convert_TaxiBJ(index=index, depth=depth, config_name=f"exchange_idx={index}_depth={depth}")
            print(fr"exchange_idx={index}, depth={depth}")
    convert_BikeNYC()
    convert_TaxiNYC()


def generate_depth_args():
    for depth in range(1, 9):
        convert_TaxiBJ(depth=depth, config_name=f"depth={depth}_c10_p4_no_pretrain_pretrain_1")
        print(fr" depth={depth}")


def generate_depth_shuffle_args():
    for depth in range(2, 9):
        for pretrain_time in [1, 4]:
            convert_TaxiBJ(c=10, t=4, pretrain_way_="shuffle",
                           pretrain_times_=pretrain_time, depth=depth,
                           config_name=f"shuffle_each_pretrain_{pretrain_time}_depth={depth}_c10_p4")
            print(fr" depth={depth}")


def generate_shuffle_with_default_data_Scale():
    for pretrain_way in ["shuffle", "same_shuffle"]:
        for shuffle_mode in ["shuffle_his", "shuffle_all"]:
            for pretrain_time in [1, 4]:
                convert_TaxiBJ(c=6, t=2, depth=2,
                               pretrain_way_=pretrain_way,
                               shuffle_mode_=shuffle_mode,
                               pretrain_times_=pretrain_time,
                               config_name=f"c6t2_depth_2{pretrain_way}_{shuffle_mode}_pretrain_times_{pretrain_time}")


def generate_c4_t1_random_args():
    for depth in range(2, 9):
        convert_TaxiBJ(c=4, t=1, pretrain_way_="random",
                       depth=depth, pretrain_times_=1,
                       config_name=f"random_c4_t1_pretrain_1_depth={depth}")
        print(f"random_c4_t1_pretrain_1_depth={depth}")


def generate_random_pick(c=4, t=1, depth=6, pretrain_way_="random", pretrain_times_=1):
    convert_TaxiBJ(c=c, t=t, depth=depth,
                   pretrain_way_=pretrain_way_,
                   pretrain_times_=pretrain_times_,
                   config_name=f"c{c}t{t}_depth_{depth}_pretrain_way_{pretrain_way_}_pretrain_times_{pretrain_times_}",
                   )


def generate_TaxiBJ(c=4, t=1, pretrain_times_=1, ):
    for i in range(1, 11):
        convert_TaxiBJ(c=c, t=t, sp=False, pretrain_times_=pretrain_times_, config_name=f"TaxiBJ{i}")


def generate_TaxiNYC():
    for i in range(1, 11):
        convert_TaxiNYC(c=10, t=4, pt=1, sp=False, config_name=f"TaxiNYC{i}")


if __name__ == '__main__':
    convert_BikeNYC(c=6, t=2,
                    pt=1, pw="random",
                    pc=True, sp=True, sc=True,
                    depth=2,
                    ps=10,
                    config_name="BikeNYC", )