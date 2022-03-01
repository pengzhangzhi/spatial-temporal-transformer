'''=================================================

@Project -> File：ST-DATA->read_data

@IDE：PyCharm

@coding: utf-8

@time:2021/7/23 10:30

@author:Pengzhangzhi

@Desc：
=================================================='''
import torch
import os
from help_funcs import read_config, load_data


def load(args,pretrain=False):
    dir = os.path.dirname(os.path.realpath(__file__))
    # read config file

    # initialize hyperparams variables
    consider_external_info = bool(args.consider_external_info)
    len_closeness = int(args.len_closeness)
    len_period = int(args.len_period)
    len_trend = int(args.len_trend)
    dataset = str(args.dataset)
    prediction_offset = int(args.prediction_offset)
    ext = "ext" if consider_external_info else "noext"


    # generate model path(name) based on the dataset
    if dataset == "BikeNYC":
        filename = f'BikeNYC_offset%d_c%d_p%d_t%d_{ext}' % (prediction_offset, len_closeness, len_period, len_trend)

    elif dataset == "TaxiBJ":
        filename = f'TaxiBJ_offset%d_c%d_p%d_t%d_{ext}' % (
            prediction_offset, len_closeness, len_period, len_trend)


    elif dataset == "TaxiNYC":
        # TODO: add  TaxiNYC dataset (preprocessing, and training) DONE!
        filename = f'TaxiNYC_offset%d_c%d_p%d_t%d_{ext}' % \
                   (prediction_offset, len_closeness, len_period, len_trend)

    else:
        raise ValueError(f"Invalid dataset {dataset}. Only support BikeNYC, TaxiBJ ,and TaxiNYC.")

    filename = os.path.join(dir, "data", f'{dataset}', filename)
    print('dataset filename:', filename)

    return load_data(filename)


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, mmn, external_dim, \
    timestamp_train, timestamp_test = load("TaxiBJ")
