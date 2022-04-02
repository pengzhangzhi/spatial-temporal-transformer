"""=================================================

@Project -> File：ST-3DNet-main->prepareData

@IDE：PyCharm

@coding: utf-8

@time:2021/7/19 23:15

@author:Pengzhangzhi

@Desc：
=================================================="""
import argparse

from help_funcs import read_config
from prepareDataBJ import load_data_BJ
from prepareDataBikeNYC import load_data_TaxiNYC
from prepareDataNY import load_data_NY
from utils import *
from copy import copy


def generate_data(config_name):
    dir = os.getcwd()
    # read config file
    training_config = read_config(config_name=config_name)
    consider_external_info = bool(training_config["consider_external_info"])
    len_closeness = int(training_config["len_closeness"])
    len_period = int(training_config["len_period"])
    len_trend = int(training_config["len_trend"])
    T = int(training_config["T"])  # number of time intervals in one day
    nb_flow = int(
        training_config["nb_flow"]
    )  # there are two types of flows: new-flow and end-flow
    days_test = int(
        training_config["days_test"]
    )  # 7*4 divide data into two subsets: Train & Test, of which the test set is the last 4 weeks
    map_height = int(training_config["map_height"])  # grid size
    map_width = int(training_config["map_width"])  # grid size
    len_test = T * days_test
    dataset = str(training_config["dataset"])
    prediction_offset = int(training_config["prediction_offset"])
    ext = "ext" if consider_external_info else "noext"

    if dataset == "TaxiBJ":

        name = f"TaxiBJ_offset%d_c%d_p%d_t%d_{ext}" % (
            prediction_offset,
            len_closeness,
            len_period,
            len_trend,
        )
        filename = os.path.join(dir, "data", "TaxiBJ", name)

        (
            X_train,
            Y_train,
            X_test,
            Y_test,
            mmn,
            external_dim,
            timestamp_train,
            timestamp_test,
        ) = load_data_BJ(
            T=T,
            nb_flow=nb_flow,
            len_closeness=len_closeness,
            len_period=len_period,
            len_trend=len_trend,
            len_test=len_test,
            meta_data=consider_external_info,
            holiday_data=consider_external_info,
            meteorol_data=consider_external_info,
            prediction_offset=prediction_offset,
        )

    elif dataset == "BikeNYC":

        filename = os.path.join(
            dir,
            "data",
            "BikeNYC",
            f"NYC_offset%d_c%d_p%d_t%d_{ext}"
            % (prediction_offset, len_closeness, len_period, len_trend),
        )
        original_filename = "NYC14_M16x8_T60_NewEnd.h5"
        # generate data with external information
        (
            X_train,
            Y_train,
            X_test,
            Y_test,
            mmn,
            external_dim,
            timestamp_train,
            timestamp_test,
        ) = load_data_NY(
            original_filename,
            T=T,
            nb_flow=nb_flow,
            len_closeness=len_closeness,
            len_period=len_period,
            len_trend=len_trend,
            len_test=len_test,
            meta_data=True,
            prediction_offset=prediction_offset,
        )

    elif dataset == "TaxiNYC":

        filename = os.path.join(
            dir,
            "data",
            "TaxiNYC",
            f"TaxiNYC_offset%d_c%d_p%d_t%d_{ext}"
            % (prediction_offset, len_closeness, len_period, len_trend),
        )

        (
            X_train,
            Y_train,
            X_test,
            Y_test,
            mmn,
            external_dim,
            timestamp_train,
            timestamp_test,
        ) = load_data_TaxiNYC(
            T=T,
            nb_flow=nb_flow,
            len_closeness=len_closeness,
            len_period=len_period,
            len_trend=len_trend,
            len_test=len_test,
            meta_data=consider_external_info,
            holiday_data=consider_external_info,
            meteorol_data=consider_external_info,
            prediction_offset=prediction_offset,
        )

    print("filename:", filename)
    f = open(filename, "wb")
    pickle.dump(X_train, f)
    pickle.dump(Y_train, f)
    pickle.dump(X_test, f)
    pickle.dump(Y_test, f)
    pickle.dump(mmn, f)
    pickle.dump(external_dim, f)
    pickle.dump(timestamp_train, f)
    pickle.dump(timestamp_test, f)
    f.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="generate dataset.")
    argparser.add_argument(
        "-c",
        "--config-name",
        type=str,
        default="TaxiBJ_c4_t1_depth6_pre_train_epoch1200_random_pick",
    )

    opt = argparser.parse_args()
    config_name = opt.config_name

    generate_data(config_name=config_name)

    # generate_data(config_name="BikeNYC")
    # generate_data(config_name="TaxiBJ")
    # generate_data(config_name="TaxiNYC")
