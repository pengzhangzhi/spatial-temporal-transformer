'''=================================================

@Project -> File：ST-3DNet-main->prepareDataBikeNYC

@IDE：PyCharm

@coding: utf-8

@time:2021/7/19 23:19

@author:Pengzhangzhi

@Desc：
=================================================='''
from curses import meta
from json.tool import main
from prepareDataNY import load_data_NY
from utils import *
from copy import copy
import pandas as pd
import argparse
from help_funcs import read_config
def timestamp2str(timestamps):
    res = []
    for time in timestamps:
        ts = pd.to_datetime(str(time)) 
        d = bytes(ts.strftime('%Y%m%d%H'),encoding='utf8')
        res.append(d)
    return np.array(res)

def load_data(path):
    data_path = os.path.join(path, "data.npy")
    timestamps_path = os.path.join(path, "timesolts.npy")
    data,timestamps = np.load(data_path),np.load(timestamps_path)
    return data, timestamps

def load_BikeNYC_ext_data(path):
    ext_data_path = os.path.join(path, "ext.npy")
    return np.load(ext_data_path)

def load_holiday(timeslots, fname):
    f = open(os.path.join(fname, "Holiday.txt"), 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    print(H.sum())
    # print(timeslots[H==1])
    return H[:, None]


def load_meteorol(timeslots, fname):
    f = h5py.File(os.path.join(fname, 'Meteorology.h5'), 'r')
    Timeslot = f['date'][:]
    WindSpeed = f['WindSpeed'][:]
    Weather = f['Weather'][:]
    Temperature = f['Temperature'][:]
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    print("shape: ", WS.shape, WR.shape, TE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data

def load_data_BikeNYC(T=24, nb_flow=2,dataset="BikeNYC",
                      len_closeness=None, len_period=None, len_trend=None,
                      len_test=None, meta_data=True, holiday_data=True, meteorol_data=True,prediction_offset=0):
    assert (len_closeness + len_period + len_trend > 0)
    dir = os.getcwd()
    # load data
    # 13 - 16
    data_all = []
    timestamps_all = list()

    data_dir = os.path.join(dir, 'data',dataset)
    print("file name: ", data_dir)
    data, timestamps = load_data(data_dir)
    
    
    # remove a certain day which does not have 48 timestamps
    timestamps = timestamp2str(timestamps)
    print("timestamps: ", timestamps)
    print('train_data shape: ', data.shape)
    # data, timestamps = remove_incomplete_days(data, timestamps, T)
    # data = data[:, :nb_flow]
    data[data < 0] = 0.
    if data.shape[-1] == 2: # the last dimension is NumOfFlow 
        data = rearrange(data,"L H W N -> L N H W")
    data_all.append(data)
    timestamps_all.append(timestamps)
    print("\n")
    print('train_data shape: ', data.shape)
    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]
    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        print("data: ", data.shape)
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        # _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
        #     len_closeness=len_closeness, len_period=len_period, len_trend=len_trend)
        # print("create dataset gsn")
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset_3D(len_closeness=len_closeness, len_period=len_period,
                                                                 len_trend=len_trend,prediction_offset=prediction_offset)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    
        
    
    
    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[
                                            :-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[
                                        -len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[
                                      :-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    meta_feature = []
    if meta_data:
        time_feature = timestamp2array(timestamps_Y, T)
        print("meta_feature:",time_feature.shape)
        
        meta_feature = time_feature if len(
            time_feature) > 0 else np.asarray(meta_feature)
        metadata_dim = meta_feature.shape[1] if len(
            meta_feature.shape) > 1 else None
        metadata_dim = meta_feature.shape[1]
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
        print('time feature:', meta_feature.shape)
        print("day or night, weekday or weekend, day of week, time of day.")
    else:
        metadata_dim = None
        

    print('X train shape:')
    for _X in X_train:
        print(_X.shape, )
    print()

    print('X test shape:')
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test


def generate_BikeNYC_dataset(config):
    dir = os.getcwd()
    prediction_offset = config["prediction_offset"]
    len_closeness = config["len_closeness"]
    len_period = config["len_period"]
    len_trend = config["len_trend"]
    dataset = config["dataset"]
    T = int(config['T'])  # number of time intervals in one day
    consider_external_info = bool(config['consider_external_info'])
    days_test = int(config[ 'days_test'])
    ext = "ext" if consider_external_info else "noext"
    len_test = T * days_test
    X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test \
        = load_data_BikeNYC(T=T, nb_flow=2,dataset=dataset,
                      len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,
                      len_test=28*24, meta_data=consider_external_info, holiday_data=consider_external_info, 
                      meteorol_data=consider_external_info,prediction_offset=prediction_offset)
    
    filename = os.path.join(dir, 'data', f'{dataset}',
                            f'{dataset}_offset%d_c%d_p%d_t%d_{ext}' % (
                                prediction_offset, len_closeness, len_period, len_trend))
    print('filename:', filename)
    f = open(filename, 'wb')
    pickle.dump(X_train, f)
    pickle.dump(Y_train, f)
    pickle.dump(X_test, f)
    pickle.dump(Y_test, f)
    pickle.dump(mmn, f)
    pickle.dump(metadata_dim, f)
    pickle.dump(timestamp_train, f)
    pickle.dump(timestamp_test, f)
    f.close()
    
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser("generate BikeNYC Dataset")
    argparser.add_argument("-c","--config-name",type=str,default="BikeDC")
    opt = argparser.parse_args()
    config_name = opt.config_name
    training_config = read_config(config_name=config_name)
    generate_BikeNYC_dataset(training_config)

