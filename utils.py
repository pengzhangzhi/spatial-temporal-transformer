import copy
import random
import numpy as np
import pandas as pd
from datetime import datetime
import os
import six.moves.cPickle as pickle
import numpy as np
import h5py
import time
import pickle
import sys
import time
from datetime import datetime
from time import mktime
import torch
from einops import rearrange
from tqdm import tqdm

from help_funcs import make_experiment_dir, print_run_time


def write_pickle(list_info: list, file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, "rb") as f:
        info_list = pickle.load(f)
        return info_list


def pretrain_shuffle(xc, xt, x_ext, y):
    """

    :param xc: (batch size, nb_flow, c, h, w)
    :param xt: (batch size, nb_flow, c, h, w)
    :param x_ext:
    :param y: (batch size, nb_flow, h, w)

    """
    xc, xt = list(map(lambda x: rearrange(x, "b n l h w -> l b n h w"), [xc, xt]))
    y = rearrange(y, "b n h w -> 1 b n h w")
    data = torch.cat([xc, xt, y], dim=0)  # l' b n h w
    his_len = len(data) - 1
    idx = torch.randint(0, his_len - 1, (1,))
    temp_y = data[-1].clone()  # normalize data[idx]
    data[-1] = data[idx]
    data[idx] = temp_y
    chunk_len = [len(xc), len(xt), 1]
    xc, xt, y = list(
        map(
            lambda x: rearrange(x, "l b n h w ->  b n l h w"),
            list(torch.split(data, chunk_len)),
        )
    )
    y = rearrange(y, "b n l h w -> b n h w")
    # renormalize y
    return xc, xt, x_ext, y


def fix_seed(seed=666):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # reference: https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    """train the given model with optimizer on data_loader at eepoch.

    Args:
        model (_type_):
        optimizer (_type_):
        data_loader (_type_):
        device (_type_):
        epoch (_type_):
        alpha (float, optional): weight of auxilary loss. Defaults to 1.0.

    Returns:
        flow prediction MSE loss
        flow prediction RMSE
        day of week classification CEloss
        day of week classification accuracy
        time of day classification CEloss
        time of day classification accuracy
    """
    torch.autograd.set_detect_anomaly(True)
    model.train()
    (
        mse_loss_criterion,
        cross_entropy_day_of_week,
        cross_entropy_time_of_day,
        accu_loss,
        accu_rmse,
        accu_correct_day_of_week,
        accu_class_loss_day_of_week,
        accu_correct_time_of_day,
        accu_class_loss_time_of_day,
        sample_num,
    ) = init_metrics()

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        optimizer.zero_grad()
        xc, xt, x_ext, y, day_of_week, time_of_day = load_data(device, data)

        pred, pred_day_of_week, pred_time_of_day = model(xc, xt, x_ext)
        (
            mse_loss,
            cross_entropy_loss_day_of_week,
            cross_entropy_loss_time_of_day,
        ) = cal_loss(
            mse_loss_criterion,
            cross_entropy_day_of_week,
            cross_entropy_time_of_day,
            y,
            day_of_week,
            time_of_day,
            pred,
            pred_day_of_week,
            pred_time_of_day,
        )

        loss = (
            mse_loss
            + args.alpha * cross_entropy_loss_day_of_week
            + args.beta * cross_entropy_loss_time_of_day
        )
        loss.backward()
        optimizer.step()

        (
            avg_mse,
            avg_rmse,
            num_correct_day_of_week,
            num_correct_time_of_day,
            accu_loss,
            accu_rmse,
            accu_correct_day_of_week,
            accu_correct_time_of_day,
            sample_num,
        ) = accumulate_res(
            accu_loss,
            accu_rmse,
            accu_correct_day_of_week,
            accu_class_loss_day_of_week,
            accu_correct_time_of_day,
            accu_class_loss_time_of_day,
            sample_num,
            y,
            day_of_week,
            time_of_day,
            pred_day_of_week,
            pred_time_of_day,
            mse_loss,
            cross_entropy_loss_day_of_week,
            cross_entropy_loss_time_of_day,
        )

        data_loader.desc = (
            "[train epoch {}] MSELoss: {:.3f}, RMSE: {:.3f}, "
            "Accuracy_Day_of_Week: {:.3f} "
            "Accuracy_Time_Of_Day: {:.3f} ".format(
                epoch,
                avg_mse,
                avg_rmse,
                num_correct_day_of_week / len(day_of_week),
                num_correct_time_of_day / len(time_of_day),
            )
        )
        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)

    accu_correct_day_of_week, accu_correct_time_of_day, MSE, RMSE = cal_batch_metrics(
        accu_loss,
        accu_rmse,
        accu_correct_day_of_week,
        accu_correct_time_of_day,
        sample_num,
    )
    return MSE, RMSE, accu_correct_day_of_week, accu_correct_time_of_day


def cal_batch_metrics(
    accu_loss, accu_rmse, accu_correct_day_of_week, accu_correct_time_of_day, sample_num
):
    """calculate the metrics on whole iteration(all data).

    Args:
        accu_loss (_type_): accumulated mse loss
        accu_rmse (_type_): accumulated rmse loss
        accu_correct_day_of_week (_type_): accumulated number of correct day of week
        accu_correct_time_of_day (_type_): accumulated number of correct time of day
        sample_num (_type_): number of samples.

    Returns:
        tuple of float: (accu_correct_day_of_week, accu_correct_time_of_day, MSE, RMSE)
    """
    MSE = accu_loss / sample_num
    RMSE = accu_rmse / sample_num
    accu_correct_day_of_week = (accu_correct_day_of_week / sample_num).item()
    accu_correct_time_of_day = (accu_correct_time_of_day / sample_num).item()
    return accu_correct_day_of_week, accu_correct_time_of_day, MSE, RMSE


def accumulate_res(
    accu_loss,
    accu_rmse,
    accu_correct_day_of_week,
    accu_class_loss_day_of_week,
    accu_correct_time_of_day,
    accu_class_loss_time_of_day,
    sample_num,
    y,
    day_of_week,
    time_of_day,
    pred_day_of_week,
    pred_time_of_day,
    mse_loss,
    cross_entropy_loss_day_of_week,
    cross_entropy_loss_time_of_day,
):
    """calculate the metrics of a batch of data and accumalte metrics.

    Args:
        accu_loss (_type_): _description_
        accu_rmse (_type_): _description_
        accu_correct_day_of_week (_type_): _description_
        accu_class_loss_day_of_week (_type_): _description_
        accu_correct_time_of_day (_type_): _description_
        accu_class_loss_time_of_day (_type_): _description_
        sample_num (_type_): _description_
        y (_type_): _description_
        day_of_week (_type_): _description_
        time_of_day (_type_): _description_
        pred_day_of_week (_type_): _description_
        pred_time_of_day (_type_): _description_
        mse_loss (_type_): _description_
        cross_entropy_loss_day_of_week (_type_): _description_
        cross_entropy_loss_time_of_day (_type_): _description_

    Returns:
        _type_: _description_
    """
    avg_mse = mse_loss.item()
    avg_rmse = avg_mse**0.5
    batch_mse = avg_mse * len(y)
    batch_rmse = avg_rmse * len(y)
    accu_loss += batch_mse
    accu_rmse += batch_rmse
    accu_class_loss_day_of_week += cross_entropy_loss_day_of_week.item() * len(y)
    num_correct_day_of_week = (pred_day_of_week.argmax(-1) == day_of_week).sum()
    accu_correct_day_of_week += num_correct_day_of_week

    accu_class_loss_time_of_day += cross_entropy_loss_time_of_day.item() * len(y)
    num_correct_time_of_day = (pred_time_of_day.argmax(-1) == time_of_day).sum()
    accu_correct_time_of_day += num_correct_time_of_day
    sample_num += len(y)
    return (
        avg_mse,
        avg_rmse,
        num_correct_day_of_week,
        num_correct_time_of_day,
        accu_loss,
        accu_rmse,
        accu_correct_day_of_week,
        accu_correct_time_of_day,
        sample_num,
    )


def cal_loss(
    mse_loss_criterion,
    cross_entropy_day_of_week,
    cross_entropy_time_of_day,
    y,
    day_of_week,
    time_of_day,
    pred,
    pred_day_of_week,
    pred_time_of_day,
):
    """calculate loss of given three criterion on the given true and predicted data.

    Args:
        mse_loss_criterion (_type_): _description_
        cross_entropy_day_of_week (_type_): _description_
        cross_entropy_time_of_day (_type_): _description_
        y (_type_): _description_
        day_of_week (_type_): _description_
        time_of_day (_type_): _description_
        pred (_type_): _description_
        pred_day_of_week (_type_): _description_
        pred_time_of_day (_type_): _description_

    Returns:
        tuple(torch.Tensor): loss of three criterion
    """
    mse_loss = mse_loss_criterion(pred, y)
    cross_entropy_loss_day_of_week = cross_entropy_day_of_week(
        pred_day_of_week, day_of_week
    )
    cross_entropy_loss_time_of_day = cross_entropy_time_of_day(
        pred_time_of_day, time_of_day
    )
    return mse_loss, cross_entropy_loss_day_of_week, cross_entropy_loss_time_of_day


def init_metrics():
    """return all local variables in this function.

    Returns:
        _type_: _description_
    """
    mse_loss_criterion = torch.nn.MSELoss()
    cross_entropy_day_of_week = torch.nn.CrossEntropyLoss()
    cross_entropy_time_of_day = torch.nn.CrossEntropyLoss()
    accu_loss = 0.0  # 累计损失
    accu_rmse = 0.0  # 累计rmse
    accu_correct_day_of_week = 0
    accu_class_loss_day_of_week = 0.0
    accu_correct_time_of_day = 0
    accu_class_loss_time_of_day = 0.0
    sample_num = 0
    return (
        mse_loss_criterion,
        cross_entropy_day_of_week,
        cross_entropy_time_of_day,
        accu_loss,
        accu_rmse,
        accu_correct_day_of_week,
        accu_class_loss_day_of_week,
        accu_correct_time_of_day,
        accu_class_loss_time_of_day,
        sample_num,
    )


def load_data(device, data):
    for i in range(0, len(data)):
        data[i] = data[i].to(device)
    if len(data[2].T) > 16:
        data[2] = data[2][:,8:] # filter out the first 8 dim, which represents meta data. the leftover is meterogical data.
    return data


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):

    model.eval()
    (
        mse_loss_criterion,
        cross_entropy_day_of_week,
        cross_entropy_time_of_day,
        accu_loss,
        accu_rmse,
        accu_correct_day_of_week,
        accu_class_loss_day_of_week,
        accu_correct_time_of_day,
        accu_class_loss_time_of_day,
        sample_num,
    ) = init_metrics()

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        xc, xt, x_ext, y, day_of_week, time_of_day = load_data(device, data)
        pred, pred_day_of_week, pred_time_of_day = model(xc, xt, x_ext)
        (
            mse_loss,
            cross_entropy_loss_day_of_week,
            cross_entropy_loss_time_of_day,
        ) = cal_loss(
            mse_loss_criterion,
            cross_entropy_day_of_week,
            cross_entropy_time_of_day,
            y,
            day_of_week,
            time_of_day,
            pred,
            pred_day_of_week,
            pred_time_of_day,
        )

        (
            avg_mse,
            avg_rmse,
            num_correct_day_of_week,
            num_correct_time_of_day,
            accu_loss,
            accu_rmse,
            accu_correct_day_of_week,
            accu_correct_time_of_day,
            sample_num,
        ) = accumulate_res(
            accu_loss,
            accu_rmse,
            accu_correct_day_of_week,
            accu_class_loss_day_of_week,
            accu_correct_time_of_day,
            accu_class_loss_time_of_day,
            sample_num,
            y,
            day_of_week,
            time_of_day,
            pred_day_of_week,
            pred_time_of_day,
            mse_loss,
            cross_entropy_loss_day_of_week,
            cross_entropy_loss_time_of_day,
        )

        data_loader.desc = (
            "[Val epoch {}] MSELoss: {:.3f}, RMSE: {:.3f}, "
            "Class_Loss_Day_of_Week: {:.3f}, Accuracy_Day_of_Week: {:.3f} "
            "Class_Loss_Time_Of_Day: {:.3f}, Accuracy_Time_Of_Day: {:.3f} ".format(
                epoch,
                avg_mse,
                avg_rmse,
                cross_entropy_loss_day_of_week.item(),
                num_correct_day_of_week / len(day_of_week),
                cross_entropy_loss_time_of_day.item(),
                num_correct_time_of_day / len(time_of_day),
            )
        )
    accu_correct_day_of_week, accu_correct_time_of_day, MSE, RMSE = cal_batch_metrics(
        accu_loss,
        accu_rmse,
        accu_correct_day_of_week,
        accu_correct_time_of_day,
        sample_num,
    )
    return MSE, RMSE, accu_correct_day_of_week, accu_correct_time_of_day


@torch.no_grad()
def test(model, data_loader, device, args):
    """test the model performance on test set.
    Specifically, the model are tested on the overall dataset and partitioned groups.
    the performance are evaluated on ten metrics (5 masked + 5 unmasked).
    Args:
        model (_type_): _description_
        data_loader (_type_): _description_
        device (_type_): _description_
        args (_type_): _description_

    Returns:
        dict: the performance of the model on the overall dataset and partitioned groups.
    """
    assert data_loader.batch_size == len(
        data_loader.dataset
    ), f"{data_loader.batch_size}！= {len(data_loader.dataset)}"
    model.eval()
    data = next(iter(data_loader))
    xc, xt, x_ext, y, day_of_week, time_of_day = load_data(device, data)
    pred, pred_day_of_week, pred_time_of_day = model(xc, xt, x_ext)
    res = compute_metrics(y, pred)
    mse, rmse, mae, mape, ape = get_error_performance(res)
    print(
        f"[Test] MSE: {mse:.2f}, RMSE(real): {rmse * args.m_factor:.2f},"
        f" MAE: {mae:.2f}, MAPE: {mape:.2f}, APE: {ape:.2f}"
    )
    day_of_week_accuracy = cal_accuracy(day_of_week, pred_day_of_week)
    time_of_day_accuracy = cal_accuracy(time_of_day, pred_time_of_day)
    group_result = group_test(model, data, device, args)

    performance = {
        "overall": res,
        "group_performance": group_result,
        "day_of_week_accuracy": day_of_week_accuracy,
        "time_of_day_accuracy": time_of_day_accuracy,
    }
    return performance


def cal_accuracy(true_label, pred_label):
    """get the accuracy of predicting time label (day of week and time of day).

    Args:
        true_label (torch.Tensor): (batch size, num_classes)
        pred_label (torch.Tensor): (batch size)

    Returns:
        float: accuracy
    """
    assert len(pred_label) == len(
        true_label
    ), "pred_label and true_label should have the same length"
    num_corrects = ((pred_label.argmax(-1) == true_label).sum()) / len(true_label)
    return (num_corrects / len(true_label)).item()


def get_error_performance(res):
    mse = res["mse"]["real"]
    rmse = res["rmse"]["real"]
    mae = res["mae"]["real"]
    mape = res["mape"]["real"]
    ape = res["ape"]["real"]
    return mse, rmse, mae, mape, ape


def to_numerical_value(data):
    """convert list of torch.tensor to numeical value

    Args:
        data (list): _description_

    Returns:
        list: _description_
    """
    assert isinstance(data, list) or isinstance(
        data, tuple
    ), "the data must be list/tuple of tensors"
    return [x.item() for x in data]


def compute_metrics(y, pred):
    """evaluate the predicted performance on five metrics and their masked version.

    Args:
        y (_type_): _description_
        pred (_type_): _description_

    Returns:
        dict: the performance of five evaluation metrics and the performance on masked version.
    """
    mse, rmse, mae, mape, ape = to_numerical_value(compute(y, pred))
    masked_mse, masked_rmse, masked_mae, masked_mape, masked_ape = to_numerical_value(
        compute_masked(y, pred)
    )
    res = {
        "mse": {"real": mse, "masked": masked_mse},
        "rmse": {"real": rmse, "masked": masked_rmse},
        "mae": {"real": mae, "masked": masked_mae},
        "mape": {"real": mape, "masked": masked_mape},
        "ape": {"real": ape, "masked": masked_ape},
    }
    return res


def group_test(model, data, device, args):
    """group the test by its time interval and test the performance of each group.

    Args:
        model (_type_): _description_
        data (_type_): _description_
        device (_type_): _description_
        args (_type_): _description_

    Returns:
        dict: the performance of five group:
            day_of_week: the performance of 7 day of week
            day_and_night: the performance of day and night
            weekdays: the performance of weekdays and weekends
            time_of_day: the performance of different time interval of day
            multi_step: the performance of multi_step predictions.
            The performance are evaluated on 5 metrics: MSE, RMSE, MAE, MAPE, APE and five masked metrics of them.
    """
    model = model.to(device)
    day_of_week_res = group_test_day_of_week(model, data, device, args)
    day_and_night_res = group_test_day_and_night(model, data, device, args)
    weekday_and_weekend_res = group_test_weekday_weekend(model, data, device, args)
    time_of_day_res = group_test_time_of_day(model, data, device, args)
    multi_step_res = multi_step_test(model, data, device, args)
    res = {}
    res["day_of_week"] = day_of_week_res
    res["day_and_night"] = day_and_night_res
    res["weekday_and_weekend"] = weekday_and_weekend_res
    res["time_of_day"] = time_of_day_res
    res["multi_step"] = multi_step_res
    return res


def group_test_day_of_week(model, data, device, args):
    """group the data by day of week.

    Args:
        model (_type_): _description_
        data (_type_): _description_
        device (_type_): _description_
        args (_type_): _description_
    returns (dict):
        the performance of each day of week.
        for each day,
        1. the performance are evaluated on 5 metrics: MSE, RMSE, MAE, MAPE, APE and five masked metrics of them.
        2. the day of week prediction accuracy on this day.


    """
    *_, day_of_week, time_of_day = load_data(device, data)
    day_idexs = [day_of_week == i for i in range(7)]  # get the idx of each day of week.
    peformance_res = {}
    week_name = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    for (
        day,
        flow_prediction_performance,
        accuracy_day_of_week,
        accuracy_time_of_day,
    ) in group_by(model, data, device, day_idexs, week_name):
        peformance_res[day] = {
            "flow_prediction_performance": flow_prediction_performance,
            f"accuracy_day_of_week_{day}": accuracy_day_of_week,
        }
    return peformance_res


def group_by(model, data, device, group_indexs, group_names):
    """Each group has its group index and corresponding group name,
    group the data are  by group_indexs, and test the performance of each group.

    Args:
        model (_type_):
        data (_type_):
        device (_type_):
        group_indexs (_type_):
        group_names (_type_):

    Yields:
        group_name: name of a group
        flow_prediction_performance: performance on a group of data.
        accuracy_day_of_week: accuracy of a specific day_of_week
        accuracy_time_of_day: accuracy of a specific time_of_day
    """
    assert len(group_indexs) == len(
        group_names
    ), "the length of group_indexs and group_names must be equal"
    for group_name, group_idx in zip(group_names, group_indexs):
        data_list = copy.copy(data)
        for i in range(len(data_list)):
            data_list[i] = data_list[i][group_idx]
        xc, xt, x_ext, y, day_of_week, time_of_day = load_data(device, data_list)
        # assert (day_of_week.cpu().sum()).item() == (
        #     day_of_week[0] * len(day_of_week)
        # ).item(), "the test data should be group to of the same day of week."
        pred, pred_day_of_week, pred_time_of_day = model(xc, xt, x_ext)

        flow_prediction_performance = compute_metrics(y, pred)
        accuracy_day_of_week = cal_accuracy(day_of_week, pred_day_of_week)
        accuracy_time_of_day = cal_accuracy(time_of_day, pred_time_of_day)

        yield group_name, flow_prediction_performance, accuracy_day_of_week, accuracy_time_of_day
    return


def get_day_range(T):
    """split the number of intervals of a day into day and night.
        We regard intervals at [7,20) as day and intervals at [20,7) as night.
        i.e., the time interval from morning 7 AM to 7 PM as day, and the time interval from 8 PM to 6 AM as night.
        If the T > 24, meaning the unit time interval is not 1 hour. For example, if T == 48, the time interval is half an hour.
        In this case, we compute the day range that takes half hour as base time interval.
    Args:
        T (int): number of intervals of a day
    Returns:
        day (list): two element list, the first one is the starting interval of day and the second one is the ending interval of day.
        night (list): two element list, the first one is the starting interval of night and the second one is the ending interval of night.
    """
    day = [7, 20]
    if T != 24:
        day = list(map(lambda x: (T // 24) * x, day))
    return day


def group_test_day_and_night(model, data, device, args):
    *_, day_of_week, time_of_day = load_data(device, data)
    day_range = get_day_range(args.T)
    day_index = cal_interval_in_range(time_of_day, day_range)
    night_index = ~day_index
    group_indexs = [day_index, night_index]
    group_names = ["day", "night"]
    res = {}
    for (
        name,
        flow_prediction_performance,
        accuracy_day_of_week,
        accuracy_time_of_day,
    ) in group_by(model, data, device, group_indexs, group_names):
        res[name] = {
            "flow_prediction_performance": flow_prediction_performance,
            f"accuracy_time_of_day_{name}": accuracy_time_of_day,
        }
    return res


def cal_interval_in_range(time_intervals, time_range):
    """calculate the index of intervals in the given time range.

    Args:
        time_intervals (torch.Tensor): must be of [,) format. compare equal in the left but not in the right.
        time_range (list): two element list, the first one is the starting interval of day and the second one is the ending interval of day

    Returns:
        torch.Tensor of boolean type: the index of intervals in the given time range.
    """
    return (time_intervals >= min(time_range)) & (time_intervals < max(time_range))


def group_test_weekday_weekend(model, data, device, args):
    *_, day_of_week, time_of_day = load_data(device, data)
    weekday_range = [0, 5]
    weekday_index = cal_interval_in_range(day_of_week, weekday_range)
    weekend_index = ~weekday_index
    group_indexs = [weekday_index, weekend_index]
    group_names = ["weekday", "weekend"]
    res = {}
    for (
        name,
        flow_prediction_performance,
        accuracy_day_of_week,
        accuracy_time_of_day,
    ) in group_by(model, data, device, group_indexs, group_names):
        res[name] = {
            "flow_prediction_performance": flow_prediction_performance,
            f"accuracy_time_of_day_{name}": accuracy_day_of_week,
        }
    return res


def group_test_time_of_day(model, data, device, args):
    *_, day_of_week, time_of_day = load_data(device, data)
    time_of_day_indexs = [
        time_of_day == i for i in range(args.T)
    ]  # get the idx of each day of week.
    group_names = [str(i) for i in range(args.T)]
    res = {}
    for (
        name,
        flow_prediction_performance,
        accuracy_day_of_week,
        accuracy_time_of_day,
    ) in group_by(model, data, device, time_of_day_indexs, group_names):
        res[name] = {
            "flow_prediction_performance": flow_prediction_performance,
            f"accuracy_time_of_day_at_interval_{name}": accuracy_time_of_day,
        }
    return res


def multi_step_test(model, data, device, args):
    return "Not Implemented!"


def make_pretrain_path(args):
    experiment_path = make_experiment_dir(args)
    pretrain_dir = "pretrain"
    pretrain_dir = os.path.join(experiment_path, pretrain_dir)
    if os.path.exists(pretrain_dir) is False:
        os.mkdir(pretrain_dir)
    return pretrain_dir, experiment_path


class MinMaxNormalization(object):
    """MinMax Normalization --> [-1, 1]
    x = (x - min) / (max - min).
    x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1.0 * (X - self._min) / (self._max - self._min)
        X = X * 2.0 - 1.0
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.0) / 2.0
        X = 1.0 * X * (self._max - self._min) + self._min
        return X


def compute_masked(y_true, y_pred, threshold=10):
    idx = y_true > threshold
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = mse**0.5
    mae = torch.mean(torch.abs(y_true - y_pred))
    mape = torch.mean(torch.abs((y_true - y_pred) / y_true))
    ape = torch.sum(torch.abs((y_true - y_pred) / y_true))
    return mse, rmse, mae, mape, ape


def compute(y_true, y_pred):
    """
    support computing Error metrics on two data type, torch.Tensor and np.ndarray.
    """

    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = mse**0.5
    mae = torch.mean(torch.abs(y_true - y_pred))
    idx = y_true > 1e-6
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    mape = torch.mean(torch.abs((y_true - y_pred) / y_true))
    ape = torch.sum(torch.abs((y_true - y_pred) / y_true))
    return mse, rmse, mae, mape, ape


def remove_incomplete_days(data, timestamps, T=48):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


def load_stdata(fname):
    # print('fname:', fname)
    f = h5py.File(fname, "r")
    data = f["data"][:]
    timestamps = f["date"][:]
    f.close()
    return data, timestamps


def string2timestamp(strings, T=48):
    """
    strings: list, eg. ['2017080912','2017080913']
    return: list, eg. [Timestamp('2017-08-09 05:30:00'), Timestamp('2017-08-09 06:00:00')]
    """
    timestamps = []
    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:]) - 1
        timestamps.append(
            pd.Timestamp(
                datetime(
                    year,
                    month,
                    day,
                    hour=int(slot * time_per_slot),
                    minute=(slot % num_per_T) * int(60.0 * time_per_slot),
                )
            )
        )

    return timestamps


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.data_1 = data[:, 0, :, :]
        self.data_2 = data[:, 1, :, :]
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append(
                    "(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i])
                )
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def get_matrix_1(self, timestamp):  # in_flow
        ori_matrix = self.data_1[self.get_index[timestamp]]
        new_matrix = ori_matrix[np.newaxis, :]
        # print("new_matrix shape:",new_matrix.shape) #(1, 32, 32)
        return new_matrix

    def get_matrix_2(self, timestamp):  # out_flow
        ori_matrix = self.data_2[self.get_index[timestamp]]
        new_matrix = ori_matrix[np.newaxis, :]
        # print("new_matrix shape:",new_matrix.shape) #(1, 32, 32)
        return new_matrix

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset_3D(
        self,
        len_closeness=3,
        len_trend=3,
        TrendInterval=7,
        len_period=3,
        PeriodInterval=1,
        prediction_offset=0,
    ):
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [
            range(1, len_closeness + 1),
            [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
            [TrendInterval * self.T * j for j in range(1, len_trend + 1)],
        ]

        i = max(
            self.T * TrendInterval * len_trend,
            self.T * PeriodInterval * len_period,
            len_closeness,
        )
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it(
                    [self.pd_timestamps[i] - j * offset_frame for j in depend]
                )

            if Flag is False:
                i += 1
                continue

            # closeness
            c_1_depends = list(depends[0])  # in_flow
            c_1_depends.sort(reverse=True)
            # print('----- c_1_depends:',c_1_depends)

            c_2_depends = list(depends[0])  # out_flow
            c_2_depends.sort(reverse=True)
            # print('----- c_2_depends:',c_2_depends)

            x_c_1 = [
                self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame)
                for j in c_1_depends
            ]  # [(1,32,32),(1,32,32),(1,32,32)] in
            x_c_2 = [
                self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame)
                for j in c_2_depends
            ]  # [(1,32,32),(1,32,32),(1,32,32)] out

            x_c_1_all = np.vstack(x_c_1)  # x_c_1_all.shape  (3, 32, 32)
            x_c_2_all = np.vstack(x_c_2)  # x_c_1_all.shape  (3, 32, 32)

            x_c_1_new = x_c_1_all[np.newaxis, :]  # (1, 3, 32, 32)
            x_c_2_new = x_c_2_all[np.newaxis, :]  # (1, 3, 32, 32)

            x_c = np.vstack([x_c_1_new, x_c_2_new])  # (2, 3, 32, 32)

            # period
            p_depends = list(depends[1])
            if len(p_depends) > 0:
                p_depends.sort(reverse=True)
                # print('----- p_depends:',p_depends)

                x_p_1 = [
                    self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame)
                    for j in p_depends
                ]
                x_p_2 = [
                    self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame)
                    for j in p_depends
                ]

                x_p_1_all = np.vstack(x_p_1)  # [(3,32,32),(3,32,32),...]
                x_p_2_all = np.vstack(x_p_2)  # [(3,32,32),(3,32,32),...]

                x_p_1_new = x_p_1_all[np.newaxis, :]  # (1, 3, 32, 32)
                x_p_2_new = x_p_2_all[np.newaxis, :]  # (1, 3, 32, 32)

                x_p = np.vstack([x_p_1_new, x_p_2_new])  # (2, 3, 32, 32)

            # trend
            t_depends = list(depends[2])
            if len(t_depends) > 0:
                t_depends.sort(reverse=True)

                x_t_1 = [
                    self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame)
                    for j in t_depends
                ]
                x_t_2 = [
                    self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame)
                    for j in t_depends
                ]

                x_t_1_all = np.vstack(x_t_1)  # [(3,32,32),(3,32,32),...]
                x_t_2_all = np.vstack(x_t_2)  # [(3,32,32),(3,32,32),...]

                x_t_1_new = x_t_1_all[np.newaxis, :]  # (1, 3, 32, 32)
                x_t_2_new = x_t_2_all[np.newaxis, :]  # (1, 3, 32, 32)

                x_t = np.vstack([x_t_1_new, x_t_2_new])  # (2, 3, 32, 32)

            y = self.get_matrix(self.pd_timestamps[i + prediction_offset])

            if len_closeness > 0:
                XC.append(x_c)
            if len_period > 0:
                XP.append(x_p)
            if len_trend > 0:
                XT.append(x_t)
            Y.append(y)
            timestamps_Y.append(self.timestamps[i + prediction_offset])
            i += 1

        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print(
            "3D matrix - XC shape: ",
            XC.shape,
            "XP shape: ",
            XP.shape,
            "XT shape: ",
            XT.shape,
            "Y shape:",
            Y.shape,
        )
        return XC, XP, XT, Y, timestamps_Y


def convert_timestr_to_day_of_week(time_byte):
    """convert a string of time to a day of week.
    Args:
        time_byte: a time.
    Returns:
        a day of week.
    """
    return time.strptime(str(time_byte[:8], encoding="utf-8"), "%Y%m%d").tm_wday

def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    vec = [
        time.strptime(str(t[:8], encoding="utf-8"), "%Y%m%d").tm_wday
        for t in timestamps
    ]  # python3
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)
