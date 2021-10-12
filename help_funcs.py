'''=================================================

@Project -> File：ST-3DNet-main->help_funcs

@IDE：PyCharm

@coding: utf-8

@time:2021/7/19 9:42

@author:Pengzhangzhi

@Desc：
=================================================='''
import datetime
import math
import os
import pickle
import time
from shutil import copyfile
import sys
import pandas as pd
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

from arg_convertor import json2dict, arg_class2json


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def read_config(config_name="BikeNYC"):
    """read in hyperparams from config file, return a dict.

    """
    dir = os.getcwd()
    config_file = os.path.join(dir, 'config', f'{config_name}.json')
    print(config_file)
    if not os.path.exists(config_file):
        raise ValueError(f'config file {config_file} not exists.')
    training_config = json2dict(config_file)
    return training_config


def read_config_class(config_name="BikeNYC"):
    config_dict = read_config(config_name)
    return type("temp", (), config_dict)


def split_dataset(dataset, split=0.1, batch_size=32, shuffle=True, *args, **kwargs):
    """split a dataset into a larger loader and smaller loader as the given argument split.
    args:
        dataset(torch.utils.data.Dataset): torch dataset.
        split(float): fraction between 0 to 1, denoting the portion of the smaller loader. the size of the larger loader is 1-split.
        batch_size: number of mini-batch batch_size
        shuffle(noolean): shuffle the dataset if shuffle is true.
    returns:
        samll_loader: torch loader with less samples.
        larger_loader: torch loader with more samples.
    example:
        >>> from torchvision import datasets
        >>> dataset = datasets.MNIST('data', train=True, download=True)
        >>> train_loader, test_loader = split_dataset(dataset)

    """
    from torch.utils.data import SubsetRandomSampler, DataLoader
    import numpy as np
    total_len = len(dataset)
    split_len = int(total_len * split)
    indices = list(range(total_len))
    if shuffle:
        np.random.shuffle(indices)
    larger_indices = indices[split_len:]
    smaller_indices = indices[:split_len]
    larger_sampler = SubsetRandomSampler(larger_indices)
    samll_sampler = SubsetRandomSampler(smaller_indices)
    larger_loader = DataLoader(dataset, batch_size, sampler=larger_sampler, *args, **kwargs)
    samll_loader = DataLoader(dataset, batch_size, sampler=samll_sampler, *args, **kwargs)
    return larger_loader, samll_loader


def make_experiment_dir(args):
    """
    generate experiment directory for saving result.


    :param filename:
    :param data:
    returns:
        experiment_path: experiment directory varies with time.
        fname_param: name of saved model
    """
    dir = os.getcwd()
    if not os.path.exists(os.path.join(dir, "experiment")):  # make experiment directory
        os.mkdir(os.path.join(dir, "experiment"))
    expdir = os.path.join(dir, "experiment", args.dataset)  # directory of dataset experiment: experiment/TaxiBJ
    if os.path.exists(expdir) is False:
        os.mkdir(expdir)  # make dataset experiment
    time = datetime.datetime.now().strftime('%m-%d, %H-%M')  # experiment name
    experiment_name = args.experiment_name if args.experiment_name is not None else time

    experiment_path = os.path.join(expdir, experiment_name)  # experiment/TaxiBJ/Time

    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    print('experiment_path:', experiment_path)

    # save arg to experiment_path
    save_experiment_args(args, experiment_path)

    return experiment_path


def save_experiment_args(args, experiment_path):
    """

    :param args:  arg class
    :param experiment_path: experiment/TaxiBJ/Time
    """
    target_path = os.path.join(experiment_path, "arg.json")
    arg_class2json(args, target_path)


def save_train_history(experiment_path, results, epoch, tb_writer=None):
    """

    :param experiment_path: experiment/TaxiBJ/Time
    :param results: [train_loss, train_rmse, val_loss, val_rmse, optimizer.param_groups[0]["lr"]]
    :param epoch: current epoch
    :param tb_writer: tensorbard
    """
    tags = ["train_loss", "train_rmse", "train_class_loss","train_class_accuracy",
            "val_loss", "val_rmse","val_class_loss","val_class_accuracy", "learning_rate"]
    history = {}
    print(f"[Train Epoch({epoch})]")
    for tag, result in zip(tags, results):
        history.setdefault(tag, []).append(result)

        print(f"{tag}: {result:.2f}")
        if tb_writer:
            tb_writer.add_scalar(tag, result, epoch)
    print("=" * 20)
    save_history(history, experiment_path)



def copy_config_file(target_dir, config_name="BikeNYC"):
    dir = os.getcwd()
    config_dir = os.path.join(dir, "config")
    config_name = f"{config_name}.json"
    config_path = os.path.join(config_dir, config_name)
    target_path = os.path.join(target_dir, config_name)
    copyfile(config_path, target_path)


def load_data(filename):
    # load data
    f = open(filename, 'rb')
    X_train = pickle.load(f)
    Y_train = pickle.load(f)
    X_test = pickle.load(f)
    Y_test = pickle.load(f)
    mmn = pickle.load(f)
    external_dim = pickle.load(f)
    timestamp_train = pickle.load(f)
    timestamp_test = pickle.load(f)

    print('X_train:')
    for i in range(len(X_train)):  # x_train: xc,xt,(xp,x_ext)
        X_train[i] = torch.tensor(X_train[i], dtype=torch.float32)
        print(X_train[i].shape)
    print('X_test:')
    for i in range(len(X_test)):  # x_train: xc,xt,(xp,x_ext)
        X_test[i] = torch.tensor(X_test[i], dtype=torch.float32)
        print(X_train[i].shape)

    Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
    Y_test = mmn.inverse_transform(Y_test)

    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test


def save_history(history, experiment_path="./"):
    # TODO: append instead of overide.
    df = pd.DataFrame(history)
    name = "history.csv"
    path = os.path.join(experiment_path, name)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a+", header=header)


def save_test_results(test_results, experiment_path):
    # TODO: append instead of overide.
    """
    save test result.
    :param test_results: [MSE, y_rmse, y_mae, y_mape, relative_error]
    :param experiment_path: path for saving experiment result.
    """
    test_tags = ["MSE", "RMSE", "MAE", "MAPE", "relative_error"]
    test_result = {}
    for tag, result in zip(test_tags, test_results):
        test_result.setdefault(tag, []).append(result)
    name = "test_result.csv"
    df = pd.DataFrame(test_result)
    path = os.path.join(experiment_path, name)
    header = not os.path.exists(path)

    df.to_csv(path, mode="a", header=header)


def save_results(test_rmse, test_mae, test_mape, experiment_path):
    """ save test result.
    """
    name = "test_result.csv"
    result_dict = {"test_rmse": [test_rmse], "test_mae": [test_mae], "test_mape": [test_mape]}
    df = pd.DataFrame(result_dict)
    path = os.path.join(experiment_path, name)
    df.to_csv(path)


def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        duration = (time.time() - local_time) / 60
        print('run time is %.2f min' % duration)

    return wrapper


def summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary


class EarlyStop():
    """ Apply earlyStop and save_checkpoint during the training process.
    usage:


    """

    def __init__(self, patience: int = 50, mode: str = "min", delta: float = 0.,
                 path: str = "best-Model.pth", verbose: bool = True):
        """

            Args:
            patience(int): How long to wait after last model improved.
            mode(str): One of { "min", "max"}.
                        In min mode, training will stop when the quantity monitored has stopped decreasing;
                        in "max" mode it will stop when the quantity monitored has stopped increasing;
                         in "auto" mode, the direction is automatically inferred from the name of the monitored quantity.

            delta(float): Minimum change in the monitored quantity.
            path(str): Path to save the best model.
            verbose(bool): if True print out messages each time.
            example:
            >>> earlyStop = EarlyStop(verbose=True)  # intialize earlyStop class
            >>> loss = 50000
            >>> model = torch.nn.Sequential(torch.nn.Linear(10, 10))
            >>> for i in range(1000):
            >>>     if i < 100:
            >>>         loss -= 5
            >>>     print(f"loss:{loss}")
            >>>     if earlyStop(loss, model):  # add earlyStop,
            >>>         break

        """

        self.mode = mode.lower()
        assert self.mode in ["min", "max"], 'mode must be one of [min, max]'
        self.best_score = math.inf if self.mode == "min" else -math.inf
        self.patience = patience
        self.delta = delta
        self.cnt = 0
        self.stop = False
        self.path = path
        self.vobose = verbose
        self.messages = []

    def __call__(self, record: float, model: torch.nn.Module) -> bool:

        """
        monitor the training process.
        args:

        record(float): the quantity you want to monitored, can be val_loss or val_accuracy.
        model(torch.nn.Module): pytorch model.
        self.stop(bool): if model do not improve after patience,return true.

        return:
            self.stop(bool): if model do not improve after patience,return true.



        """

        if self._achieve_better(record):
            if self.vobose:
                message = f"Metric has imporved from " \
                          f"{self.best_score:.2f} --> {record:.2f}"
                self.messages.append(message)
                print(message)

            self.best_score = record
            self.cnt = 0
            self.save_checkpoint(model)

        else:
            self.cnt += 1
            if self.cnt > self.patience:
                if self.vobose:
                    message = "Training out of patience, Early Stop!"
                    self.messages.append(message)
                    print(message)
                self.stop = True

        return self.stop

    def save_checkpoint(self, model):
        """  save pytorch model to the specified file path."""
        torch.save(model.state_dict(), self.path)
        if self.vobose:
            message = f"Best Model Saved in path: {self.path}!"
            self.messages.append(message)
            print(message)

    def _achieve_better(self, record: float) -> bool:
        ''' decide whether the record is better than the existing best score.'''
        if self.mode == "min":
            return record < self.best_score - self.delta

        return record > self.best_score + self.delta
