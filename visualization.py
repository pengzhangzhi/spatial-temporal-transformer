'''=================================================

@Project -> File：base transformer->visualization

@IDE：PyCharm

@coding: utf-8

@time:2021/8/5 8:15

@author:Pengzhangzhi

@Desc：
load model from given path.
debug to checkout value
=================================================='''
import argparse
import os

import torch
from STTransformer import create_model
from help_funcs import read_config_class
from train import get_loaders
from utils import test

def main(args, path):
    model = create_model(arg=args)
    model.load_state_dict(torch.load(path))
    train_loader, val_loader, test_loader = get_loaders(args)
    MSE, y_rmse, y_mae, y_mape, relative_error = test(model=model,
                                                      data_loader=test_loader,
                                                      device=args.device,
                                                      args=args)


def load_aux_dict(model_path,model):
    """
    load main network's weight to aux network.

    :param model_path: path of saved model state_dict.
    :param model: created model.
    :return: model that have been loaded weights.
    """
    source_param_dict = torch.load(model_path)
    # param_dict = {k:v for k,v in param_dict.items() if "main" in k}

    model_dict = model.state_dict()
    # temp = model_dict.copy()
    # model_dict = {k:param_dict[k.replace("auxiliary","main")] for k,v in param_dict.items() if "auxiliary" in k}
    for k in model_dict:
        if "auxiliary" in k:
            model_dict[k] = source_param_dict[k.replace("auxiliary", "main")]
    model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",'--config_name', type=str, default="TaxiBJ")
    opt = parser.parse_args()
    config_name = opt.config_name

    args = read_config_class(config_name=config_name)
    dataset = args.dataset
    experiment_name = args.experiment_name
    experiment_path = os.path.join("./experiment",dataset, experiment_name)
    model_path = os.path.join(experiment_path,"best_model.pt")

    model = create_model(arg=args)
    # load_aux_dict(model_path,model)

    train_loader, val_loader, test_loader = get_loaders(args)
    MSE, y_rmse, y_mae, y_mape, relative_error = test(model=model,
                                                      data_loader=test_loader,
                                                      device=args.device,
                                                      args=args)
