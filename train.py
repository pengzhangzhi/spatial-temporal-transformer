'''=================================================

@Project -> File：ST-Transformer->train

@IDE：PyCharm

@coding: utf-8

@time:2021/7/24 7:21

@author:Pengzhangzhi

@Desc：
=================================================='''
import argparse
import math
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter

from STTransformer import create_model
from help_funcs import read_config_class, split_dataset, make_experiment_dir, save_train_history, save_test_results, \
    Logger, print_run_time, EarlyStop
from read_data import load
from utils import train_one_epoch, evaluate, test


def make_pretrain_path(args):
    experiment_path = make_experiment_dir(args)
    pretrain_dir = "pretrain"
    pretrain_dir = os.path.join(experiment_path, pretrain_dir)
    if os.path.exists(pretrain_dir) is False:
        os.mkdir(pretrain_dir)

    return pretrain_dir, experiment_path


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, x, y,timestamp,*args,**kwargs):
        super(Mydataset, self).__init__()
        assert len(x[0]) == len(y)
        self.x = x
        self.y = y
       # half_hour = list(map(lambda x: (x.astype(int))%100, timestamp))
       #  if min(half_hour) == 1:
       #     half_hour = list(map(lambda x: x - 1, half_hour))
       # half_hour = torch.LongTensor(half_hour)
        day_of_week = list(map(lambda x: int(x[-4:-2]) % 7 , timestamp))
        self.timestamp = day_of_week
        self.length = len(y)

    def __getitem__(self, idx):
        ret = (*[x[idx] for x in self.x], self.y[idx],self.timestamp[idx])
        return ret

    def __len__(self):
        return self.length


def get_optim(model, args):
    pg = [p for p in model.parameters() if p.requires_grad]

    # close_softmax = model.closeness_transformer.softmax.parameters()
    # trend_softmax = model.trend_transformer.softmax.parameters()
    # softmax_params = list(close_softmax)+list(trend_softmax)
    # ignored_params = list(map(id, softmax_params))  # 返回的是parameters的 内存地址
    # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = optim.Adam(pg, lr=args.lr, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs/4),eta_min=args.lr*args.lrf)

    return optimizer, scheduler


def get_loaders(args, pretrain=False):
    """

    :param pretrain:
    :param args:
    :return:
    """
    # if pretrain, load pretrain dataset else load train dataset(normal dataset)
    X_train, Y_train, X_test, Y_test, \
    mmn, external_dim, timestamp_train, timestamp_test = list(load(args,pretrain))
    batch_size = args.batch_size
    split = args.split
    split = split * 0.3 if pretrain else split
    train_dataset = Mydataset(X_train, Y_train,timestamp_train)
    test_dataset = Mydataset(X_test, Y_test, timestamp_test)

    nw = 0
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader, val_loader = split_dataset(train_dataset,
                                             split=split,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=len(test_dataset),
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              )

    return train_loader, val_loader, test_loader


@print_run_time
def train(args, model=None, experiment_path=None):
    """
    :param args: arg class.
    """
    print("Training.")
    # tb_writer = SummaryWriter()
    tb_writer = None
    device = args.device

    experiment_path = make_experiment_dir(args) if experiment_path is None else experiment_path

    # save print info to log.txt
    sys.stdout = Logger(os.path.join(experiment_path, "log.txt"))

    train_loader, val_loader, test_loader = get_loaders(args)
    if model is None:
        model = create_model(arg=args)
    else:
        model = model

    optimizer, scheduler = get_optim(model, args)
    model_path = os.path.join(experiment_path, "best_model.pt")
    early_stop = EarlyStop(patience=int(args.epochs * 0.5), path=model_path)

    for epoch in range(args.epochs):
        # train
        train_loss, train_rmse, class_loss, class_accuracy = train_one_epoch(model=model,
                                                                             optimizer=optimizer,
                                                                             data_loader=train_loader,
                                                                             device=device,
                                                                             epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_rmse, val_class_loss, val_class_accuracy = evaluate(model=model,
                                                                          data_loader=val_loader,
                                                                          device=device,
                                                                          epoch=epoch)

        results = [train_loss, train_rmse, class_loss, class_accuracy
            , val_loss, val_rmse, val_class_loss, val_class_accuracy
            , optimizer.param_groups[0]["lr"]]

        save_train_history(experiment_path, results, epoch, tb_writer)

        # test
        MSE, y_rmse, y_mae, y_mape, relative_error = test(model=model,
                                                          data_loader=test_loader,
                                                          device=device,
                                                          args=args)

        test_results = [MSE, y_rmse * args.m_factor_2, y_mae, y_mape, relative_error]

        save_test_results(test_results, experiment_path)

        if early_stop(val_rmse, model):
            print("early stop~")
            break

    model.load_state_dict(torch.load(model_path))
    MSE, y_rmse, y_mae, y_mape, relative_error = test(model=model,
                                                      data_loader=test_loader,
                                                      device=device,
                                                      args=args)
    save_test_results(test_results, experiment_path)
    result = str(y_rmse)
    rmse_format = "a".join(result.split("."))

    with open(f"{rmse_format}.txt", "w") as f:
        f.write(f"{y_rmse}")


@print_run_time
def pretrain(args, ):
    print("pretraining model.")
    # tb_writer = SummaryWriter() if you want to use tensorboard
    tb_writer = None
    device = args.device

    pretrain_dir, experiment_path = make_pretrain_path(args)
    # save print info to log.txt
    sys.stdout = Logger(os.path.join(pretrain_dir, "log.txt"))

    model = create_model(arg=args)

    # if has model pretrain checkpoint, load it.
    model_checkpoint_path = os.path.join(pretrain_dir, "best_model.pt")
    if os.path.exists(model_checkpoint_path):
        print(f"load checkpoint: {model_checkpoint_path} to pretrain.")
        model.load_state_dict(torch.load(model_checkpoint_path))  # load best model

    for i in range(args.pretrain_times):
        optimizer, scheduler = get_optim(model, args)

        early_stop = EarlyStop(patience=int(args.epochs * 0.5), path=model_checkpoint_path)

        print(f"pretraining model at {i} th times")
        train_loader, val_loader, test_loader = get_loaders(args, pretrain=True)
        for epoch in range(args.pretrain_epochs):
            # train
            train_loss, train_rmse,class_loss,class_accuracy = train_one_epoch(model=model,
                                                     optimizer=optimizer,
                                                     data_loader=train_loader,
                                                     device=device,
                                                     epoch=epoch)

            scheduler.step()

            # validate
            val_loss, val_rmse,val_class_loss,val_class_accuracy = evaluate(model=model,
                                          data_loader=val_loader,
                                          device=device,
                                          epoch=epoch)

            results = [train_loss, train_rmse,class_loss,class_accuracy
                , val_loss, val_rmse,val_class_loss,val_class_accuracy
                , optimizer.param_groups[0]["lr"]]

            save_train_history(pretrain_dir, results, epoch, tb_writer)

            # test
            MSE, y_rmse, y_mae, y_mape, relative_error = test(model=model,
                                                              data_loader=test_loader,
                                                              device=device,
                                                              args=args)

            test_results = [MSE, y_rmse*args.m_factor_2, y_mae, y_mape, relative_error]

            save_test_results(test_results, pretrain_dir)

            if early_stop(val_rmse, model):
                print("early stop~")
                break
    if os.path.exists(model_checkpoint_path):
        model.load_state_dict(torch.load(model_checkpoint_path))  # load best model
    train(args, model, experiment_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config-name', type=str, default="TaxiNYC")
    parser.add_argument("-e", '--exp-name', type=str)
    parser.add_argument("-pe", '--pretrain-epochs', type=int, default=0)
    parser.add_argument("-pt", '--pretrain-times', type=int, default=-1)
    parser.add_argument("-noPretrain", action="store_true")
    opt = parser.parse_args()
    config_name = opt.config_name
    exp_name = opt.exp_name
    args = read_config_class(config_name=config_name)
    args.experiment_name = exp_name if exp_name else args.experiment_name
    args.pretrain_times = opt.pretrain_times if opt.pretrain_times is not -1 else args.pretrain_times
    args.pretrain_epochs = opt.pretrain_epochs if opt.pretrain_epochs else args.pretrain_epochs

    if opt.noPretrain:
        print("No pretrain, directly training.")
        train(args)
    else:
        pretrain(args)


