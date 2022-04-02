"""=================================================

@Project -> File：ST-Transformer->train

@IDE：PyCharm

@coding: utf-8

@time:2021/7/24 7:21

@author:Pengzhangzhi

@Desc：
=================================================="""
import argparse
import math
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from einops import rearrange
from STTransformer import create_model
from help_funcs import (
    read_config_class,
    split_dataset,
    make_experiment_dir,
    save_train_history,
    save_test_results,
    Logger,
    print_run_time,
    EarlyStop,
)
from read_data import load
from utils import make_pretrain_path, train_one_epoch, evaluate, test


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, x, y, timestamp, args, **kwargs):
        super(Mydataset, self).__init__()
        assert len(x[0]) == len(y)
        self.length = len(y)
        self.x = x
        self.y = y
        self.day_of_week = list(map(lambda x: int(x[-4:-2]) % 7, timestamp))
        self.time_of_day = list(map(lambda x: (x.astype(int)) % 100, timestamp))
        if min(self.time_of_day) == 1:
            self.time_of_day = list(map(lambda x: x - 1, self.time_of_day))

    def __getitem__(self, idx):
        ret = (
            *[x[idx] for x in self.x],
            self.y[idx],
            self.day_of_week[idx],
            self.time_of_day[idx],
        )
        return ret

    def __len__(self):
        return self.length


def get_optim(model, args):
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, weight_decay=5e-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf)
        + args.lrf
    )  # cosine
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
    (
        X_train,
        Y_train,
        X_test,
        Y_test,
        mmn,
        external_dim,
        timestamp_train,
        timestamp_test,
    ) = list(load(args))
    batch_size = args.batch_size
    split = args.split
    split = split * 0.3 if pretrain else split
    train_dataset = Mydataset(X_train, Y_train, timestamp_train, args)
    test_dataset = Mydataset(X_test, Y_test, timestamp_test, args)

    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )  # number of workers
    print("Using {} dataloader workers every process".format(nw))

    train_loader, val_loader = split_dataset(
        train_dataset,
        split=split,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
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
    device = args.device
    experiment_path = (
        make_experiment_dir(args) if experiment_path is None else experiment_path
    )

    # save print info to log.txt
    sys.stdout = Logger(os.path.join(experiment_path, "log.txt"))
    train_loader, val_loader, test_loader = get_loaders(args)
    model = load_model(args, model)
    optimizer, scheduler = get_optim(model, args)
    model_path = os.path.join(experiment_path, "best_model.pt")
    early_stop = EarlyStop(patience=int(60), path=model_path)
    train_loop(
        args.epochs,
        args,
        device,
        experiment_path,
        model,
        optimizer,
        scheduler,
        early_stop,
        train_loader,
        val_loader,
    )

    model.load_state_dict(torch.load(model_path))
    test_results = test(model=model, data_loader=test_loader, device=device, args=args)
    save_test_results(test_results, experiment_path)


def load_model(args, model):
    if model is None:
        model = create_model(arg=args)
    return model


@print_run_time
def pretrain(
    args,
):
    print("pretraining model.")
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
    if args.pretrain_times:
        optimizer, scheduler = get_optim(model, args)
        early_stop = EarlyStop(
            patience=int(args.epochs * 0.5), path=model_checkpoint_path
        )
        train_loader, val_loader, test_loader = get_loaders(args, pretrain=True)
        train_loop(
            args.pretrain_epochs,
            args,
            device,
            pretrain_dir,
            model,
            optimizer,
            scheduler,
            early_stop,
            train_loader,
            val_loader,
        )
    # test
    if os.path.exists(model_checkpoint_path):
        model.load_state_dict(torch.load(model_checkpoint_path))  # load best model
    test_results = test(model=model, data_loader=test_loader, device=device, args=args)
    save_test_results(test_results, pretrain_dir)
    train(args, model, experiment_path)


def train_loop(
    epochs,
    args,
    device,
    pretrain_dir,
    model,
    optimizer,
    scheduler,
    early_stop,
    train_loader,
    val_loader,
):
    """an training loop (train + validate) of `epochs` times.

    Args:
        args (_type_): _description_
        device (_type_): _description_
        pretrain_dir (_type_): _description_
        model (_type_): _description_
        optimizer (_type_): _description_
        scheduler (_type_): _description_
        early_stop (_type_): _description_
        train_loader (_type_): _description_
        val_loader (_type_): _description_
    """
    for epoch in range(epochs):
        # train
        (
            train_loss,
            train_rmse,
            train_day_of_week_accuracy,
            train_time_of_day_accuracy,
        ) = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            args=args,
        )

        scheduler.step()

        # validate
        (
            val_loss,
            val_rmse,
            val_day_of_week_accuracy,
            val_time_of_day_accuracy,
        ) = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)

        results = [
            train_loss,
            train_rmse,
            train_day_of_week_accuracy,
            train_time_of_day_accuracy,
            val_loss,
            val_rmse,
            val_day_of_week_accuracy,
            val_time_of_day_accuracy,
            optimizer.param_groups[0]["lr"],
        ]

        save_train_history(pretrain_dir, results, epoch)
        if early_stop(val_rmse, model):
            print("early stop~")
            break


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-name", type=str, default="TaxiNYC")
    parser.add_argument("-e", "--exp-name", type=str)
    parser.add_argument("-pe", "--pretrain-epochs", type=int, default=0)
    parser.add_argument("-pt", "--pretrain-times", type=int, default=-1)
    parser.add_argument("-noPretrain", action="store_true")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_arg()
    config_name = opt.config_name
    exp_name = opt.exp_name
    args = read_config_class(config_name=config_name)
    args.experiment_name = exp_name if exp_name else args.experiment_name
    args.pretrain_times = (
        opt.pretrain_times if opt.pretrain_times is not -1 else args.pretrain_times
    )
    args.pretrain_epochs = (
        opt.pretrain_epochs if opt.pretrain_epochs else args.pretrain_epochs
    )

    if opt.noPretrain:
        print("No pretrain, directly training.")
        train(args)
    else:
        pretrain(args)
