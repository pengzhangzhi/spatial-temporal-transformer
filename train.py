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
    def __init__(self, x, y,normalization=None):
        super(Mydataset, self).__init__()
        assert len(x[0]) == len(y)
        self.x = x
        self.y = y
        self.length = len(y)
        self.normalization = normalization

    def __getitem__(self, idx):
        # x_all = [x[idx] for x in self.x[:-1]]
        # x_all = [rearrange("n c h w -> c n h w", x) for x in x_all]
        # lengths = [len(x) for x in x_all]
        # lengths += [self.x[-1],1]
        # y = rearrange("n h w -> 1 n h w", self.y[idx])
        # x_all.append(y)
        # set = torch.cat(x_all, dim=0)
        # his_length = len(set) - 1
        # exchange_idx = torch.randint(0, his_length - 1, size=(1,))
        # set[exchange_idx], set[-1] = set[-1], set[exchange_idx]
        # # set = rearrange(set,"c n h w -> n c h w")
        #
        # for item in list(torch.split(set,lengths))
        if self.normalization:
            xc,xt,x_ext = [x[idx] for x in self.x]
            y = self.y[idx]
            y = rearrange(y, "n h w -> 1 n h w")

            xc, xt = list(map(lambda x: rearrange(x, "n l h w -> l n h w"), [xc, xt]))

            data = torch.cat([xc, xt, y], dim=0)  # l' b n h w
            his_len = len(data) - 1
            idx = torch.randint(0, his_len - 1, (1,))
            temp_y = data[-1].clone()

            data[-1] = data[idx]
            data[idx] = temp_y
            chunk_len = [len(xc), len(xt), 1]
            xc, xt, y = list(map(lambda x: rearrange(x, "l n h w ->  n l h w"), list(torch.split(data, chunk_len))))
            y = rearrange(y, "n 1 h w -> n h w")
            # normalize data[idx]
            data[idx] = self.normalization.transform(data[idx])
            # renormalize y
            y = self.normalization.inverse_transform(y)
            ret = (xc, xt,x_ext, y)
        else:

            ret = (*[x[idx] for x in self.x], self.y[idx])

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
    pretrain_way = args.pretrain_way
    split = split * 0.3 if pretrain else split
    mmn = mmn if pretrain else None
    train_dataset = Mydataset(X_train, Y_train, normalization=mmn)
    test_dataset = Mydataset(X_test, Y_test, )

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
        train_loss, train_rmse = train_one_epoch(model=model,
                                                 optimizer=optimizer,
                                                 data_loader=train_loader,
                                                 device=device,
                                                 epoch=epoch)

        # print("train:",train_loss, train_rmse)
        scheduler.step()

        # validate
        val_loss, val_rmse = evaluate(model=model,
                                      data_loader=val_loader,
                                      device=device,
                                      epoch=epoch)

        # print("val:", val_loss, val_rmse)

        results = [train_loss, train_rmse, val_loss, val_rmse, optimizer.param_groups[0]["lr"]]

        save_train_history(experiment_path, results, epoch, tb_writer)

        # test
        MSE, y_rmse, y_mae, y_mape, relative_error = test(model=model,
                                                          data_loader=test_loader,
                                                          device=device,
                                                          args=args)
        # append a error list so as to save a error list.
        test_results = [MSE, y_rmse*args.m_factor_2, y_mae, y_mape, relative_error]

        save_test_results(test_results, experiment_path)

        if early_stop(val_rmse, model):
            print("early_stop")
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
            train_loss, train_rmse = train_one_epoch(model=model,
                                                     optimizer=optimizer,
                                                     data_loader=train_loader,
                                                     device=device,
                                                     epoch=epoch)

            scheduler.step()

            # validate
            val_loss, val_rmse = evaluate(model=model,
                                          data_loader=val_loader,
                                          device=device,
                                          epoch=epoch)

            results = [train_loss, train_rmse, val_loss, val_rmse, optimizer.param_groups[0]["lr"]]

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
    parser.add_argument("-c", '--config-name', type=str, default="TaxiBJ")
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


