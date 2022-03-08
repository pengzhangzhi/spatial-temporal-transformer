
import optuna
from optuna.trial import TrialState
from utils import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from STTransformer import create_model
from help_funcs import EarlyStop, print_run_time, read_config_class
from train import get_loaders, get_optim

from utils import reproducibility, train_one_epoch


def objective(trial):
    """
    :param args: arg class.
    """
    try:
        args = read_config_class("arg")
        args.lr = trial.suggest_float("lr", 0.005, 0.01, log=True)
        args.batch_size = trial.suggest_categorical("batch_size",[32,62,128,256,512])
        args.epochs = trial.suggest_int("epochs",500,700,50)
        args.conv_channels = trial.suggest_int("conv_channels",32,256,32)
        args.patch_size = trial.suggest_categorical("patch_size",[2,4,8,16])
        args.close_dim = trial.suggest_categorical("close_dim",[128,64,32,256])
        args.trend_dim = trial.suggest_categorical("trend_dim",[128,64,32,256])
        args.close_depth = trial.suggest_categorical("close_depth",[1,2,3,4,5,6,7,8])
        args.trend_depth = trial.suggest_categorical("trend_depth",[1,2,3,4,5,6,7,8])
        args.close_head = trial.suggest_categorical("close_head",[1,2,3,4])
        args.trend_head = trial.suggest_categorical("trend_head",[1,2,3,4])
        args.close_mlp_dim = trial.suggest_categorical("close_mlp_dim",[128,256,512,1024])
        args.trend_mlp_dim = trial.suggest_categorical("trend_mlp_dim",[128,256,512,1024])
        device = args.device


        # save print info to log.txt

        train_loader, val_loader, test_loader = get_loaders(args)

        model = create_model(arg=args)
    

        optimizer, scheduler = get_optim(model, args)
        for epoch in range(1):
            # train
            train_loss, train_rmse = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)

            # print("train:",train_loss, train_rmse)
            scheduler.step()

            # validate
            val_loss, val_rmse,val_mape = evaluate(model=model,
                                        data_loader=val_loader,
                                        device=device,
                                        epoch=epoch)

            # print("val:", val_loss, val_rmse)

            # results = [train_loss, train_rmse, val_loss, val_rmse, optimizer.param_groups[0]["lr"]]

            
        if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    except Exception as e:
        print(e)
    # print("Val_RMSE:",val_rmse,type(val_rmse))
    return val_rmse


if __name__ == '__main__':
    reproducibility()
    study = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50, timeout=3600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))