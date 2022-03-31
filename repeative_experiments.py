import os
import numpy as np
import pandas as pd
from arg_convertor import arg_class2json
from help_funcs import read_config_class
from train import pretrain, train
from utils import reproducibility
import copy

"""
run a experiments multiple times given args file.

args
    args: argument class where each property is a hyper-parameter.
            total_num_of_experiments: total number of experiments needed to be runed.
            num_of_experiment_once: number of experiments runed each time.
    
returns:
    test_results ([list[list]]): list of test results in each experiment.
"""


def run_repeative_experiments(args, total_num_of_experiments=10, path=None):
    """
    implementation:
        1. fix random seed.
        2. load args.
        3. generate different experiment args for each experiment (only ).
        4. run num_of_experiment_once experiments each time.
    """
    

    base_experiment_name = args.experiment_name
    test_results = []
    for i in range(total_num_of_experiments):
        reproducibility(i)
        experiment_name = f"{base_experiment_name}___{i}"
        args_temp = copy.copy(args)
        args_temp.experiment_name = experiment_name
        test_result = pretrain(args_temp) if args_temp.pretrain_times != 0 else train(args_temp)
        test_results.append(test_result)
    result = calculate_average_results(test_results)
    save_result(result, path)
    arg_class2json(args,os.path.join(path,"arg.json"))
    return result


def save_result(result, path):
    """save pandas dataframe to given path.

    Args:
        result (pd.DataFrame): result in dataframe format.
        path (str):
    """
    if not os.path.exists(path):
        os.makedirs(path)
    result.to_csv(os.path.join(path, "result.csv"))


def calculate_average_results(test_results):
    """calculate the mean and std of test results.

    Args:
        test_results (list[list]): the test results of all experiments, each result saves the five evaluations of "MSE", "RMSE", "MAE", "MAPE", "APE"

    Returns:
        result (DataFrame): the result of total experiments where the last two raw are the mean result and the std.
    """
    test_results = np.array(test_results)
    mean_results = pd.DataFrame(test_results.mean(axis=0))
    std_results = pd.DataFrame(test_results.std(axis=0))
    test_results_df = pd.DataFrame(
        test_results, columns=["MSE", "RMSE", "MAE", "MAPE", "APE"]
    )
    
    test_results_df = test_results_df.append(mean_results)
    test_results_df = test_results_df.append(std_results)
    return test_results_df


def ablation_study(total_num_of_experiments, args):
    """conduct ablation_study in an increamental fashion.

    Args:
        total_num_of_experiments (int): number of experiments in an ablation.
        args (_type_): config class
    """
    arg_temp = copy.copy(args)
    arg_temp.seq_pool = False
    base_experiment_name = args.experiment_name

    ext_dim = args.ext_dim
    arg_temp.pre_conv = False
    arg_temp.shortcut = False

    ablation_results = {}
    suffixs = [
        "ablation_all_dark",
        "_ablation_pre_conv",
        "_ablation_pre_conv_shortcut",
        "_ablation_pre_conv_shortcut_ext_dim",
        "_ablation_pre_conv_shortcut_ext_dim_pretrain",
    ]
    for i, suffix in enumerate(suffixs):
        fold_path = os.path.join("ablation_results",base_experiment_name,suffix)
        arg_temp.pre_conv = (i >= 1) 
        arg_temp.shortcut = i >= 2
        arg_temp.ext_dim = ext_dim if i >= 3 else 0
        arg_temp.pretrain_times = 1 if i >= 4 else 0
        ablation_results[suffix] = run_ablation_experiments(
            suffix, total_num_of_experiments, base_experiment_name, arg_temp, path=fold_path
        )
    


def run_ablation_experiments(
    suffix, total_num_of_experiments, base_experiment_name, arg_temp, path
):

    arg_temp.experiment_name = f"{base_experiment_name}_{suffix}"
    return run_repeative_experiments(arg_temp,total_num_of_experiments, path)

def test_repeative_experiments():
    args = read_config_class("arg1")
    args.epochs = 1
    args.pretrain_times = 1
    args.pretrain_epochs = 1
    args.experiment_name = "repeative_experiments_test!"
    run_repeative_experiments(args, total_num_of_experiments=1)

def test_repeative_experiments_no_pretrain():
    args = read_config_class("arg1")
    args.epochs = 1
    args.pretrain_times = 0
    args.pretrain_epochs = 1
    run_repeative_experiments(args, total_num_of_experiments=1)


def test_ablation_experiments():
    args = read_config_class("arg1")
    args.epochs = 1
    args.pretrain_times = 1
    args.pretrain_epochs = 1
    ablation_study(total_num_of_experiments=1,args=args)


def run_ablation_BikeNYC():
    args = read_config_class("BikeNYC_c6_t2")
    ablation_study(total_num_of_experiments=10,args=args)

def ablation_pre_conv_shortcut():
    args = read_config_class("BikeNYC_c6_t2")
    base_experiment_name = args.experiment_name 
    suffix = "_ablation_pre_conv_shortcut"
    fold_path = os.path.join("ablation_results",base_experiment_name,suffix)
    args.pre_conv = True
    args.shortcut = True
    args.ext_dim = 0
    args.pretrain_times = 0
    run_repeative_experiments(args, total_num_of_experiments=10, path=fold_path)

def ablation_pre_conv_shortcut_ext_dim():
    args = read_config_class("BikeNYC_c6_t2")
    base_experiment_name = args.experiment_name 
    suffix = "_ablation_pre_conv_shortcut_ext_dim"
    fold_path = os.path.join("ablation_results",base_experiment_name,suffix)
    args.pre_conv = True
    args.shortcut = True
    args.pretrain_times = 0
    run_repeative_experiments(args, total_num_of_experiments=10, path=fold_path)


def ablation_pre_conv_shortcut_ext_dim_pretrain():
    args = read_config_class("BikeNYC_c6_t2")
    base_experiment_name = args.experiment_name 
    suffix = "_ablation_pre_conv_shortcut_ext_dim_pretrain"
    fold_path = os.path.join("ablation_results",base_experiment_name,suffix)
    args.pretrain_times = 1
    args.pre_conv = True
    args.shortcut = True
    run_repeative_experiments(args, total_num_of_experiments=10, path=fold_path)
    

if __name__ == "__main__":
    args = read_config_class("BikeDC_c6_t2")
    ablation_study(10, args)