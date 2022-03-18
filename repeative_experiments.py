import os
import numpy as np
import pandas as pd
from help_funcs import read_config_class
from train import pretrain, train
from utils import reproducibility


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
    
    if path is None:
        path = f"./{args.experiment_name}_num_of_exp_{total_num_of_experiments}.csv"

    base_experiment_name = args.experiment_name
    test_results = []
    for i in range(len(total_num_of_experiments)):
        reproducibility(i)
        experiment_name = f"{base_experiment_name}___{i}"
        args_temp = args.copy()
        args_temp.experiment_name = experiment_name
        if args_temp.pretrain:
            test_result = pretrain(args_temp)
        else:
            test_result = train(args_temp)
        test_results.append(test_result)
    result = calculate_average_results(test_results)
    save_result(result, path)
    return result


def save_result(result, path):
    """save pandas dataframe to given path.

    Args:
        result (pd.DataFrame): result in dataframe format.
        path (str):
    """
    os.makedirs(path)
    result.to_csv(path)


def calculate_average_results(test_results):
    """calculate the mean and std of test results.

    Args:
        test_results (list[list]): the test results of all experiments, each result saves the five evaluations of "MSE", "RMSE", "MAE", "MAPE", "APE"

    Returns:
        result (DataFrame): the first raw is the mean result and the second raw is the std.
    """
    test_results = np.array(test_results)
    mean_results = test_results.mean(axis=1)
    std_results = test_results.std(axis=1)
    mean_results = pd.DataFrame(
        mean_results, columns=["MSE", "RMSE", "MAE", "MAPE", "APE"]
    )
    result = mean_results.join(std_results)
    return result


def ablation_study(total_num_of_experiments, args, path):

    arg_temp.seq_pool = False
    base_experiment_name = args.experiment_name

    arg_temp = args.copy()
    arg_temp.pre_conv = False
    arg_temp.shortcut = False
    arg_temp.ext_dim = 0
    arg_temp.pretrain_times = 0

    ablation_results = {}
    suffixs = [
        "ablation_all_dark",
        "_ablation_pre_conv",
        "_ablation_pre_conv_shortcut",
        "_ablation_pre_conv_shortcut_ext_dim",
        "_ablation_pre_conv_shortcut_ext_dim_pretrain",
    ]
    for i, suffix in enumerate(suffixs):
        arg_temp.pre_conv = i == 1
        arg_temp.shortcut = i == 2
        arg_temp.ext_dim = args.ext_dim if i == 3 else 0
        arg_temp.pretrain_times = 1 if i == 4 else 0
        ablation_results[suffix] = run_ablation_experiments(
            suffix, total_num_of_experiments, base_experiment_name, arg_temp, path
        )


def run_ablation_experiments(
    suffix, total_num_of_experiments, base_experiment_name, arg_temp, path
):

    arg_temp.experiment_name = f"{base_experiment_name}_{suffix}"
    return run_repeative_experiments(total_num_of_experiments, arg_temp, path)


if __name__ == "__main__":
    args = read_config_class("arg1")
    run_repeative_experiments(args, total_num_of_experiments=10)
