import argparse
import os
import torch
from STTransformer import create_model
from help_funcs import read_config_class
from train import get_loaders
from utils import compute, test
import pandas as pd


    
def mask_data(x,p=0.5):
    """randomly mask the historical data with bernoulli distribution and probability p.

    Args:
        x (torch.Tensor): historical data 
        p (int, optional): probability that controls the masking. Defaults to 0, do not mask any historical data.

    Raises:
        ValueError: p must be in the range (0,1)

    Returns:
        torch.Tensor: masked historical data
    """    
    if not (0 < p < 1):
        raise ValueError(f"the probability p must be in the range (0,1), received {p}")
    mask = torch.empty(x.shape).fill_(p)
    mask = torch.bernoulli(mask).bool()
    x[~mask] = 0
    return x

def rescale_flow_value(x,alpha=1.3):
    """scale the flow value of historical data

    Args:
        x (Torch.Tensor): historical data
        alpha (float, optional): . Defaults to 1.3.

    Returns:
        Torch.Tensor: scaled historical data.
    """    
    if alpha == 1:
        raise ValueError("alpha must be non-1 as to scale the flow value.")
    if alpha <= 0:
        raise ValueError("alpha must be non-negtive, but got %f" % alpha)

    return x * alpha

def get_test_data(data_loader):
    """get test data from test_data loader.

    Args:
        data_loader (_type_): _description_

    Returns:
        List[torch.Tensor]: input and predicted target.
    """    
    assert data_loader.batch_size == len(
        data_loader.dataset
    ), f"{data_loader.batch_size}！= {len(data_loader.dataset)}"
    data = next(iter(data_loader))
    xc, xt, x_ext, y = data
    return xc, xt, x_ext, y


def test_pertubated_data(model,pertubated_data, device):
    """test the performance of pertubated data."

    Args:
        model (nn.Module): torch model      
        pertubated_data (torch.Tensor): _description_
        device (_type_): cpu or cuda.

    Returns:
        List[Float]: test results
    """    
    loss_function = torch.nn.MSELoss()
    model.eval()
    xc, xt, x_ext, y = pertubated_data
    xc, xt, x_ext, y = xc.to(device), xt.to(device), x_ext.to(device), y.to(device)
    model = model.to(device)
    pred = model(xc, xt, x_ext)
    loss = loss_function(pred, y)
    MSE = loss.item()
    result = compute(y, pred)
    y_rmse, y_mae, y_mape, APE = result
    print(
        f"[Test] MSE: {MSE:.2f}, RMSE(real): {y_rmse:.2f},"
        f" MAE: {y_mae:.2f}, MAPE: {y_mape:.2f}, APE: {APE:.2f}"
    )
    return result
        
    
@torch.no_grad()
def pertubation_pipeline(model, data_loader, pertubation_func,device="cuda", *args, **kwargs):
    """ the pipeline of pertubation (load data, pertubate data, test performance)         

    Args:
        model (nn.Module): torch model
        data_loader (_type_): test loader
        pertubation_func (function): function that pertubate historical data
        device (str, optional): _description_. Defaults to "cuda".

    Raises:
        ValueError: test data must be xc,xt,x_ext, y.

    Returns:
        _type_: _description_
    """    
    test_data = get_test_data(data_loader)
    if len(test_data) != 4:
        raise ValueError("the test data must contain 4 elements.")
    # pertubate data. mask xc, xt
    test_data = list(test_data)
    for i in range(2):
        test_data[i] = pertubation_func(test_data[i],*args, **kwargs)
    # test the pertubation performance.
    results = test_pertubated_data(model,test_data, device=device)
    headers=[ "RMSE", "MAE", "MAPE", "APE"]
    df = {}
    for result,header in zip(results,headers):
        df.setdefault(header, []).append(result)
    df = pd.DataFrame(df) 
    df.to_csv("mask_pertubation.csv")
    return df  

def load_model(args, model_path):
    """generate a model with given achitecture and weights.

    Args:
        args (class): model config
        model_path (os.path): path where the model weights are stored. 

    Returns:
        _type_: _description_
    """    
    model = create_model(arg=args)
    model.load_state_dict(torch.load(model_path),)
    return model


if __name__ == "__main__":
    config_name="arg"
    experiment_path = "experiment/BikeDC/BikeDC_c6_t2"
    model_path = os.path.join(experiment_path, "best_model.pt")
    args = read_config_class(config_name)
    model = load_model(args,model_path=model_path)
    train_loader, val_loader, test_loader = get_loaders(args)
    pertubation_pipeline(model, test_loader, mask_data, device="cuda",p=0.5)
    """
    
    """