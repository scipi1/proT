from os.path import dirname, abspath
root_dir = dirname(dirname(abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from labels import *

def predict(
    model: pl.LightningModule,
    dm: pl.LightningDataModule,
    dataset_label: str,
    input_mask: torch.Tensor=None,
    debug_flag: bool=False,
    kwargs=None):
    
    assert dataset_label in ["train","test","val"], AssertionError("Invalid dataset label!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dm = dm
    forecaster = model.to(device)
    
    forecaster.eval()
    input_list, output_list, target_list,cross_att_list = nn.ParameterList(), nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
    zero_list = []
    vs_list = []
    
    # select dataset
    if dataset_label == "train":
        dataset = dm.train_dataloader()
        print("Train dataset selected.")
    else:
        dataset = dm.test_dataloader()
        print("Test dataset selected (default).")
    
    for batch in dataset:
        
        if isinstance(batch, (list, tuple)):  # In case your DataLoader returns a tuple (data, target)
            batch = [item.to(device) for item in batch]
        else:
            batch = batch.to(device)  # If batch is a single tensor

        X,trg = batch
        
        # mask the input for GSA
        if input_mask is not None:
            binary_mask = input_mask.float()
            
            if X.dim() == 3:
                M = binary_mask.view(1,1,binary_mask.size(0))
            elif X.dim() == 2:
                M = binary_mask.view(1,binary_mask.size(0))
                
            M_expanded = M.expand_as(X)
            M_expanded=M_expanded.to(device)
            
            X = X*M_expanded
            
        with torch.no_grad():
            forecast_output, recon_output, (enc_self_attns, dec_cross_attns,vs) = forecaster.forward(
                data_input=X,
                data_trg=trg,
                kwargs=kwargs)
            
        # append batch predictions
        input_list.append(X)
        output_list.append(forecast_output)
        target_list.append(trg)
        cross_att_list.append(dec_cross_attns[0])

        
        vs_list.append(vs)
            
        if debug_flag:
            print("Debug mode: stopping after one batch...")
            break
    
    # detach predictions
    input_tensor = torch.cat([t.cpu().detach() for t in input_list], dim=0)
    output_tensor = torch.cat([t.cpu().detach() for t in output_list], dim=0)
    target_tensor = torch.cat([t.cpu().detach() for t in target_list], dim=0)
    cross_att_tensor = torch.cat([t.cpu().detach() for t in cross_att_list], dim=0)
    vs_list = [[t.cpu().detach()for t in p] for p in vs_list]
    
    # convert predictions to numpy
    input_array = input_tensor.numpy().squeeze()
    output_array = output_tensor.numpy().squeeze()
    target_array = target_tensor.numpy().squeeze()
    cross_att_array = cross_att_tensor.numpy().squeeze()

    vs_list = 0
    
    return input_array, output_array, target_array, cross_att_array,vs_list