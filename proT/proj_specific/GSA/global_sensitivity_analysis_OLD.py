
# FUNCTIONS TEMPORARY PARKED HERE; 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import dirname, abspath, join
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from typing import Tuple
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
from tqdm import tqdm
import yaml

# Local imports
from proT.training.dataloader import ProcessDataModule
from proT.training.forecasters.transformer_forecaster import TransformerForecaster
from proT.labels import *
from proT.training.experiment_control import update_config





# TODO move somewhere else
def predict_GSA_kill_feature(
    config: dict, 
    datadir_path: Path, 
    checkpoint_path: Path, 
    cluster: bool=False
    )->Tuple[np.ndarray]:
    
    # TODO. documentation
    
    seed = config["training"]["seed"]
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    
    # get data module from config
    dm = ProcessDataModule(
        data_dir = join(datadir_path,config["data"]["dataset"]),
        input_file =  config["data"]["filename_input"],
        target_file = config["data"]["filename_target"],
        batch_size =  config["training"]["batch_size"],
        num_workers = 1 if cluster else 20,
        data_format = "float32",
        seed = seed,
    )
    
    
    # TODO: move into fun arguments
    control_idx = 3
    kill_idx = 4
    #_________________________________
    
    
    # get test indexes from config
    test_ds_idx_filename = config["data"]["test_ds_ixd"]
    assert test_ds_idx_filename is not None, AssertionError("Invalid test dataset index file")
    
    test_idx = np.load(join(datadir_path,config["data"]["dataset"],test_ds_idx_filename))
    # TODO: add the possibility to use the whole dataset (training+test) for GSA
    
    # update data module ds
    dm.update_idx(train_idx=test_idx, val_idx=test_idx,test_idx=test_idx)
    
    # get unique keys of the control index
    keys = get_control_keys(
        dm=dm, 
        dataset_label="test",
        control_idx=control_idx)
    
    
    # update config
    config_updated = update_config(config)
    
    # load model
    model = TransformerForecaster(config_updated)

    # load checkpoint
    forecaster = model.load_from_checkpoint(checkpoint_path)
    
    # Control statement to check if the model loaded correctly
    if forecaster is None:
        raise RuntimeError("Model failed to load from checkpoint.")

    # Check if model parameters are properly loaded (ensure they are not uninitialized)
    if not any(param.requires_grad for param in forecaster.parameters()):
        raise RuntimeError("Model parameters seem uninitialized. Check the checkpoint path.")
    
    kill_keys_dict = {}
    
    for key in keys:
    
        # call predict
        pred_tuple = predict_mask(
            model=forecaster,
            dm=dm,
            dataset_label = "test",
            key=key,
            control_idx=control_idx,
            kill_idx=kill_idx,
            input_mask=None,
            debug_flag=False,
            kwargs=None)
        
        kill_keys_dict[key] = pred_tuple
        
    
    # TODO: save prediction on dictionary
    
    return input_array, output_array, target_array, cross_att_array



def get_control_keys(
    dm: pl.LightningDataModule,
    dataset_label: str,
    control_idx: int
    ):
    # TODO: add documentation
    
    assert dataset_label in ["train","test","val"], AssertionError("Invalid dataset label!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # prepare data module
    dm.prepare_data()
    dm.setup(stage=None)
    
    # select dataset
    if dataset_label == "train":
        dataset = dm.train_dataloader()
        print("Train dataset selected.")
    else:
        dataset = dm.test_dataloader()
        print("Test dataset selected (default).")
    
    keys = []
    
    # loop over prediction batches
    print("Scanning...")
    for batch in tqdm(dataset):
        
        if isinstance(batch, (list, tuple)):  # In case your DataLoader returns a tuple (data, target)
            batch = [item.to(device) for item in batch]
        else:
            batch = batch.to(device)  # If batch is a single tensor

        X,_ = batch
        
        keys.append(X[:,:,control_idx].unique())
        
    keys = torch.cat(keys).unique()
    keys = keys[~keys.isnan()]
    
    return keys


def predict_mask(
    model: pl.LightningModule,
    dm: pl.LightningDataModule,
    dataset_label: str,
    key: str|int|float,
    control_idx: int,
    kill_idx: int,
    input_mask: torch.Tensor=None,
    debug_flag: bool=False,
    kwargs=None)-> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Makes prediction feeding the `model` with the dataset specified by a data module
    and a label.

    Args:
        model (pl.LightningModule): transformer model
        dm (pl.LightningDataModule): data module
        dataset_label (str): choose between ["test", "train"]
        debug_flag (bool, optional): Predicts one batch and stops. Defaults to False.
        kwargs (optional): kwargs. Defaults to None.

    Returns:
        Tuple[  
            np.ndarray (input data),
            np.ndarray (prediction),
            np.ndarray (target data),
            list       (decoder cross-attention)
                ]: input_array, output_array, target_array, cross_att_array
    """
    
    assert dataset_label in ["train","test","val"], AssertionError("Invalid dataset label!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set model for prediction
    forecaster = model.to(device)
    forecaster.eval()
    
    input_list, output_list, target_list,cross_att_list = nn.ParameterList(), nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
    
    # prepare data module
    dm.prepare_data()
    dm.setup(stage=None)
    
    # select dataset
    if dataset_label == "train":
        dataset = dm.train_dataloader()
        print("Train dataset selected.")
    else:
        dataset = dm.test_dataloader()
        print("Test dataset selected (default).")
        
    # loop over prediction batches
    print("Predicting...")
    for batch in tqdm(dataset):
        
        if isinstance(batch, (list, tuple)):  # In case your DataLoader returns a tuple (data, target)
            batch = [item.to(device) for item in batch]
        else:
            batch = batch.to(device)  # If batch is a single tensor

        X,trg = batch
        
        
        mask = X[:, :, control_idx] == key
        X[:, :, kill_idx][mask] = float('nan')
        
        
        with torch.no_grad():
            forecast_output, recon_output, (enc_self_att, dec_self_att, dec_cross_att), enc_mask = forecaster.forward(
                data_input=X,
                data_trg=trg,
                kwargs=kwargs)
            
        # append batch predictions
        input_list.append(X)
        output_list.append(forecast_output)
        target_list.append(trg)
        cross_att_list.append(dec_cross_att[0])

        if debug_flag:
            print("Debug mode: stopping after one batch...")
            break
    
    # detach predictions
    input_tensor = torch.cat([t.cpu().detach() for t in input_list], dim=0)
    output_tensor = torch.cat([t.cpu().detach() for t in output_list], dim=0)
    target_tensor = torch.cat([t.cpu().detach() for t in target_list], dim=0)
    cross_att_tensor = torch.cat([t.cpu().detach() for t in cross_att_list], dim=0)
    
    # convert predictions to numpy
    input_array = input_tensor.numpy().squeeze()
    output_array = output_tensor.numpy().squeeze()
    target_array = target_tensor.numpy().squeeze()
    cross_att_array = cross_att_tensor.numpy().squeeze()
    
    return input_array, output_array, target_array, cross_att_array