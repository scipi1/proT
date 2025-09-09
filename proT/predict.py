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

root_path = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(root_path)
from proT.dataloader import ProcessDataModule
from proT.forecaster import TransformerForecaster
from proT.labels import *
from proT.experiment_control import update_config



def predict(
    model: pl.LightningModule,
    dm: pl.LightningDataModule,
    dataset_label: str,
    debug_flag: bool=False,
    show_trg_max_idx: int=None)-> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list, list]:
    
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
    
    assert dataset_label in ["train","test", "all"], AssertionError("Invalid dataset label!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set model for prediction
    forecaster = model.to(device)
    forecaster.eval()
    
    input_list, output_list, target_list, cross_att_list,enc_self_att_list, dec_self_att_list = nn.ParameterList(), nn.ParameterList(), nn.ParameterList(), nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
    
    # prepare data module
    dm.prepare_data()
    dm.setup(stage=None)
    
    # select dataset
    if dataset_label == "train":
        dataset = dm.train_dataloader()
        print("Train dataset selected.")
        
    elif dataset_label == "test":
        dataset = dm.test_dataloader()
        print("Test dataset selected (default).")
        
    elif dataset_label == "all":
        dataset = dm.all_dataloader()
        print("All data selected (default).")
        
    # loop over prediction batches
    print("Predicting...")
    for batch in tqdm(dataset):
        
        if isinstance(batch, (list, tuple)):  # In case your DataLoader returns a tuple (data, target)
            batch = [item.to(device) for item in batch]
        else:
            batch = batch.to(device)  # If batch is a single tensor

        X,trg = batch
        
        with torch.no_grad():
            forecast_output,_,(enc_self_att, dec_self_att, dec_cross_att),_, _ = forecaster.forward(
                data_input=X,
                data_trg=trg,
                show_trg_max_idx=show_trg_max_idx,
                predict_mode=True,
                )
            
        # append batch predictions
        input_list.append(X)
        output_list.append(forecast_output)
        target_list.append(trg)
        cross_att_list.append(dec_cross_att[0])
        enc_self_att_list.append(enc_self_att[0])
        dec_self_att_list.append(dec_self_att[0])

        if debug_flag:
            print("Debug mode: stopping after one batch...")
            break
    
    # detach predictions
    input_tensor = torch.cat([t.cpu().detach() for t in input_list], dim=0)
    output_tensor = torch.cat([t.cpu().detach() for t in output_list], dim=0)
    target_tensor = torch.cat([t.cpu().detach() for t in target_list], dim=0)
    cross_att_tensor = torch.cat([t.cpu().detach() for t in cross_att_list], dim=0)
    enc_self_att_tensor = torch.cat([t.cpu().detach() for t in enc_self_att_list], dim=0)
    dec_self_att_tensor = torch.cat([t.cpu().detach() for t in dec_self_att_list], dim=0)
    
    # convert predictions to numpy
    input_array = input_tensor.numpy().squeeze()
    output_array = output_tensor.numpy().squeeze()
    target_array = target_tensor.numpy().squeeze()
    cross_att_array = cross_att_tensor.numpy().squeeze()
    enc_self_att_array = enc_self_att_tensor.numpy().squeeze()
    dec_self_att_array = dec_self_att_tensor.numpy().squeeze()
    
    return input_array, output_array, target_array, cross_att_array, enc_self_att_array, dec_self_att_array



def mk_quick_pred_plot(model: pl.LightningModule, dm: pl.LightningDataModule, val_idx: int, save_dir: Path):
    
    input_array, output_array, target_array, cross_att_array,_,_ = predict(model=model, dm=dm, dataset_label="test")
    
    # in case we have only one sample
    if len(output_array.shape) == 1:
        output_array = np.expand_dims(output_array,axis=0)
        target_array = np.expand_dims(target_array,axis=0)
        cross_att_array = np.expand_dims(cross_att_array,axis=0)
        input_array = np.expand_dims(input_array,axis=0)
    
    y_out = output_array[0,:]
    y_trg = target_array[0,:,val_idx]
    cross_att = cross_att_array[0]
    input_miss_bool = np.isnan(input_array[0,:,val_idx].squeeze())
    input_miss = input_miss_bool[np.newaxis,:].astype(int)
    
    N, M = cross_att.shape
    assert len(y_out) == len(y_trg)
    x = np.arange(len(y_out))

    fig, ax = plt.subplots()
    ax.plot(x,y_out)
    ax.plot(x,y_trg)
    fig.savefig(join(save_dir,"quick_pred_plot.png"))
    
    # Create figure and grid
    fig2 = plt.figure(figsize=(6, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1], hspace=0.5)

    # Main heatmap axis
    ax0 = fig2.add_subplot(gs[0])
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.1)
    im0 = ax0.imshow(cross_att, cmap='viridis', aspect='auto', origin='upper')
    fig2.colorbar(im0, cax=cax0, label='Value')
    ax0.set_xticks([])
    ax0.set_ylabel("Rows")
    ax0.set_title("Heatmap with Boolean Mask")

    # Boolean mask axis
    ax1 = fig2.add_subplot(gs[1], sharex=ax0)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    im1 = ax1.imshow(input_miss, cmap='Greys', aspect='auto', origin='upper', vmin=0, vmax=1)
    fig2.colorbar(im1, cax=cax1, ticks=[0, 1], label='Missing')
    cax1.set_yticklabels(['False', 'True'])

    ax1.set_yticks([])
    ax1.set_xlabel("Columns")
    num_labels = min(M, 10)  # Ensure we don't exceed 10 labels
    step = M // num_labels if M > 10 else 1  # Calculate step for the labels

    ax1.set_xticks(np.arange(0, M, step))  # Set ticks at evenly spaced intervals
    ax1.set_xticklabels(np.arange(0, M, step))  # Show corresponding labels
    
    fig2.savefig(join(save_dir,"cross_att.png"), dpi=300, bbox_inches='tight')



def predict_test_from_ckpt(
    config: dict, 
    datadir_path: Path, 
    checkpoint_path: Path,
    external_dataset: dict=None,
    dataset_label: str="test",
    show_trg_max_idx: int=None,
    cluster: bool=False
    )->Tuple[np.ndarray]:
    
    """
    Runs the prediction steps according to config. In the specific:
    1) Loads the model at a given checkpoint
    2) Loads the dataset as specified by the config
    3) Calls the `predict` method
    Args:
        config (dict): conf for model loading
        data_dir (Path): path to data directory
        checkpoint_path (Path): path to checkpoint
        cluster (bool, optional): Running on the cluster? Defaults to False.

    Returns:
        Tuple[np.ndarray]: input_array, output_array, target_array, cross_att_array
    """
    
    assert dataset_label in ["train","test","all"], AssertionError(f"{dataset_label} is not a proper label!")
    
    seed = config["training"]["seed"]
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    
    if external_dataset is not None:
        dm = ProcessDataModule(
            data_dir = join(datadir_path, external_dataset["dataset"]),
            input_file =  external_dataset["filename_input"],
            target_file = external_dataset["filename_target"],
            batch_size =  config["training"]["batch_size"],
            num_workers = 1 if cluster else 20,
            data_format = "float32",
            seed = seed,
            )
    
    else:
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
    
    if config["data"]["test_ds_ixd"] is not None:
        test_idx = np.load(join(datadir_path, config["data"]["dataset"], config["data"]["test_ds_ixd"]))
    
        if external_dataset:
            # update data module ds
            dm.update_idx(train_idx=None, val_idx=None, test_idx=None)
        else:
            dm.update_idx(train_idx=None, val_idx=None, test_idx=test_idx)
    
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
    
    # call predict
    input_array, output_array, target_array, cross_att_array, enc_self_att_array, dec_self_att_array = predict(
        model=forecaster,
        dm=dm,
        dataset_label = dataset_label,
        debug_flag=False,
        show_trg_max_idx=show_trg_max_idx
        )
    
    return input_array, output_array, target_array, cross_att_array, enc_self_att_array, dec_self_att_array






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








# if __name__ == "__main__":
    
#     # exp_id = "test"
#     # exp_path = join(root_path, "experiments","training")
#     datadir_path = join(root_path,"data","input")
    
#     checkpoint_path = r"C:\Users\ScipioneFrancesco\Documents\Projects\prochain_transformer\experiments\training\test\k_0\checkpoints\epoch0-initial.ckpt"
#     config_path = r"C:\Users\ScipioneFrancesco\Documents\Projects\prochain_transformer\experiments\training\test\config.yaml"
    
#     config = OmegaConf.load(config_path)
#     input_array, output_array, target_array, cross_att_array = predict_GSA_kill_feature(config, datadir_path, checkpoint_path, cluster=False)
    
#     print(f"input shape: {input_array.shape}")
#     print(f"output shape: {output_array.shape}")
#     print(f"target shape: {target_array.shape}")
#     print(f"attention shape: {cross_att_array.shape}")
    
    
if __name__ == "__main__":
    datadir_path = r"../data/input"
    config_path = r"../experiments/training/proT/proT_cat_dyconex_optimized/config_proT_dyconex_v5_1.yaml"
    checkpoint_path = r"../experiments/training/proT/proT_cat_dyconex_optimized/k_0/checkpoints/epoch0-initial.ckpt"

    external_dataset =  {
        "dataset":"ds_dx_pred_panel_MSI_01_01_2022-07_07_2025",
        "filename_input": "X.npy",
        "filename_target": "Y.npy",
    }

    config = OmegaConf.load(config_path)
    input_array, output_array, target_array, cross_att_array = predict_test_from_ckpt(
        config, 
        datadir_path, 
        checkpoint_path, 
        external_dataset,
        dataset_label="all",
        cluster=False
        )

