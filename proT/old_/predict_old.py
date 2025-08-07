from os import makedirs
from os.path import dirname, abspath, join, exists
root_dir = dirname(dirname(abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning as L
from proT.dataloader import ProcessDataModule
from forecaster import TransformerForecaster
from labels import *
from config import load_config

def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                

def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)                                             


def predict(
    exp_id,
    checkpoint,
    model_given=False,
    dm=None,
    model=None,
    input_mask:torch.Tensor=None,
    kwargs=None):
    
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    INPUT_DIR,_,_, EXPERIMENTS_DIR = get_dirs(ROOT_DIR)
    EXP_DIR = join(EXPERIMENTS_DIR,exp_id)
    
    output_path = join(EXP_DIR,"output")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not exists(output_path):
        makedirs(output_path)
    
    if model_given:
        dm = dm
        forecaster = model.to(device)
    
    else:
        
        config = load_config(EXP_DIR)
        dataset = config["data"]["dataset"]
        
        checkpoint_path = join(EXP_DIR,"checkpoints",checkpoint)
        
        dm = ProcessDataModule(
            data_dir=join(INPUT_DIR,dataset),
            files=("X_np.npy","Y_np.npy"),
            batch_size=config["training"]["batch_size"],
            num_workers = 20)
        
        dm.prepare_data()
        dm.setup(stage=None)
        model = TransformerForecaster(config)
        forecaster = model.load_from_checkpoint(checkpoint_path)
    
    forecaster.eval()
    output_list, target_list = nn.ParameterList(),nn.ParameterList()
    cross_att_list = []
    vs_list = []
    
    batches = [batch for batch in dm.train_dataloader()]
    batches = batches[0]
    
    debug_flag= True
    for batch in dm.train_dataloader():
        
        if debug_flag==True:
        
            if isinstance(batch, (list, tuple)):  # In case your DataLoader returns a tuple (data, target)
                batch = [item.to(device) for item in batch]
            else:
                batch = batch.to(device)  # If batch is a single tensor

            X,trg = batch
            
            if input_mask is not None:
                binary_mask = input_mask.float()
                #binary_mask=binary_mask = torch.where(binary_mask == 0, torch.tensor(float('-inf')), torch.tensor(1.0))
                expanded_mask = binary_mask.unsqueeze(0).expand(X.shape[0], X.shape[1])
                expanded_mask=expanded_mask.to(device)
                print(expanded_mask)
                X = X*expanded_mask
                
                
            #with torch.no_grad():
            forecast_output, recon_output, (enc_self_attns, dec_cross_attns,vs) = forecaster.forward(
                data_input=X,
                data_trg=trg,
                kwargs=kwargs)
            
            # H = hessian(forecast_output,vs[0])
            H=0
            
            
            
            output_list.append(forecast_output)
            target_list.append(trg)
            cross_att_list.append(dec_cross_attns)
            vs_list.append(vs)
            
        debug_flag=False
        
    output_tensor = torch.cat([t.cpu().detach() for t in output_list], dim=0)
    target_tensor = torch.cat([t.cpu().detach() for t in target_list], dim=0)
    cross_att_list = [[t.cpu().detach()for t in p] for p in cross_att_list]
    vs_list = [[t.cpu().detach()for t in p] for p in vs_list]

    
    
    output_array = output_tensor.numpy().flatten()
    target_array = target_tensor.numpy().flatten()
    cross_att_tensor_array = np.array(cross_att_list)
    vs_list = np.array(vs_list)
    # cross_att_tensor_array = [param.detach().cpu().numpy() for param in cross_att_list]
    
    # pred_val = np.array([t.squeeze().cpu().detach().numpy() for t in output_list])
    # true_val = np.array([t.squeeze().cpu().detach().numpy() for t in target_list])
    # cross_att = np.array([[l.squeeze().cpu() for l in t] for t in cross_att_list])
    
    
    
    # SAVE OPTS
    # with open(join(output_path,"output.npy"), 'wb') as f:
    #     np.save(f, pred_val)
    # with open(join(output_path,"target.npy"), 'wb') as f:
    #     np.save(f, true_val)
    # with open(join(output_path,"cross_att.npy"), 'wb') as f:
    #     np.save(f, cross_att)
        
    return output_array, target_array, cross_att_tensor_array,vs_list,H


if __name__ == "__main__":
    exp_id = "experiment_002"
    checkpoint = "epoch=86-val_loss=0.00.ckpt"
    predict(exp_id,checkpoint)