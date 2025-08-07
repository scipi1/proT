import numpy as np
from os.path import dirname, abspath, join
import torch
from pytorch_lightning import seed_everything
from pathlib import Path
from omegaconf import OmegaConf
from collections import OrderedDict
import math
root_path = dirname(dirname(abspath(__file__)))
import sys
sys.path.append(root_path)
from proT.forecaster import TransformerForecaster
from proT.labels import *
from proT.experiment_control import update_config



def record_called_modules(model):
        called = OrderedDict()                        # preserves call order

        def _make(name):
            def _hook(m, inp, out):
                called.setdefault(name, m.__class__.__name__)
            return _hook

        handles = [m.register_forward_hook(_make(n))
                   for n, m in model.named_modules()]

        return called, handles           # • called fills during forward
                                        # • handles lets you remove hooks



def get_emb_gradients(
    config_path: Path,
    datadir_path: Path, 
    checkpoint_path: Path,
    debug: bool=False
    )->dict:
    
    """
    Calculates the gradients of a given model output w.r.t. the embedding components.
    The model is loaded from a specified checkpoint and settings are loaded from a config file
    The gradients are calculated via forward and backward pass using the full input dataset
    Args:
        config_path (Path): absolute path to configuration file
        datadir_path (Path): absolute path to data directory
        checkpoint_path (Path): path to checkpoint
        features_dict (dict): dictionary {index: feature}, default None. 
                                If None, automatically look for it
        debug (bool): shows some debugging info
    Returns:
        dict: sensitivities for features_dict
    """
    
    config = OmegaConf.load(config_path) # load config
    
    # settings
    seed = config["training"]["seed"]
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = join(datadir_path,config["data"]["dataset"])
    input_file =  config["data"]["filename_input"]
    target_file = config["data"]["filename_target"]
    
    # load dataset arrays
    X = np.load(join(data_dir, input_file), allow_pickle=True, mmap_mode='r')
    Y = np.load(join(data_dir, target_file), allow_pickle=True, mmap_mode='r')
    
    # load tensors
    x_all = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)  # full input
    y_all = torch.tensor(Y, dtype=torch.float32, device=device, requires_grad=True)  # full input
    
    B,L,_ = x_all.shape    # batch size and seq length used for normalization
    
    config_updated = update_config(config)                      # update config
    model = TransformerForecaster(config_updated)               # load model
    forecaster = model.load_from_checkpoint(checkpoint_path)    # load checkpoint
    
    # Control statement to check if the model loaded correctly
    if forecaster is None:
        raise RuntimeError("Model failed to load from checkpoint.")

    # Check if model parameters are properly loaded (ensure they are not uninitialized)
    if not any(param.requires_grad for param in forecaster.parameters()):
        raise RuntimeError("Model parameters seem uninitialized. Check the checkpoint path.")
    
    # set up model
    forecaster.to(device)
    
    if debug:
        called, handles = record_called_modules(forecaster.model)
    
    forecaster.eval().requires_grad_(True)
    
    
    # Hook for embeddings 
    encoder_embedding = forecaster.model.enc_embedding.embed_modules_list
    
    acts = {}
    
    def make_hook(name):
        def _hook(module, inp, out):
            if debug:
                print(f"[HOOK] {name:10s}  out.shape = {tuple(out.shape)}")
            
            if out.requires_grad:          # guard
                out.retain_grad()
                acts[name] = out
            else:
                # still create a placeholder so .items() won't skip this feature
                acts.setdefault(name, None)
                print(f"{name} has requires_grad_ off")
        return _hook
    
    for idx, emb in enumerate(encoder_embedding):          # loop once
        emb.embedding.register_forward_hook(make_hook(f"feat{idx}"))   # dynamic name
    
    
    
    y, *_ = forecaster(    # forward pass
        data_input=x_all,
        data_trg=y_all,
        kwargs=None
        )
    
    if debug:
        # show what actually executed
        for name, cls in called.items():
            print(f"{name:40s}  {cls}")

        for h in handles: h.remove()  
    
    y.sum().backward()          # backwards
    
    if debug:
        print("acts keys:", list(acts.keys()))
        for k, t in acts.items():
            print(f"{k:10s} grad mean={t.grad.abs().mean().item():.3e}")
    
    # calculate sensitivities
    S = {
        name: (
        math.nan                               # placeholder for “no grad”
        if act is None or act.grad is None
        else act.grad.square().sum().sqrt().item() / (B * L)
        )
        for name, act in acts.items()
        }
    
    for act in acts.values():
        if act is not None:
            act.grad = None
    
    return S





if __name__ == "__main__":
    
    datadir_path = join(root_path,"data","input")
    
    checkpoint_path = r"C:\Users\ScipioneFrancesco\Documents\Projects\prochain_transformer\experiments\training\dx_250618_cat\emb20_mod20\sweeps\sweep_dec_pos_emb_hidden\sweep_dec_pos_emb_hidden_20\k_0\checkpoints\epoch=199-train_loss=0.00.ckpt"
    config_path = r"C:\Users\ScipioneFrancesco\Documents\Projects\prochain_transformer\experiments\training\dx_250618_cat\emb20_mod20\sweeps\sweep_dec_pos_emb_hidden\sweep_dec_pos_emb_hidden_20\config.yaml"
    
    S = get_emb_gradients(
        config_path,
        datadir_path, 
        checkpoint_path,
        debug=True
        )
    breakpoint()
