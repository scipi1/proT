from os.path import exists,join
from os import mkdir, scandir
import fnmatch
from typing import Tuple
import warnings
from omegaconf import OmegaConf
import logging
import itertools


def find_yml_files(dir:str)-> Tuple[dict]:
    """
    Look for configuration and sweep file in a directory (dir)

    Args:
        dir (str): path to look for file, usually experiment folder

    Raises:
        FileNotFoundError: dir doesn't contain the config file
        Warning: dir doesn't contain the sweep filed

    Returns:
        Tuple[dict]: config_dict, sweep_dict
    """
    config_control_string = "config"
    sweep_control_string = "sweep"
    
    config, sweep_config = None, None
    
    with scandir(dir) as entries:
    
        for entry in entries:
            if entry.is_file():
                
                if fnmatch.fnmatch(entry.name,f"*{config_control_string}*.yaml"):
                    config = OmegaConf.load(entry.path)
                    
                if fnmatch.fnmatch(entry.name,f"*{sweep_control_string}*.yaml"):
                    sweep_config = OmegaConf.load(entry.path)
                    
    if config is None:
        raise FileNotFoundError("No configuration file found")
    
    if sweep_config is None:
        warnings.warn("No available sweep found")
        config = update_config(config) # update config anyways
    
    return config, sweep_config





def calculate_concat_dim(config, embed_prefix: str, modules_path: str) -> int:
    """
    Dynamically calculate concatenation dimension by checking which embeddings
    are actually used in the modules configuration.
    
    This prevents errors when:
    - An embedding dimension exists in config but is not used (embed not in modules)
    - An embedding dimension doesn't exist in config (e.g., dec_val_given_emb_hidden for non-adaptive models)
    
    Args:
        config: OmegaConf configuration object
        embed_prefix: "enc" or "dec" 
        modules_path: Path to modules list (e.g., "model.kwargs.ds_embed_enc.modules")
        
    Returns:
        int: Total dimension for concatenation
        
    Example:
        # For encoder: checks model.kwargs.ds_embed_enc.modules
        # For decoder: checks model.kwargs.ds_embed_dec.modules
    """
    total_dim = 0
    modules = OmegaConf.select(config, modules_path)
    
    if modules is None:
        return 0
    
    # Map embedding labels to their dimension keys in config
    label_to_dim = {
        "process": f"{embed_prefix}_pro_emb_hidden",
        "occurrence": f"{embed_prefix}_occ_emb_hidden",
        "step": f"{embed_prefix}_step_emb_hidden",
        "value": f"{embed_prefix}_val_emb_hidden",
        "variable": f"{embed_prefix}_var_emb_hidden", 
        "position": f"{embed_prefix}_pos_emb_hidden",
        "time": f"{embed_prefix}_time_emb_hidden",
        "online_pos_mask": f"{embed_prefix}_val_given_emb_hidden",  # For adaptive models
    }
    
    # Check which embeddings are actually used and sum their dimensions
    for module in modules:
        label = module.get("label")
        if label in label_to_dim:
            dim_key = f"model.embed_dim.{label_to_dim[label]}"
            dim_value = OmegaConf.select(config, dim_key)
            # Only add if dimension exists and is > 0
            if dim_value is not None and dim_value > 0:
                total_dim += dim_value
    
    return total_dim


def update_config(config: dict)->dict:
    """
    Updates the config file where placeholders are set 

    Args:
        config (dict): config with placeholders

    Returns:
        dict: updated config
    """
    
    assert config.model.model_object in ["proT", "proT_sim", "proT_adaptive", "GRU","LSTM", "TCN","MLP", "S6"], AssertionError("This model is not available in experiment_control!")
    
    # Handle all proT variants (proT, proT_sim, proT_adaptive) with unified logic
    if config.model.model_object in ["proT", "proT_sim", "proT_adaptive"]:
        
        # Encoder dimension calculation
        if config.model.kwargs.comps_embed_enc == "concat":
            config.model.kwargs.d_model_enc = calculate_concat_dim(
                config, "enc", "model.kwargs.ds_embed_enc.modules"
            )
            
        elif config.model.kwargs.comps_embed_enc == "summation":
            # For summation, d_model_set must be defined
            d_model_set = config.model.embed_dim.d_model_set
            if d_model_set is None:
                raise ValueError(
                    "When using summation embeddings (comps_embed_enc='summation'), "
                    "you must set 'model.embed_dim.d_model_set' to a numeric value in your config file."
                )
            # Set model dimension
            config.model.kwargs.d_model_enc = d_model_set
            # Update all encoder embedding dimensions to match d_model_set for summation
            encoder_embedding_keys = [
                "enc_pro_emb_hidden",
                "enc_occ_emb_hidden",
                "enc_step_emb_hidden",
                "enc_val_emb_hidden",
                "enc_var_emb_hidden",
                "enc_pos_emb_hidden",
                "enc_time_emb_hidden"
            ]
            for key in encoder_embedding_keys:
                current_value = OmegaConf.select(config, f"model.embed_dim.{key}")
                if current_value is not None and current_value > 0:
                    OmegaConf.update(config, f"model.embed_dim.{key}", d_model_set)
            
        elif config.model.kwargs.comps_embed_enc == "spatiotemporal":
            config.model.kwargs.d_model_enc = config.model.embed_dim.d_model_set
        
        # Decoder dimension calculation    
        if config.model.kwargs.comps_embed_dec == "concat":
            config.model.kwargs.d_model_dec = calculate_concat_dim(
                config, "dec", "model.kwargs.ds_embed_dec.modules"
            )
            
        elif config.model.kwargs.comps_embed_dec == "summation":
            # For summation, d_model_set must be defined
            d_model_set = config.model.embed_dim.d_model_set
            if d_model_set is None:
                raise ValueError(
                    "When using summation embeddings (comps_embed_dec='summation'), "
                    "you must set 'model.embed_dim.d_model_set' to a numeric value in your config file."
                )
            # Set model dimension
            config.model.kwargs.d_model_dec = d_model_set
            # Update all decoder embedding dimensions to match d_model_set for summation
            decoder_embedding_keys = [
                "dec_val_emb_hidden",
                "dec_var_emb_hidden",
                "dec_pos_emb_hidden",
                "dec_time_emb_hidden"
            ]
            for key in decoder_embedding_keys:
                current_value = OmegaConf.select(config, f"model.embed_dim.{key}")
                if current_value is not None and current_value > 0:
                    OmegaConf.update(config, f"model.embed_dim.{key}", d_model_set)
            
        elif config.model.kwargs.comps_embed_dec == "spatiotemporal":
            config.model.kwargs.d_model_dec = config.model.embed_dim.d_model_set
            
    # Handle other model types
    elif config.model.model_object in ["GRU","LSTM", "TCN","MLP", "S6"]:
        D_in = len(config.model.kwargs.ds_embed_in.modules)
        D_trg = len(config.model.kwargs.ds_embed_trg.modules)
        
        # Check composition mode
        comps_embed = config.model.kwargs.get("comps_embed", "concat")
        
        if comps_embed == "summation":
            # Summation: all embeddings sum to d_model_set dimension
            config.model.kwargs.d_in = config.experiment.d_model_set
            config.model.kwargs.d_emb = config.experiment.d_model_set
        else:
            # Concatenation (default): embeddings are concatenated
            config.model.kwargs.d_in = D_in * config.experiment.d_model_set
            config.model.kwargs.d_emb = D_trg * config.experiment.d_model_set
    return config
