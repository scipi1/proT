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





def combination_sweep(exp_dir, mode="combination"):
    """
    This function scans for configuration files in a folder and runs accordingly
    to the specified sweep mode:
    
    - "independent": One-at-a-time parameter sweep (like independent_sweep)
    - "combination": Sweep across all possible combinations of parameter values
    
    This function has to be used as decorator on the function "func"
    func (function)-> None: callable function with input (config, save_dir)
    
    It also creates the folder structure for sweeps and passes it to func.
    
    For independent mode, the folder structure looks like:
    
    ./experiments
    |__exp1
    |__exp2 <-- experiment folder "exp_dir"
        |__config.yaml <-- starting config file
        |__config_sweep.yaml <-- sweep instructions
        |__sweeps
            |__sweep_param1
            |__sweep_param2
            |__sweep_param3
                |__sweep_param3_val1
                |__sweep_param3_val2
                    |__config.yaml <-- sweep (modified) config file
                    |__results
    
    For combination mode, the folder structure looks like:
    
    ./experiments
    |__exp1
    |__exp2 <-- experiment folder "exp_dir"
        |__config.yaml <-- starting config file
        |__config_sweep.yaml <-- sweep instructions
        |__combinations
            |__combo_param1_val1_param2_val1
            |__combo_param1_val1_param2_val2
                |__config.yaml <-- sweep (modified) config file
                |__results
                    
    Args:
        exp_dir (str): experiment folder
        mode (str): sweep mode, either "independent" or "combination"
    """
    
    config, sweep_config = find_yml_files(dir=exp_dir)
    # set logging
    logger_info = logging.getLogger("logger_info")
    
    def decorator_repeat(func):
        def wrapper(*args, **kwargs):
        
            if sweep_config is not None:
                if mode == "independent":
                    # Independent sweep logic (like independent_sweep)
                    for cat in sweep_config:
                        for param in sweep_config[cat]:

                            sweep_param_path = join(exp_dir, "sweeps", f"sweep_{param}")
                            if not exists(sweep_param_path):
                                mkdir(sweep_param_path)

                            assert param in config[cat], AssertionError(f"Parameters {param} not in config file!")
                            for val in sweep_config[cat][param]:
                                
                                # logging
                                logger_info.info(f"Sweep at {cat}-{param}: {val}")
                                
                                # update the config file for the current sweep
                                config_sweep = config.copy()
                                config_sweep[cat][param] = val
                                
                                # update the saving dir
                                save_dir = join(sweep_param_path, f"sweep_{param}_{val}")
                                if not exists(save_dir):
                                    mkdir(save_dir)

                                # execute function
                                func(*args, **kwargs, config=config_sweep, save_dir=save_dir)

                                # save config_sweep as yaml file
                                config_sweep_path = join(save_dir, "config.yaml")
                                OmegaConf.save(config_sweep, config_sweep_path)
                
                elif mode == "combination":
                    # Combination sweep logic
                    
                    # Extract all parameters and their values
                    param_values = {}
                    param_categories = {}
                    
                    for cat in sweep_config:
                        for param in sweep_config[cat]:
                            assert param in config[cat], AssertionError(f"Parameters {param} not in config file!")
                            param_values[param] = sweep_config[cat][param]
                            param_categories[param] = cat
                    
                    # Generate all possible combinations
                    param_names = list(param_values.keys())
                    value_combinations = list(itertools.product(*(param_values[param] for param in param_names)))
                    
                    # Create combinations directory
                    combinations_path = join(exp_dir, "combinations")
                    if not exists(combinations_path):
                        mkdir(combinations_path)
                    
                    # Process each combination
                    for combo in value_combinations:
                        # Create a descriptive name for this combination
                        combo_name_parts = []
                        
                        # Update config with this combination
                        config_sweep = config.copy()
                        
                        # Log the combination
                        combo_log_parts = []
                        
                        for i, param in enumerate(param_names):
                            val = combo[i]
                            cat = param_categories[param]
                            
                            # update the config file for the current sweep
                            config_sweep[cat][param] = val
                            config_sweep = update_config(config_sweep)
                            
                            # Add to combo name
                            combo_name_parts.append(f"{param}_{val}")
                            
                            # Add to log
                            combo_log_parts.append(f"{cat}-{param}: {val}")
                        
                        # Log the combination
                        logger_info.info(f"Combination sweep with " + ", ".join(combo_log_parts))
                        
                        # Create directory for this combination
                        combo_name = "combo_" + "_".join(combo_name_parts)
                        combo_dir = join(combinations_path, combo_name)
                        if not exists(combo_dir):
                            mkdir(combo_dir)
                        
                        # Execute function with this combination
                        func(*args, **kwargs, config=config_sweep, save_dir=combo_dir)
                        
                        # Save config
                        config_sweep_path = join(combo_dir, "config.yaml")
                        OmegaConf.save(config_sweep, config_sweep_path)
                
                else:
                    raise ValueError(f"Unknown sweep mode: {mode}. Use 'independent' or 'combination'.")
            
            else:
                func(config, save_dir=exp_dir)   
        return wrapper
    return decorator_repeat












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
            if config.model.embed_dim.enc_val_emb_hidden is not None and config.model.embed_dim.enc_val_emb_hidden > 0:
                config.model.embed_dim.enc_val_emb_hidden = d_model_set
            if config.model.embed_dim.enc_var_emb_hidden is not None and config.model.embed_dim.enc_var_emb_hidden > 0:
                config.model.embed_dim.enc_var_emb_hidden = d_model_set
            if config.model.embed_dim.enc_pos_emb_hidden is not None and config.model.embed_dim.enc_pos_emb_hidden > 0:
                config.model.embed_dim.enc_pos_emb_hidden = d_model_set
            if config.model.embed_dim.enc_time_emb_hidden is not None and config.model.embed_dim.enc_time_emb_hidden > 0:
                config.model.embed_dim.enc_time_emb_hidden = d_model_set
            
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
            if config.model.embed_dim.dec_val_emb_hidden is not None and config.model.embed_dim.dec_val_emb_hidden > 0:
                config.model.embed_dim.dec_val_emb_hidden = d_model_set
            if config.model.embed_dim.dec_var_emb_hidden is not None and config.model.embed_dim.dec_var_emb_hidden > 0:
                config.model.embed_dim.dec_var_emb_hidden = d_model_set
            if config.model.embed_dim.dec_pos_emb_hidden is not None and config.model.embed_dim.dec_pos_emb_hidden > 0:
                config.model.embed_dim.dec_pos_emb_hidden = d_model_set
            if config.model.embed_dim.dec_time_emb_hidden is not None and config.model.embed_dim.dec_time_emb_hidden > 0:
                config.model.embed_dim.dec_time_emb_hidden = d_model_set
            
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
