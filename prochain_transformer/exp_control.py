import yaml
import numpy as np
from os.path import exists,join
from os import makedirs, getcwd, scandir
import fnmatch
from typing import Tuple
import warnings
from omegaconf import OmegaConf
import logging


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
    
    return config, sweep_config





def independent_sweep(exp_dir):
    """
    This function scans for configuration files in a folder and runs accordingly
    to an independent sweep
    - func(config) is no sweeps
    - func(config(sweep)) for sweep in sweeps
    
    This function has to be used as decorator on the the function "func"
    func (function)-> None: callable function with input (config, save_dir)
    
    It also creates the folder structure for sweeps and passes it to func.
    Here how it looks like:
    
    ./experiments
    |__exp1
    |__exp2 <-- experiment folder “exp_dir”
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
                    
    Args:
        exp_dir (str): experiment folder
    """
    
    config, sweep_config = find_yml_files(dir=exp_dir)
    # set logging
    logger_info = logging.getLogger("logger_info")
    
    def decorator_repeat(func):
        def wrapper(*args, **kwargs):
        
            if sweep_config is not None:
                
                for cat in sweep_config:
                    for param in sweep_config[cat]:

                        sweep_param_path = join(exp_dir,"sweeps",f"sweep_{param}")

                        assert param in config[cat], AssertionError(f"Parameters {param} not in config file!")
                        for val in sweep_config[cat][param]:
                            
                            # logging
                            logger_info.info(f"Sweep at {cat}-{param}: {val}")
                            
                            # update the config file for the current sweep
                            config_sweep = config.copy()
                            config_sweep[cat][param] = val
                            config_sweep = update_config(config_sweep)
                            
                            # update the saving dir
                            save_dir = join(sweep_param_path,f"sweep_{param}_{val}")

                            # execute function
                            func(*args, **kwargs, config=config_sweep, save_dir=save_dir)

                            # save config_sweep as yaml file
                            config_sweep_path = join(save_dir,"config.yaml")
                            OmegaConf.save(config_sweep, config_sweep_path)
            
            else:
                func(config,save_dir=exp_dir)   
        return wrapper
    return decorator_repeat


# TODO ch
def update_config(config: dict)->dict:
    """
    Updates the config file where placeholders are set 

    Args:
        config (dict): config with placeholders

    Returns:
        dict: updated config
    """
    
    # TODO check if the d_model has a placeholder
    
    
        
    if config.model.d_model_enc == None:
        config.model.d_model_enc = config.model.enc_val_emb_hidden + config.model.enc_var_emb_hidden + config.model.enc_pos_emb_hidden + config.model.enc_time_emb_hidden
        
    if config.model.d_model_dec == None:
        config.model.d_model_dec = config.model.dec_val_emb_hidden + config.model.dec_var_emb_hidden + config.model.dec_pos_emb_hidden + config.model.dec_time_emb_hidden
    
    return config
