from omegaconf import OmegaConf
import shutil, subprocess, pathlib, yaml
from pathlib import Path
import optuna, time, subprocess, os
from os.path import dirname, abspath, join
import sys
import re
from functools import partial
from typing import Callable
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)
from prochain_transformer.experiment_control import update_config
from prochain_transformer.trainer import trainer




# search space
def MLP_sample_params(trial):
    """Return a flat dict param_name -> sampled value."""
    return {
        "training.lr"   : trial.suggest_float("lr", 3e-4, 3e-3, log=True),
    }
    
    
def RNN_sample_params(trial):
    """Return a flat dict param_name -> sampled value."""
    return {
        "training.lr"   : trial.suggest_float("lr", 3e-4, 3e-3, log=True),
    }

def TCN_sample_params(trial):
    """Return a flat dict param_name -> sampled value."""
    return {
        "training.lr"   : trial.suggest_float("lr", 3e-4, 3e-3, log=True),
    }
    
def proT_sample_params(trial):
    """Return a flat dict param_name -> sampled value."""
    return {
        "training.lr"   : trial.suggest_float("lr", 3e-4, 3e-3, log=True),
    }

# helper
def get_config_run(base_config, exp_path, params, trial_id):
    """
    Loads base config, applies param overrides, writes to exp_path/run_<id>/config.yaml.
    Returns the path to the new file and save directory for the trainer
    """
    # create hard-copy of config
    config_ = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    
    # set sweep values in new config
    for dotted_key, val in params.items():
        OmegaConf.update(config_, dotted_key, val, merge=True)
    
    # update new config
    config_updated = update_config(config_)

    # create a directory where to save the current run
    save_dir = Path(join(exp_path,"optuna",f"run_{trial_id}")) 
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # save new config in the directory
    config_path = save_dir / "config.yaml"
    OmegaConf.save(config_updated, config_path)
    
    return config_path, save_dir



def objective_extended(
    trial, 
    sample_params: Callable, 
    config: dict, 
    exp_path: Path, 
    data_dir: Path, 
    experiment_tag: str, 
    cluster: bool):
    
    params   = sample_params(trial)
    
    config_path, save_dir = get_config_run(config, exp_path, params, trial.number)

    config_updated = OmegaConf.load(config_path)
    
    df_metric = trainer(
        config = config_updated,
        data_dir = data_dir, 
        save_dir = save_dir,
        experiment_tag = experiment_tag, 
        cluster = cluster, 
        resume_ckpt = None,
        plot_pred_check = False,
        debug = False)

    # save metrics in trial user attributes
    trial.set_user_attr("config_path", str(config_path))
    trial.set_user_attr("val_mae_mean", df_metric["val_mae"].mean())
    trial.set_user_attr("val_r2_mean", df_metric["val_r2"].mean())
    trial.set_user_attr("val_rmse_mean", df_metric["val_rmse"].mean())
    trial.set_user_attr("val_mae_std", df_metric["val_mae"].std())
    trial.set_user_attr("val_r2_std", df_metric["val_r2"].std())
    trial.set_user_attr("val_rmse_std", df_metric["val_rmse"].std())
    trial.set_user_attr("test_mae_mean", df_metric["test_mae"].mean())
    trial.set_user_attr("test_r2_mean", df_metric["test_r2"].mean())
    trial.set_user_attr("test_rmse_mean", df_metric["test_rmse"].mean())
    trial.set_user_attr("test_mae_std", df_metric["test_mae"].std())
    trial.set_user_attr("test_r2_std", df_metric["test_r2"].std())
    trial.set_user_attr("test_rmse_std", df_metric["test_rmse"].std())
    
    # define score as the mean of the validation MAE
    score = df_metric["val_mae"].mean()
    
    return score           



def optuna_study(
    exp_dir: Path,
    data_dir: Path,
    cluster: bool,
    study_name: str,
    manifest_tag: str,
    ):
    
    # look for config file
    pattern = re.compile(r'config_.*\.yaml')
    matching_files = [file for file in os.listdir(exp_dir) if pattern.match(file)]
    
    if len(matching_files) == 0:
        raise ValueError(f"No config file found in {exp_dir}")
    elif len(matching_files) > 1:
        raise ValueError(f"More than one config file found in {exp_dir}")
    else:
        config_file_name = matching_files[0]
    
    # load base config
    config = OmegaConf.load(join(exp_dir,config_file_name))
    
    if config["model"]["model_object"] == "MLP":
        sample_params = MLP_sample_params
    elif config["model"]["model_object"] in ["LSTM", "GRU"]:
        sample_params = RNN_sample_params
    elif config["model"]["model_object"] == "TCN":
        sample_params = TCN_sample_params
    elif config["model"]["model_object"] == "proT":
        sample_params = proT_sample_params
    else:
        raise ValueError("Invalid model_object, no sampling function available!")
        
    
    obj_kwargs = {
        "sample_params" : sample_params,
        "config"        : config,
        "exp_path"      : exp_dir,
        "data_dir"      : data_dir,
        "experiment_tag": manifest_tag,
        "cluster"       : cluster
    }
    
    objective = partial(objective_extended, **obj_kwargs)
    study_path = Path(exp_dir) / "optuna"
    study_path.mkdir(parents=True, exist_ok=True)
    study_file_path = study_path / "study.db"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.QMCSampler(qmc_type="sobol"),  # Sobol inside Optuna
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        storage=f"sqlite:///{study_file_path}",
    )
    # run the optimization
    study.optimize(objective, n_trials=40, n_jobs=4)
    
    # save study
    best = study.best_trial
    to_dump = {
        "params":       best.params,
        "config_path":  best.user_attrs["config_path"],
        "val_mae_mean": best.user_attrs["val_mae_mean"],
        "val_mae_std":  best.user_attrs["val_mae_std"],
        "val_r2_mean":  best.user_attrs["val_r2_mean"], 
        "val_r2_std":   best.user_attrs["val_r2_std"],
        "val_rmse_mean": best.user_attrs["val_rmse_mean"],
        "val_rmse_std":  best.user_attrs["val_rmse_std"],
        "test_mae_mean": best.user_attrs["test_mae_mean"],
        "test_mae_std":  best.user_attrs["test_mae_std"],
        "test_r2_mean":  best.user_attrs["test_r2_mean"],
        "test_r2_std":   best.user_attrs["test_r2_std"],
        "test_rmse_mean": best.user_attrs["test_rmse_mean"],
        "test_rmse_std":  best.user_attrs["test_rmse_std"],
        }
    
    with open(Path(exp_dir) / "best_trial.yaml" , 'w') as yaml_file:
        yaml.dump(to_dump, yaml_file, default_flow_style=False)
        
    
    
if __name__ == "__main__":
    
    
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    exp_dir = join(ROOT_DIR, "experiments/training/test_MLP_ishigami")
    data_dir = join(ROOT_DIR, "data/input/") # input data, do not touch
    
    optuna_study(
        exp_dir=exp_dir,
        data_dir=data_dir,
        cluster=False,
        study_name="MLP_baseline",
        manifest_tag = "baseline"
    )
