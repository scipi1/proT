from omegaconf import OmegaConf
import torch
import yaml
from pathlib import Path
import optuna, os
from optuna.exceptions import DuplicatedStudyError
from optuna.study import MaxTrialsCallback
from os.path import dirname, abspath, join
import sys
import re
from functools import partial
from typing import Callable
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)
from proT.experiment_control import update_config
from proT.trainer import trainer


# search space
def MLP_sample_params(trial):
    """Return a flat dict param_name -> sampled value."""
    
    MLP_ARCHS = {
        "mlp_s-128x3": [128,128,128],
        "mlp_m-256x3": [256,256,256],
        "mlp_m-512x3": [512,512,512],
        #"mlp_l-256x4": [256,256,256,256],
    }
    arch_key = trial.suggest_categorical("mlp_arch", list(MLP_ARCHS.keys()))
    
    return {
        "experiment.hidden_set"     : MLP_ARCHS[arch_key],
        "experiment.d_model_set"    : trial.suggest_int("d_model_set", 64, 512, step=16),
        "experiment.d_hidden_set"   : trial.suggest_int("d_hidden_set", 64, 512, step=16),
        "training.lr"               : trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "experiment.dropout"        : trial.suggest_float("dropout", 0.0, 0.3),
        }
    
    
def RNN_sample_params(trial):
    """Return a flat dict param_name -> sampled value."""
    return {
        "experiment.n_layers_set"   : trial.suggest_int("n_layers_set", 1, 4, step=1),
        "experiment.d_model_set"    : trial.suggest_int("d_model_set", 64, 512, step=16),
        "experiment.d_hidden_set"   : trial.suggest_int("d_hidden_set", 64, 512, step=16),
        "training.lr"               : trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "experiment.dropout"        : trial.suggest_float("dropout", 0.0, 0.3),

    }

def TCN_sample_params(trial):
    """Return a flat dict param_name -> sampled value."""
    
    TCN_ARCHS = {
        "tcn_s-32-32-64"    : [32, 32, 64],
        "tcn_s-64-64-128"   : [64, 64, 128],
        "tcn_m-64-128-256"  : [64, 128, 256],
        "tcn_m-128-128-128" : [128,128,128],
        "tcn_l-128-256-256" : [128,256,256],
    }
    arch_key = trial.suggest_categorical("tcn_arch", list(TCN_ARCHS.keys()))
    
    return {
        "experiment.channels_set"   : TCN_ARCHS[arch_key],
        "experiment.d_model_set"    : trial.suggest_int("d_model_set", 64, 512, step=16),
        "training.lr"               : trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "experiment.dropout"        : trial.suggest_float("dropout", 0.0, 0.3),
    }
    
def proT_sample_params(trial):
    """Return a flat dict param_name -> sampled value."""
    return {
        "model.embed_dim.enc_val_emb_hidden"    : trial.suggest_int("enc_val_emb_hidden", 50, 200, step=10),
        "model.embed_dim.enc_var_emb_hidden"    : trial.suggest_int("enc_var_emb_hidden", 50, 200, step=10),
        "model.embed_dim.enc_pos_emb_hidden"    : trial.suggest_int("enc_pos_emb_hidden", 50, 200, step=10),
        "model.embed_dim.enc_time_emb_hidden"   : trial.suggest_int("enc_time_emb_hidden", 10, 100, step=10),
        "model.embed_dim.dec_val_emb_hidden"    : trial.suggest_int("dec_val_emb_hidden", 50, 200, step=10),
        "model.embed_dim.dec_var_emb_hidden"    : trial.suggest_int("dec_var_emb_hidden", 50, 200, step=10),
        "model.embed_dim.dec_pos_emb_hidden"    : trial.suggest_int("dec_pos_emb_hidden", 50, 200, step=10),
        "model.embed_dim.dec_time_emb_hidden"   : trial.suggest_int("dec_time_emb_hidden", 10, 100, step=10),
        "model.kwargs.n_heads"                  : trial.suggest_int("n_heads", 1, 3, step=1),
        "model.kwargs.d_ff"                     : trial.suggest_int("d_ff", 200, 600, step=100),
        "model.kwargs.d_qk"                     : trial.suggest_int("d_qk", 100, 200, step=50),
        "model.kwargs.dropout_emb"                  : trial.suggest_float("dropout_emb", 0.0, 0.3, step=0.1),
        "model.kwargs.dropout_data"                 : trial.suggest_float("dropout_data", 0.0, 0.3, step=0.1),
        "model.kwargs.dropout_attn_out"             : trial.suggest_float("dropout_attn_out", 0.0, 0.3, step=0.1),
        "model.kwargs.dropout_ff"                   : trial.suggest_float("dropout_ff", 0.0, 0.3, step=0.1),
        "model.kwargs.enc_dropout_qkv"              : trial.suggest_float("enc_dropout_qkv", 0.0, 0.3, step=0.1),
        "model.kwargs.enc_attention_dropout"        : trial.suggest_float("enc_attention_dropout", 0.0, 0.3, step=0.1),
        "model.kwargs.dec_self_dropout_qkv"         : trial.suggest_float("dec_self_dropout_qkv", 0.0, 0.3, step=0.1),
        "model.kwargs.dec_self_attention_dropout"   : trial.suggest_float("dec_self_attention_dropout", 0.0, 0.3, step=0.1),
        "model.kwargs.dec_cross_dropout_qkv"        : trial.suggest_float("dec_cross_dropout_qkv", 0.0, 0.3, step=0.1),
        "model.kwargs.dec_cross_attention_dropout"  : trial.suggest_float("dec_cross_attention_dropout", 0.0, 0.3, step=0.1),
        "training.lr"   : trial.suggest_float("lr", 1e-4, 1e-3, step=1e-4),
    }

# helper
def get_config_run(base_config, exp_path, params, trial_id):
    """
    Loads base config, applies param overrides, writes to exp_path/run_<id>/config.yaml.
    Returns the path to the new file and save directory for the trainer
    """
    # create hard-copy of config
    config_ = base_config.copy() #OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    
    # set sweep values in new config
    for dotted_key, val in params.items():
        OmegaConf.update(config_, dotted_key, val, merge=True)

    # create a directory where to save the current run
    save_dir = Path(join(exp_path,"optuna",f"run_{trial_id}")) 
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # save new config in the directory
    config_path = save_dir / "config.yaml"
    OmegaConf.save(config_, config_path)
    
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
    
    # load and update run config
    config_run = OmegaConf.load(config_path)
    config_updated = update_config(config_run)
    
    
    if cluster and torch.cuda.is_available():
        torch.cuda.set_device(0)      # first GPU in the mask
    
    df_metric = trainer(
        config = config_updated,
        data_dir = data_dir, 
        save_dir = save_dir,
        experiment_tag = experiment_tag, 
        cluster = cluster, 
        resume_ckpt = None,
        plot_pred_check = False,
        debug = False)
        
    # Define the metrics and their corresponding operations
    metrics = {
        "val_mae": ["mean", "std"],
        "val_r2": ["mean", "std"],
        "val_rmse": ["mean", "std"],
        "test_mae": ["mean", "std"],
        "test_r2": ["mean", "std"],
        "test_rmse": ["mean", "std"]
    }
    
    # Iterate over the metrics and set the attributes
    for metric, operations in metrics.items():
        for operation in operations:
            if operation == "mean":
                trial.set_user_attr(f"{metric}_{operation}", df_metric[metric].mean())
            elif operation == "std":
                trial.set_user_attr(f"{metric}_{operation}", df_metric[metric].std())
    
    # Set the config path
    trial.set_user_attr("config_path", str(config_path))
    
    # define score as the mean of the validation MAE
    score = df_metric["val_mae"].mean()
    
    return score           


class OptunaStudy:
    def __init__(
        self,
        exp_dir: Path,
        data_dir: Path,
        cluster: bool,
        study_name: str,
        manifest_tag: str,
        study_path: str = None,
        ):
        self.exp_dir = exp_dir
        self.data_dir = data_dir
        self.cluster = cluster
        self.study_name = study_name
        self.manifest_tag = manifest_tag
        
        # look for config file
        pattern_config = re.compile(r'config_.*\.yaml')
        config_matching_files = [file for file in os.listdir(exp_dir) if pattern_config.match(file)]
        if len(config_matching_files) == 1:
            config = OmegaConf.load(join(exp_dir,config_matching_files[0]))
        else:
            raise ValueError(f"None or more than one config file found in {exp_dir}")

        # look for optuna settings file
        pattern_optuna = re.compile(r'optuna.*\.yaml')
        optuna_matching_files = [file for file in os.listdir(exp_dir) if pattern_optuna.match(file)]
        if len(optuna_matching_files) == 1:
            self.optuna_settings = OmegaConf.load(join(exp_dir,optuna_matching_files[0]))
        else:
            self.optuna_settings = None

        # select sampling parameters based on model_object
        params_select = {
            "MLP":MLP_sample_params,
            "LSTM":RNN_sample_params,
            "GRU":RNN_sample_params,
            "TCN":TCN_sample_params,
            "proT":proT_sample_params
            }
        try:
            sample_params = params_select[config["model"]["model_object"]]
        except KeyError:
            raise ValueError(f"Invalid model_object: {config['model']['model_object']}. No sampling function available!")
        
        obj_kwargs = {
        "sample_params" : sample_params,
        "config"        : config,
        "exp_path"      : exp_dir,
        "data_dir"      : data_dir,
        "experiment_tag": manifest_tag,
        "cluster"       : cluster
        }

        # define the objective function
        self.objective = partial(objective_extended, **obj_kwargs)
    
        # define the storage path (local for now)
        if study_path is None:
            study_path = (Path(exp_dir) / "optuna").resolve()
            study_path.mkdir(parents=True, exist_ok=True)
        
        self.study_file_path = join(study_path,"study.db")
        
        # set the storage and other parameters
        self.storage = f"sqlite:///{self.study_file_path}?timeout=60" 
        self.max_trials = self._setting("n_trials", 50)
        self.direction = self._setting("direction", "minimize")
        self.pruner = self._setting("pruner", "none")
        
    def create(self):
        try:            
            study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                sampler=self._build_sampler(),
                pruner=None if self._setting("pruner", "none") == "none" else optuna.pruners.MedianPruner(n_warmup_steps=5),
                storage=self.storage,
                )
            print(f"Created study {self.study_name} stored at {self.storage}.")
            
        except DuplicatedStudyError:
            print(f"Study {self.study_name} already exists — switch to --mode resume")
            return
        
        
    def resume(self, storage=None):
        
        stor = storage if storage is not None else self.storage
        study = optuna.load_study(study_name=self.study_name, storage=stor)
        study.optimize(
            self.objective,
            n_trials=None,          # loop until no trials remain
            callbacks=[MaxTrialsCallback(self.max_trials, states=None)],
            gc_after_trial=True,
            catch=(RuntimeError,),
            )
        
        
    def summary(self):
        study = optuna.load_study(study_name=self.study_name, storage=self.storage)
        best = study.best_trial
    
        # Define the keys for the metrics
        metric_keys = [
            "config_path",
            "val_mae_mean", "val_mae_std",
            "val_r2_mean", "val_r2_std",
            "val_rmse_mean", "val_rmse_std",
            "test_mae_mean", "test_mae_std",
            "test_r2_mean", "test_r2_std",
            "test_rmse_mean", "test_rmse_std"
        ]
        # Create the to_dump dictionary using dictionary comprehension
        to_dump = {key: best.user_attrs.get(key) for key in metric_keys}
        to_dump["params"] = best.params

        with open(Path(self.exp_dir) / "best_trial.yaml" , 'w') as yaml_file:
            yaml.dump(to_dump, yaml_file, default_flow_style=False)
    
    
    def _setting(self, key, default):
        return (self.optuna_settings or {}).get(key, default)
    
    
    def _build_sampler(self):
        """Return an Optuna sampler according to self.optuna_settings['sampler']."""
        if self.optuna_settings is None or "sampler" not in self.optuna_settings:
            return optuna.samplers.QMCSampler(qmc_type="sobol")
    
        cfg = dict(self.optuna_settings["sampler"])   # OmegaConf → plain dict
        name = cfg.pop("name", "sobol").lower()
    
        if name == "sobol":
            return optuna.samplers.QMCSampler(qmc_type="sobol", **cfg)
        if name == "tpe":
            return optuna.samplers.TPESampler(**cfg) # use for multivariate/categorical
        raise ValueError(f"Unknown sampler name: {name}")
