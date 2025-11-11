"""
CLI for proT Hyperparameter Optimization

This module provides a command-line interface for running Optuna hyperparameter
optimization studies with proT models.

The implementation includes:
- Model-specific sampling functions (MLP, RNN, TCN, proT)
- Integration with proT's trainer
- Metrics extraction from k-fold cross-validation results

USAGE EXAMPLES

Basic Workflow (Local)
----------------------
# 1. Create a new study
python -m proT.euler_optuna.cli paramsopt
    --exp_id baseline_LSTM_ishigami_cat
    --study_name my_lstm_study
    --mode create

# 2. Resume optimization (run multiple trials)
python -m proT.euler_optuna.cli paramsopt
    --exp_id baseline_LSTM_ishigami_cat
    --study_name my_lstm_study
    --mode resume

# 3. View best results
python -m proT.euler_optuna.cli paramsopt
    --exp_id baseline_LSTM_ishigami_cat
    --study_name my_lstm_study
    --mode summary

Notes
-----
- Experiment folder must exist at: experiments/training/<exp_id>/
- Config file must be in experiment folder (e.g., config_*.yaml)
- Study database stored in: <exp_id>/optuna/<study_name>.db
- Can run multiple workers in parallel for faster optimization
- Use --cluster flag when running on compute cluster
- See SAMPLING_BOUNDS_DOCUMENTATION.md for parameter details
"""

# Standard library imports
import sys
import os
from os.path import abspath, join, exists, dirname
from pathlib import Path
import re

# Third-party imports
import click
from omegaconf import OmegaConf

# =============================================================================
# SETUP: Add proT to path
# =============================================================================
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)

# proT-specific imports
from proT.training.trainer import trainer
from proT.training.experiment_control import update_config
from proT.euler_optuna.optuna_opt import OptunaStudy


# =============================================================================
# HYPERPARAMETER SAMPLING CONFIGURATION
# =============================================================================

BASELINE_SAMPLING_BOUNDS = {
    # Common parameters (shared across multiple models)
    "d_model_set": {"low": 64, "high": 512, "step": 16},
    "d_hidden_set": {"low": 64, "high": 512, "step": 16},
    "dropout": {"low": 0.0, "high": 0.3},
    "lr": {"low": 1e-4, "high": 1e-3, "log": True},
    "lr_stepped": {"low": 1e-4, "high": 1e-3, "step": 1e-4},  # For proT
    
    # RNN-specific
    "n_layers": {"low": 1, "high": 4, "step": 1},
    
    # proT embedding dimensions
    "embedding_dim_standard": {"low": 50, "high": 200, "step": 10},
    "embedding_dim_time": {"low": 10, "high": 100, "step": 10},
    "embedding_dim_adaptive": {"low": 30, "high": 100, "step": 10},
    
    # proT architecture parameters
    "n_heads": {"low": 1, "high": 3, "step": 1},
    "d_ff": {"low": 200, "high": 600, "step": 100},
    "d_qk": {"low": 100, "high": 200, "step": 50},
    
    # proT dropout parameters (fine-grained control)
    "dropout_fine": {"low": 0.0, "high": 0.3, "step": 0.1},
    
    # Entropy regularization parameter
    "gamma": {"low": 1e-4, "high": 1e-2, "log": True},
}

# Dictionary mapping profile names to their configurations
SAMPLING_PROFILES = {
    "baseline": BASELINE_SAMPLING_BOUNDS,
    # Future profiles can be added here, e.g.:
    # "extended": EXTENDED_SAMPLING_BOUNDS,
    # "narrow": NARROW_SAMPLING_BOUNDS,
}

# Global variable that will be set based on CLI flag (defaults to baseline)
SAMPLING_BOUNDS = BASELINE_SAMPLING_BOUNDS


# =============================================================================
# MODEL-SPECIFIC SAMPLING FUNCTIONS
# =============================================================================

def MLP_sample_params(trial):
    """Return a flat dict param_name -> sampled value."""
    
    MLP_ARCHS = {
        "mlp_s-128x3": [128,128,128],
        "mlp_m-256x3": [256,256,256],
        "mlp_m-512x3": [512,512,512],
    }
    arch_key = trial.suggest_categorical("mlp_arch", list(MLP_ARCHS.keys()))
    
    return {
        "experiment.hidden_set"     : MLP_ARCHS[arch_key],
        "experiment.d_model_set"    : trial.suggest_int("d_model_set", **SAMPLING_BOUNDS["d_model_set"]),
        "experiment.d_hidden_set"   : trial.suggest_int("d_hidden_set", **SAMPLING_BOUNDS["d_hidden_set"]),
        "training.lr"               : trial.suggest_float("lr", **SAMPLING_BOUNDS["lr"]),
        "experiment.dropout"        : trial.suggest_float("dropout", **SAMPLING_BOUNDS["dropout"]),
    }


def RNN_sample_params(trial):
    """Return a flat dict param_name -> sampled value."""
    return {
        "experiment.n_layers_set"   : trial.suggest_int("n_layers_set", **SAMPLING_BOUNDS["n_layers"]),
        "experiment.d_model_set"    : trial.suggest_int("d_model_set", **SAMPLING_BOUNDS["d_model_set"]),
        "experiment.d_hidden_set"   : trial.suggest_int("d_hidden_set", **SAMPLING_BOUNDS["d_hidden_set"]),
        "training.lr"               : trial.suggest_float("lr", **SAMPLING_BOUNDS["lr"]),
        "experiment.dropout"        : trial.suggest_float("dropout", **SAMPLING_BOUNDS["dropout"]),
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
        "experiment.d_model_set"    : trial.suggest_int("d_model_set", **SAMPLING_BOUNDS["d_model_set"]),
        "training.lr"               : trial.suggest_float("lr", **SAMPLING_BOUNDS["lr"]),
        "experiment.dropout"        : trial.suggest_float("dropout", **SAMPLING_BOUNDS["dropout"]),
    }


def proT_sample_params(trial, config):
    """
    Return a flat dict param_name -> sampled value.
    
    Supports two modes:
    - BENCHMARK MODE: Single dimension for all embeddings
      (enabled by use_uniform_embedding_dims=True OR summation composition)
    - RESEARCH MODE: Independent dimensions per embedding
      (only when use_uniform_embedding_dims=False AND concat composition)
    """
    use_uniform = config.get("experiment", {}).get("use_uniform_embedding_dims", False)
    comps_embed_enc = config.get("model", {}).get("kwargs", {}).get("comps_embed_enc", "concat")
    comps_embed_dec = config.get("model", {}).get("kwargs", {}).get("comps_embed_dec", "concat")
    
    # Force uniform dimensions if summation is used (architectural requirement)
    if comps_embed_enc == "summation" or comps_embed_dec == "summation":
        use_uniform = True
    
    params = {}
    
    if use_uniform:
        # BENCHMARK MODE: Single dimension for all embeddings
        d_model = trial.suggest_int("d_model_set", **SAMPLING_BOUNDS["embedding_dim_standard"])
        params.update({
            "model.embed_dim.enc_val_emb_hidden": d_model,
            "model.embed_dim.enc_var_emb_hidden": d_model,
            "model.embed_dim.enc_pos_emb_hidden": d_model,
            "model.embed_dim.enc_time_emb_hidden": d_model,
            "model.embed_dim.dec_val_emb_hidden": d_model,
            "model.embed_dim.dec_var_emb_hidden": d_model,
            "model.embed_dim.dec_pos_emb_hidden": d_model,
            "model.embed_dim.dec_time_emb_hidden": d_model,
        })
    else:
        # RESEARCH MODE: Independent dimensions per embedding
        params.update({
            "model.embed_dim.enc_val_emb_hidden": trial.suggest_int("enc_val_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_standard"]),
            "model.embed_dim.enc_var_emb_hidden": trial.suggest_int("enc_var_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_standard"]),
            "model.embed_dim.enc_pos_emb_hidden": trial.suggest_int("enc_pos_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_standard"]),
            "model.embed_dim.enc_time_emb_hidden": trial.suggest_int("enc_time_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_time"]),
            "model.embed_dim.dec_val_emb_hidden": trial.suggest_int("dec_val_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_standard"]),
            "model.embed_dim.dec_var_emb_hidden": trial.suggest_int("dec_var_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_standard"]),
            "model.embed_dim.dec_pos_emb_hidden": trial.suggest_int("dec_pos_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_standard"]),
            "model.embed_dim.dec_time_emb_hidden": trial.suggest_int("dec_time_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_time"]),
        })
    
    # Other hyperparameters (same for both modes)
    params.update({
        "model.kwargs.n_heads": trial.suggest_int("n_heads", **SAMPLING_BOUNDS["n_heads"]),
        "model.kwargs.d_ff": trial.suggest_int("d_ff", **SAMPLING_BOUNDS["d_ff"]),
        "model.kwargs.d_qk": trial.suggest_int("d_qk", **SAMPLING_BOUNDS["d_qk"]),
        "model.kwargs.dropout_emb": trial.suggest_float("dropout_emb", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dropout_data": trial.suggest_float("dropout_data", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dropout_attn_out": trial.suggest_float("dropout_attn_out", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dropout_ff": trial.suggest_float("dropout_ff", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.enc_dropout_qkv": trial.suggest_float("enc_dropout_qkv", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.enc_attention_dropout": trial.suggest_float("enc_attention_dropout", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dec_self_dropout_qkv": trial.suggest_float("dec_self_dropout_qkv", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dec_self_attention_dropout": trial.suggest_float("dec_self_attention_dropout", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dec_cross_dropout_qkv": trial.suggest_float("dec_cross_dropout_qkv", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dec_cross_attention_dropout": trial.suggest_float("dec_cross_attention_dropout", **SAMPLING_BOUNDS["dropout_fine"]),
        "training.lr": trial.suggest_float("lr", **SAMPLING_BOUNDS["lr_stepped"]),
        "training.gamma": trial.suggest_float("gamma", **SAMPLING_BOUNDS["gamma"]),
    })
    
    return params


def proT_adaptive_sample_params(trial, config):
    """
    Return a flat dict param_name -> sampled value for proT_adaptive variant.
    
    Key difference from standard proT:
    - Has additional dec_val_given_emb_hidden for adaptive target embedding
    """
    return {
        "model.embed_dim.enc_val_emb_hidden"        : trial.suggest_int("enc_val_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_standard"]),
        "model.embed_dim.enc_var_emb_hidden"        : trial.suggest_int("enc_var_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_standard"]),
        "model.embed_dim.enc_pos_emb_hidden"        : trial.suggest_int("enc_pos_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_standard"]),
        "model.embed_dim.enc_time_emb_hidden"       : trial.suggest_int("enc_time_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_time"]),
        "model.embed_dim.dec_val_emb_hidden"        : trial.suggest_int("dec_val_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_standard"]),
        "model.embed_dim.dec_var_emb_hidden"        : trial.suggest_int("dec_var_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_standard"]),
        "model.embed_dim.dec_pos_emb_hidden"        : trial.suggest_int("dec_pos_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_standard"]),
        "model.embed_dim.dec_time_emb_hidden"       : trial.suggest_int("dec_time_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_time"]),
        "model.embed_dim.dec_val_given_emb_hidden"  : trial.suggest_int("dec_val_given_emb_hidden", **SAMPLING_BOUNDS["embedding_dim_adaptive"]),
        "model.kwargs.n_heads"                      : trial.suggest_int("n_heads", **SAMPLING_BOUNDS["n_heads"]),
        "model.kwargs.d_ff"                         : trial.suggest_int("d_ff", **SAMPLING_BOUNDS["d_ff"]),
        "model.kwargs.d_qk"                         : trial.suggest_int("d_qk", **SAMPLING_BOUNDS["d_qk"]),
        "model.kwargs.dropout_emb"                  : trial.suggest_float("dropout_emb", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dropout_data"                 : trial.suggest_float("dropout_data", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dropout_attn_out"             : trial.suggest_float("dropout_attn_out", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dropout_ff"                   : trial.suggest_float("dropout_ff", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.enc_dropout_qkv"              : trial.suggest_float("enc_dropout_qkv", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.enc_attention_dropout"        : trial.suggest_float("enc_attention_dropout", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dec_self_dropout_qkv"         : trial.suggest_float("dec_self_dropout_qkv", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dec_self_attention_dropout"   : trial.suggest_float("dec_self_attention_dropout", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dec_cross_dropout_qkv"        : trial.suggest_float("dec_cross_dropout_qkv", **SAMPLING_BOUNDS["dropout_fine"]),
        "model.kwargs.dec_cross_attention_dropout"  : trial.suggest_float("dec_cross_attention_dropout", **SAMPLING_BOUNDS["dropout_fine"]),
        "training.lr"                               : trial.suggest_float("lr", **SAMPLING_BOUNDS["lr_stepped"]),
        "training.gamma"                            : trial.suggest_float("gamma", **SAMPLING_BOUNDS["gamma"]),
    }


# =============================================================================
# DISPATCHER: Select sampling function based on model type
# =============================================================================

def sample_params_for_optuna(trial, config):
    """
    Dispatch to model-specific sampling function based on config.
    
    Args:
        trial: Optuna trial object
        config: OmegaConf configuration
        
    Returns:
        Dict mapping dotted config keys to sampled values
    """
    model_obj = config["model"]["model_object"]
    
    # proT models need config passed for uniform embedding dimension support
    if model_obj in ["proT", "proT_sim"]:
        return proT_sample_params(trial, config)
    elif model_obj == "proT_adaptive":
        return proT_adaptive_sample_params(trial, config)
    
    # Other models use simple trial-only sampling
    params_select = {
        "MLP": MLP_sample_params,
        "LSTM": RNN_sample_params,
        "GRU": RNN_sample_params,
        "TCN": TCN_sample_params,
    }
    
    if model_obj not in params_select:
        raise ValueError(
            f"No sampling function for model: {model_obj}. "
            f"Available models: {list(params_select.keys()) + ['proT', 'proT_sim', 'proT_adaptive']}"
        )
    
    return params_select[model_obj](trial)


# =============================================================================
# TRAINING WRAPPER FOR OPTUNA
# =============================================================================

def train_function_for_optuna(
    config,
    save_dir: Path,
    data_dir: Path,
    experiment_tag: str,
    cluster: bool,
    **kwargs
):
    """
    Wrapper for proT's trainer function.
    
    This function:
    1. Updates the config with proT-specific preprocessing
    2. Calls proT's trainer with k-fold cross-validation
    3. Returns DataFrame with metrics from all folds
    
    Args:
        config: OmegaConf configuration
        save_dir: Directory to save outputs
        data_dir: Directory containing training data
        experiment_tag: Tag for experiment tracking
        cluster: Whether running on cluster
        **kwargs: Additional arguments
        
    Returns:
        pd.DataFrame with columns like val_mae, val_r2, test_mae, etc.
    """
    # Update config (proT-specific preprocessing)
    config_updated = update_config(config)
    # Call proT's trainer
    df_metric = trainer(
        config=config_updated,
        data_dir=data_dir,
        save_dir=save_dir,
        cluster=cluster,
        experiment_tag=experiment_tag,
        resume_ckpt=None,
        plot_pred_check=False,
        debug=False
    )
    
    return df_metric


# =============================================================================
# METRICS EXTRACTION FOR OPTUNA
# =============================================================================

def get_metrics_for_optuna(train_results):
    """
    Extract metrics from proT's trainer results.
    
    proT's trainer returns a DataFrame where each row is a k-fold result
    and columns are metrics like val_mae, val_r2, test_mae, etc.
    
    This function aggregates across folds using mean and std.
    
    Args:
        train_results: pd.DataFrame from proT's trainer
        
    Returns:
        Dict of aggregated metrics
    """
    df = train_results
    
    # Define the metrics to track
    metrics = {
        "val_mae": ["mean", "std"],
        "val_r2": ["mean", "std"],
        "val_rmse": ["mean", "std"],
        "test_mae": ["mean", "std"],
        "test_r2": ["mean", "std"],
        "test_rmse": ["mean", "std"]
    }
    
    # Extract and aggregate metrics
    result = {}
    for metric, operations in metrics.items():
        for operation in operations:
            if operation == "mean":
                result[f"{metric}_{operation}"] = df[metric].mean()
            elif operation == "std":
                result[f"{metric}_{operation}"] = df[metric].std()
    
    return result


# =============================================================================
# CLI Commands
# =============================================================================

@click.group()
def cli():
    """Main CLI entry point."""
    pass


@click.command()
@click.option("--exp_id", required=True, help="Experiment folder containing the config file")
@click.option("--cluster", default=False, is_flag=True, help="Running on cluster?")
@click.option("--study_name", default="optimization_study", help="Name for the Optuna study")
@click.option("--exp_tag", default="NA", help="Tag for experiment manifest")
@click.option("--mode", required=True, type=click.Choice(['create', 'resume', 'summary']), 
              help="Mode: create new study, resume existing, or show summary")
@click.option("--scratch_path", default=None, help="SCRATCH path (for cluster)")
@click.option("--study_path", default=None, help="Path to store study database")
@click.option("--optimization_metric", default="val_mae_mean", help="Metric to optimize")
@click.option("--optimization_direction", default="minimize", 
              type=click.Choice(['minimize', 'maximize']), help="Optimization direction")
@click.option("--sampling_profile", default="baseline",
              type=click.Choice(['baseline']),
              help="Sampling bounds profile to use (default: baseline)")
def paramsopt(exp_id, cluster, study_name, exp_tag, mode, scratch_path, study_path,
              optimization_metric, optimization_direction, sampling_profile):
    """
    Run hyperparameter optimization using Optuna.
    
    Modes:
    - create: Create a new optimization study
    - resume: Resume an existing study
    - summary: Display results of the best trial
    """
    # =============================================================================
    # Initialize sampling bounds based on profile
    # =============================================================================
    global SAMPLING_BOUNDS
    SAMPLING_BOUNDS = SAMPLING_PROFILES[sampling_profile]
    
    print(f"Optuna optimization: exp_id={exp_id}, mode={mode}, study={study_name}")
    print(f"Sampling profile: {sampling_profile}")
    
    # =============================================================================
    # Set up directories
    # =============================================================================
    if scratch_path is None:
        exp_dir = join(ROOT_DIR, "experiments", exp_id)
    else:
        exp_dir = scratch_path
    
    data_dir = join(ROOT_DIR, "data", "input")
    
    # Check if experiment directory exists
    if not exists(exp_dir):
        raise ValueError(f"Experiment directory does not exist: {exp_dir}")
    
    print(f"Experiment directory: {exp_dir}")
    print(f"Data directory: {data_dir}")
    
    # =============================================================================
    # Load config to determine model type
    # =============================================================================
    pattern_config = re.compile(r'config.*\.yaml')
    config_files = [f for f in os.listdir(exp_dir) if pattern_config.match(f)]
    
    if len(config_files) != 1:
        raise ValueError(
            f"Expected 1 config file in {exp_dir}, found {len(config_files)}: {config_files}"
        )
    
    base_config = OmegaConf.load(join(exp_dir, config_files[0]))
    
    # =============================================================================
    # Create sampling function with config context
    # =============================================================================
    def sample_params_fn(trial):
        return sample_params_for_optuna(trial, base_config)
    
    # =============================================================================
    # Create Optuna Study
    # =============================================================================
    print(f"\nInitializing Optuna study...")
    print(f"Model: {base_config['model']['model_object']}")
    print(f"Optimization metric: {optimization_metric} ({optimization_direction})")
    
    optuna_study = OptunaStudy(
        exp_dir=exp_dir,
        data_dir=data_dir,
        cluster=cluster,
        study_name=study_name,
        manifest_tag=exp_tag,
        study_path=study_path,
        sample_params_fn=sample_params_fn,
        train_fn=train_function_for_optuna,
        get_metrics_fn=get_metrics_for_optuna,
        optimization_metric=optimization_metric,
        optimization_direction=optimization_direction
    )
    
    # =============================================================================
    # Execute based on mode
    # =============================================================================
    if mode == "create":
        print("\n" + "="*60)
        print("Creating new Optuna study...")
        print("="*60)
        optuna_study.create()
        print("\nStudy created! Next steps:")
        print(f"  1. Review the study configuration")
        print(f"  2. Run: python cli.py paramsopt --exp_id {exp_id} --study_name {study_name} --mode resume")
        
    elif mode == "resume":
        print("\n" + "="*60)
        print("Resuming Optuna study...")
        print("="*60)
        try:
            optuna_study.resume()
            print("\n" + "="*60)
            print("Optimization completed!")
            print("="*60)
            print(f"Run summary: python cli.py paramsopt --exp_id {exp_id} --study_name {study_name} --mode summary")
        except Exception as e:
            print(f"\nError during optimization: {e}")
            raise
        
    elif mode == "summary":
        print("\n" + "="*60)
        print("Generating study summary...")
        print("="*60)
        try:
            optuna_study.summary()
        except Exception as e:
            print(f"\nError generating summary: {e}")
            raise


# =============================================================================
# Register commands with CLI
# =============================================================================
cli.add_command(paramsopt)


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    cli()
