"""
Generic Optuna Optimization Skeleton

This module provides a reusable framework for hyperparameter optimization using Optuna.
To adapt this to your project:
1. Implement the sample_params function with your hyperparameter search space
2. Implement the train_function to train your model
3. Implement the get_metrics function to extract your evaluation metrics
4. Update the OptunaStudy class initialization if needed
"""

from omegaconf import OmegaConf
import torch
import yaml
from pathlib import Path
import optuna
import os
from optuna.exceptions import DuplicatedStudyError
from optuna.study import MaxTrialsCallback
from os.path import dirname, abspath, join
import sys
import re
from functools import partial
from typing import Callable, Dict, Any

ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)

# =============================================================================
# CUSTOMIZE: Import your project-specific training function here
# =============================================================================
# Example:
# from your_project.trainer import trainer
# from your_project.experiment_control import update_config


# =============================================================================
# CUSTOMIZE: Define your hyperparameter search space
# =============================================================================
def sample_params_template(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Template function for sampling hyperparameters.
    
    Replace this with your actual hyperparameter search space.
    This should return a flat dict of parameter_name -> sampled_value.
    Use dot notation for nested config keys (e.g., "model.hidden_dim").
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dict mapping dotted config keys to sampled values
        
    Example:
        return {
            "model.hidden_dim": trial.suggest_int("hidden_dim", 64, 512, step=64),
            "model.num_layers": trial.suggest_int("num_layers", 2, 8),
            "training.lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "training.batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "training.dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        }
    """
    # TODO: Replace with your actual search space
    return {
        "model.hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=64),
        "model.num_layers": trial.suggest_int("num_layers", 2, 6),
        "training.lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "training.dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
    }


# =============================================================================
# CUSTOMIZE: Define your training function
# =============================================================================
def train_function_template(
    config: dict,
    save_dir: Path,
    data_dir: Path,
    experiment_tag: str,
    cluster: bool,
    **kwargs
) -> Any:
    """
    Template function for training your model.
    
    Replace this with your actual training logic.
    This function should train a model and return metrics.
    
    Args:
        config: Configuration dictionary (OmegaConf)
        save_dir: Directory to save model checkpoints and outputs
        data_dir: Directory containing training data
        experiment_tag: Tag for experiment tracking
        cluster: Whether running on a cluster
        **kwargs: Additional arguments
        
    Returns:
        Metrics dictionary or DataFrame with evaluation results
        Should contain keys like 'val_loss', 'val_accuracy', etc.
        
    Example:
        model = build_model(config)
        train_loader = load_data(data_dir, config)
        metrics = train_model(model, train_loader, config)
        return metrics
    """
    # TODO: Implement your training function
    raise NotImplementedError(
        "You must implement the train_function_template with your actual training logic."
    )


# =============================================================================
# CUSTOMIZE: Define how to extract metrics from training results
# =============================================================================
def get_metrics_template(train_results: Any) -> Dict[str, Any]:
    """
    Template function for extracting metrics from training results.
    
    Replace this with your actual metric extraction logic.
    
    Args:
        train_results: Return value from train_function_template
        
    Returns:
        Dictionary mapping metric names to their values
        Should include at least the metric you want to optimize
        
    Example:
        return {
            "val_loss": train_results["val_loss"].mean(),
            "val_loss_std": train_results["val_loss"].std(),
            "test_loss": train_results["test_loss"].mean(),
            "accuracy": train_results["accuracy"].mean(),
        }
    """
    # TODO: Implement your metric extraction
    raise NotImplementedError(
        "You must implement get_metrics_template with your metric extraction logic."
    )


# =============================================================================
# Helper Functions (Generally reusable)
# =============================================================================
def get_config_run(base_config, exp_path: Path, params: Dict[str, Any], trial_id: int):
    """
    Creates a trial-specific configuration.
    
    Loads base config, applies parameter overrides, and saves to 
    exp_path/optuna/run_<trial_id>/config.yaml.
    
    Args:
        base_config: Base configuration (OmegaConf)
        exp_path: Experiment directory path
        params: Dictionary of parameters to override (dotted keys)
        trial_id: Trial number
        
    Returns:
        Tuple of (config_path, save_dir)
    """
    # Create hard-copy of config
    config_ = base_config.copy()
    
    # Ensure config is not in struct mode to allow updates
    OmegaConf.set_struct(config_, False)
    
    # Set sweep values in new config
    for dotted_key, val in params.items():
        OmegaConf.update(config_, dotted_key, val, merge=False)

    # Create a directory for the current run
    save_dir = Path(join(exp_path, "optuna", f"run_{trial_id}")) 
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save new config in the directory
    config_path = save_dir / "config.yaml"
    OmegaConf.save(config_, config_path)
    
    return config_path, save_dir


def objective_extended(
    trial: optuna.Trial,
    sample_params: Callable,
    train_function: Callable,
    get_metrics: Callable,
    config: dict,
    exp_path: Path,
    data_dir: Path,
    experiment_tag: str,
    cluster: bool,
    optimization_metric: str = "val_loss",
    optimization_direction: str = "minimize"
):
    """
    Optuna objective function.
    
    This function orchestrates:
    1. Sampling hyperparameters
    2. Creating trial-specific config
    3. Training the model
    4. Extracting and logging metrics
    5. Returning the optimization metric
    
    Args:
        trial: Optuna trial object
        sample_params: Function to sample hyperparameters
        train_function: Function to train model
        get_metrics: Function to extract metrics from training results
        config: Base configuration
        exp_path: Experiment path
        data_dir: Data directory
        experiment_tag: Experiment tag
        cluster: Whether on cluster
        optimization_metric: Name of metric to optimize (e.g., "val_loss")
        optimization_direction: "minimize" or "maximize"
        
    Returns:
        Value of the optimization metric
    """
    # Sample hyperparameters
    params = sample_params(trial)
    
    # Create trial-specific config
    config_path, save_dir = get_config_run(config, exp_path, params, trial.number)
    
    # Load and update run config (if you have a config update function)
    config_run = OmegaConf.load(config_path)
    # config_updated = update_config(config_run)  # CUSTOMIZE: Uncomment if needed
    
    # Set CUDA device if on cluster
    if cluster and torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    # Train model
    try:
        train_results = train_function(
            config=config_run,
            data_dir=data_dir,
            save_dir=save_dir,
            experiment_tag=experiment_tag,
            cluster=cluster,
            resume_ckpt=None,
            debug=False
        )
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise
    
    # Extract metrics
    metrics = get_metrics(train_results)
    
    # Log all metrics as user attributes
    for metric_name, metric_value in metrics.items():
        trial.set_user_attr(metric_name, float(metric_value))
    
    # Log config path
    trial.set_user_attr("config_path", str(config_path))
    
    # Return optimization metric
    if optimization_metric not in metrics:
        raise ValueError(
            f"Optimization metric '{optimization_metric}' not found in metrics: {list(metrics.keys())}"
        )
    
    return metrics[optimization_metric]


# =============================================================================
# Main Optuna Study Class
# =============================================================================
class OptunaStudy:
    """
    Manages Optuna hyperparameter optimization studies.
    
    This class handles:
    - Study creation with configurable samplers and pruners
    - Study resumption from storage
    - Results summary generation
    
    Args:
        exp_dir: Experiment directory containing config files
        data_dir: Data directory path
        cluster: Whether running on cluster
        study_name: Name for the Optuna study
        manifest_tag: Tag for experiment manifest
        study_path: Optional path to store study database
        sample_params_fn: Function to sample hyperparameters (default: sample_params_template)
        train_fn: Function to train model (default: train_function_template)
        get_metrics_fn: Function to extract metrics (default: get_metrics_template)
        optimization_metric: Metric to optimize (default: "val_loss")
        optimization_direction: "minimize" or "maximize" (default: "minimize")
    """
    
    def __init__(
        self,
        exp_dir: Path,
        data_dir: Path,
        cluster: bool,
        study_name: str,
        manifest_tag: str,
        study_path: str = None,
        sample_params_fn: Callable = sample_params_template,
        train_fn: Callable = train_function_template,
        get_metrics_fn: Callable = get_metrics_template,
        optimization_metric: str = "val_loss",
        optimization_direction: str = "minimize"
    ):
        self.exp_dir = exp_dir
        self.data_dir = data_dir
        self.cluster = cluster
        self.study_name = study_name
        self.manifest_tag = manifest_tag
        self.optimization_metric = optimization_metric
        self.optimization_direction = optimization_direction
        
        # Load config file
        pattern_config = re.compile(r'config.*\.yaml')
        config_matching_files = [
            file for file in os.listdir(exp_dir) 
            if pattern_config.match(file)
        ]
        if len(config_matching_files) == 1:
            config = OmegaConf.load(join(exp_dir, config_matching_files[0]))
        elif len(config_matching_files) == 0:
            raise ValueError(f"No config file found in {exp_dir}")
        else:
            raise ValueError(f"Multiple config files found in {exp_dir}: {config_matching_files}")

        # Load optuna settings file (optional)
        pattern_optuna = re.compile(r'optuna.*\.yaml')
        optuna_matching_files = [
            file for file in os.listdir(exp_dir) 
            if pattern_optuna.match(file)
        ]
        if len(optuna_matching_files) == 1:
            self.optuna_settings = OmegaConf.load(join(exp_dir, optuna_matching_files[0]))
        else:
            self.optuna_settings = None

        # Define objective function
        obj_kwargs = {
            "sample_params": sample_params_fn,
            "train_function": train_fn,
            "get_metrics": get_metrics_fn,
            "config": config,
            "exp_path": exp_dir,
            "data_dir": data_dir,
            "experiment_tag": manifest_tag,
            "cluster": cluster,
            "optimization_metric": optimization_metric,
            "optimization_direction": optimization_direction
        }
        self.objective = partial(objective_extended, **obj_kwargs)
    
        # Define storage path
        if study_path is None:
            study_path = (Path(exp_dir) / "optuna").resolve()
            study_path.mkdir(parents=True, exist_ok=True)
        
        self.study_file_path = join(study_path, "study.db")
        
        # Set study parameters
        self.storage = f"sqlite:///{self.study_file_path}?timeout=60"
        self.max_trials = self._setting("n_trials", 50)
        self.direction = self._setting("direction", optimization_direction)
        self.pruner = self._setting("pruner", "none")
        
    def create(self):
        """Create a new Optuna study."""
        try:
            study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                sampler=self._build_sampler(),
                pruner=self._build_pruner(),
                storage=self.storage,
            )
            print(f"Created study '{self.study_name}' stored at {self.storage}")
            print(f"Study will run {self.max_trials} trials")
            print(f"Optimizing '{self.optimization_metric}' ({self.direction})")
            
        except DuplicatedStudyError:
            print(f"Study '{self.study_name}' already exists — use --mode resume to continue")
            return
        
    def resume(self, storage=None):
        """
        Resume an existing Optuna study.
        
        Args:
            storage: Optional storage path (defaults to self.storage)
        """
        stor = storage if storage is not None else self.storage
        study = optuna.load_study(study_name=self.study_name, storage=stor)
        
        print(f"Resuming study '{self.study_name}' from {stor}")
        print(f"Current progress: {len(study.trials)} trials completed")
        print(f"Target: {self.max_trials} total trials")
        
        study.optimize(
            self.objective,
            n_trials=None,  # Loop until no trials remain
            callbacks=[MaxTrialsCallback(self.max_trials, states=None)],
            gc_after_trial=True,
            catch=(RuntimeError,),
        )
        
        print(f"Study completed! Total trials: {len(study.trials)}")
        
    def summary(self):
        """Generate and save a summary of the best trial."""
        study = optuna.load_study(study_name=self.study_name, storage=self.storage)
        best = study.best_trial
        
        print(f"\n{'='*60}")
        print(f"Best Trial Summary")
        print(f"{'='*60}")
        print(f"Trial number: {best.number}")
        print(f"Optimization metric ({self.optimization_metric}): {best.value:.6f}")
        print(f"\nBest Parameters:")
        for key, value in best.params.items():
            print(f"  {key}: {value}")
        
        # Collect all user attributes (metrics)
        to_dump = {
            "trial_number": best.number,
            "optimization_metric": self.optimization_metric,
            "optimization_value": float(best.value),
            "config_path": best.user_attrs.get("config_path"),
            "params": best.params,
            "metrics": {}
        }
        
        # Add all metrics from user attributes
        for key, value in best.user_attrs.items():
            if key != "config_path":
                to_dump["metrics"][key] = float(value) if isinstance(value, (int, float)) else value
        
        # Save summary
        summary_path = Path(self.exp_dir) / "best_trial.yaml"
        with open(summary_path, 'w') as yaml_file:
            yaml.dump(to_dump, yaml_file, default_flow_style=False, sort_keys=False)
        
        print(f"\nSummary saved to: {summary_path}")
        print(f"Config path: {to_dump['config_path']}")
        print(f"{'='*60}\n")
    
    def _setting(self, key: str, default: Any) -> Any:
        """Get setting from optuna_settings or return default."""
        return (self.optuna_settings or {}).get(key, default)
    
    def _build_sampler(self) -> optuna.samplers.BaseSampler:
        """Build Optuna sampler from settings."""
        if self.optuna_settings is None or "sampler" not in self.optuna_settings:
            return optuna.samplers.QMCSampler(qmc_type="sobol")
        
        cfg = dict(self.optuna_settings["sampler"])
        name = cfg.pop("name", "sobol").lower()
        
        if name == "sobol":
            return optuna.samplers.QMCSampler(qmc_type="sobol", **cfg)
        elif name == "tpe":
            return optuna.samplers.TPESampler(**cfg)
        else:
            raise ValueError(f"Unknown sampler name: {name}")
    
    def _build_pruner(self) -> optuna.pruners.BasePruner:
        """Build Optuna pruner from settings."""
        pruner_name = self._setting("pruner", "none")
        
        if pruner_name == "none":
            return None
        elif pruner_name == "median":
            n_warmup = self._setting("pruner_warmup", 5)
            return optuna.pruners.MedianPruner(n_warmup_steps=n_warmup)
        elif pruner_name == "hyperband":
            return optuna.pruners.HyperbandPruner()
        else:
            print(f"Warning: Unknown pruner '{pruner_name}', using no pruner")
            return None
