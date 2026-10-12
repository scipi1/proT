# Standard library imports
import json
import logging
import os
import sys
import time
import traceback
from os.path import dirname, abspath, join

# Third-party imports
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import KFold

# Local imports
from proT.training.callbacks.training_callbacks import (
    early_stopping_callbacks, get_checkpoint_callback, MemoryLoggerCallback, 
    GradientLogger, LayerRowStats, MetricsAggregator, PerRunManifest, 
    AttentionEntropyLogger, BestCheckpointCallback, DataIndexTracker, 
    KFoldResultsTracker, QKFreezeScheduler
)
from proT.training.forecasters import *
from proT.training.forecasters.encoder_forecaster import EncoderForecaster
from proT.labels import *
from proT.baseline.baseline_pl_modules import RNNForecaster
from proT.training.dataloader import ProcessDataModule
from proT.training.hard_mining import HardMiningState
from proT.training.experiment_control import update_config
from proT.evaluation.predict import mk_quick_pred_plot

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def trainer(
    config: dict,
    data_dir: str,
    save_dir: str,
    cluster: bool,
    experiment_tag: str="NA",
    resume_ckpt: str=None,
    plot_pred_check: bool=False,
    debug: bool=False,
    best: bool=True,
    folds: list=None,
    )->pd.DataFrame:
    """
    Training function with k-fold cross-validation

    Args:
        config (dict): configuration file
        data_dir (str): data directory
        save_dir (str): saving directory
        cluster (bool): cluster used?
        experiment_tag (str, optional): experiment tag for logging. Defaults to "NA".
        resume_ckpt (str, optional): checkpoint to resume training. Defaults to None.
        plot_pred_check (bool, optional): whether to plot prediction check. Defaults to False.
        debug (bool, optional): turn on debug options. Defaults to False.
        best (bool, optional): if True, evaluate validation and test on the best
                              checkpoint (based on val_mae) so that kfold_summary
                              and best_metrics.json are consistent. If False, use
                              final epoch weights. Defaults to True.
        folds (list, optional): fold numbers to run (e.g. [2] for one SLURM
                              array task). The split is deterministic, so
                              parallel jobs see the same folds as a
                              sequential run. None = all folds.
    
    Returns:
        pd.DataFrame: DataFrame containing metrics for each fold
    """
    # set logging
    logger_info = logging.getLogger("logger_info")
    
    # set seed
    seed = config["training"]["seed"]
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    
    # get model object from configuration (version 4.6.0 or higher)
    model_object = get_model_object(config)
    
    # Note: plot_pred_check now works with all model types (proT and baselines)
    
    
    # num_workers: 0 on Windows to avoid multiprocessing DLL paging-file crashes
    default_workers = 0 if sys.platform == "win32" else (1 if cluster else 20)
    num_workers = config["training"].get("num_workers", default_workers)
    
    dm = ProcessDataModule(
        data_dir                = join(data_dir,config["data"]["dataset"]),
        input_file              = config["data"]["filename_input"],
        target_file             = config["data"]["filename_target"],
        batch_size              =  config["training"]["batch_size"],
        num_workers             = num_workers,
        data_format             = "float32",
        max_data_size           = config["data"]["max_data_size"],
        input_p_blank           = config["data"].get("input_p_blank", None),
        input_blanking_val_idx  = config["data"]["features"]["X"]["value"],
        seed                    = seed,
        train_file              = config["data"].get("train_file", None),
        test_file               = config["data"].get("test_file", None),
        use_val_split           = True
    )
    
    
    # Check if using pre-split data
    use_presplit = config["data"].get("train_file") is not None and config["data"].get("test_file") is not None
    
    if use_presplit:
        # Pre-split data: k-fold only on training data, test is already separate
        dataset_size = dm.get_ds_len()  # This returns training data size
        train_val_idx = np.arange(dataset_size)
        test_idx = None  # Test data is already split separately
        print("Using pre-split data: k-fold will be applied only to training data.")
    
    else:
        # Normal data: need to create train/val/test split
        dataset_size = dm.get_ds_len()
        indices = np.arange(dataset_size)
        
        test_ds_idx_filename = config["data"]["test_ds_ixd"]
        
        if test_ds_idx_filename is not None:
            test_idx = np.load(join(data_dir,config["data"]["dataset"],test_ds_idx_filename))
            mask = np.isin(indices, test_idx)
            train_val_idx = indices[~mask]
        else:
            test_size = int(0.2 * dataset_size)  # Reserve 20% for testing
            test_idx = indices[:test_size]       # Fixed test indices
            train_val_idx = indices[test_size:]  # Remaining for train/val cross-validation
    
    
    
    
    # k-fold cross-validation
    k_folds = config["training"]["k_fold"]
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    metrics_dict = {}
    
    # Initialize k-fold results tracker
    kfold_tracker = KFoldResultsTracker(
        save_dir, k_folds,
        best_metric=config["training"].get("best_checkpoint_metric", "val_mae"))
    
    # Count trainable parameters (same for all folds, calculated once)
    temp_model = model_object(config)
    trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    del temp_model  # Clean up temporary model
    
    # Initialize variables for tracking last fold's checkpoint
    last_checkpoint_path = None
    last_save_dir_k = None
    
    for fold, (train_local_idx, val_local_idx) in enumerate(kfold.split(train_val_idx)):

        # fold selection (parallel runs): same deterministic split, others skipped
        if folds is not None and fold not in folds:
            continue
        # per-fold seed: results identical whether folds run sequentially
        # or in parallel jobs
        seed_everything(seed + fold)

        # re-initialize the model at any fold
        model = model_object(config)
        
        print(f"Fold {fold + 1}/{k_folds}")
        logger_info.info(f"Fold {fold + 1}/{k_folds}")
        
        save_dir_k = join(save_dir, f"k_{fold}") # make subfolder for the given fold
        logs_dir = join(save_dir_k, "logs")      # save dir for Tensorboard/CSV logs and checkpoints
        os.makedirs(logs_dir, exist_ok=True)

        # Convert local indices to global indices
        train_global_idx = train_val_idx[train_local_idx]
        val_global_idx = train_val_idx[val_local_idx]

        # define loggers and callbacks
        #logger = TensorBoardLogger(save_dir=logs_dir, name="tensorboard")
        logger_csv = CSVLogger(save_dir=logs_dir, name="csv")
        checkpoint_callback = get_checkpoint_callback(save_dir_k,config["training"]["save_ckpt_every_n_epochs"])        
        manifest_callback  = PerRunManifest(config, path=save_dir_k, tag=experiment_tag)
        
        # Add new callbacks for best checkpoint tracking and data index saving
        # For probabilistic losses (gaussian, gaussian_censored), monitor the NLL
        # instead of the MAE: the NLL accounts for the predicted uncertainty.
        best_metric = config["training"].get("best_checkpoint_metric", "val_mae")
        # Skip ALL warmup phases: metrics are not comparable while teacher
        # forcing is active (curriculum), and the objective itself changes
        # when the adaptive sample weighter / hard-example mining switch on
        # (the logged loss jumps at their start epoch). The lower bound is
        # the max of all warmup schedules; an explicit
        # best_checkpoint_min_epoch in the config overrides it.
        min_epoch = max(
            config["training"].get("epoch_show_trg", 0) + 1
                if config["training"].get("target_show_mode") is not None else 0,
            config["training"].get("loss_sw_start_epoch", 0)
                if config["training"].get("loss_sample_weight") == "adaptive" else 0,
            config["training"].get("loss_hm_start_epoch", 0)
                if config["training"].get("loss_hard_mining", False) else 0,
        )
        if config["training"].get("target_show_mode") == "scheduled":
            # scheduled curriculum: metrics are biased while any teacher forcing
            # is active; start best-checkpoint selection only from the fully
            # autoregressive floor (end of the annealing schedule)
            min_epoch = max(min_epoch, int(round(
                config["training"].get("show_trg_schedule_end_frac", 1.0)
                * config["training"]["max_epochs"])))
        min_epoch = config["training"].get("best_checkpoint_min_epoch", min_epoch)
        best_checkpoint_callback = BestCheckpointCallback(
            save_dir_k, monitor=best_metric, mode="min", min_epoch=min_epoch)
        data_index_tracker = DataIndexTracker(save_dir_k, fold, train_global_idx, val_global_idx, test_idx)
        
        callbacks_list = [cb for cb in checkpoint_callback]
        callbacks_list.append(manifest_callback)
        callbacks_list.append(best_checkpoint_callback)
        callbacks_list.append(data_index_tracker)
        
        # Add attention entropy logging callback
        entropy_enabled = True
        callbacks_list.append(AttentionEntropyLogger(enabled=entropy_enabled))
        
        # Delayed Q/K learning via hard freeze (mutually exclusive with the
        # Q/K LR ramp, which is handled in the forecaster's configure_optimizers)
        qk_freeze_epochs = config["training"].get("qk_freeze_epochs", 0)
        if qk_freeze_epochs > 0 and not config["training"].get("qk_lr_ramp", False):
            callbacks_list.append(QKFreezeScheduler(
                freeze_epochs=qk_freeze_epochs,
                scope=config["training"].get("qk_freeze_scope", "all")
            ))
        
        if "early_stopping" in config["special"]["mode"]:
            callbacks_list.append(early_stopping_callbacks)
        
        if debug:
            callbacks_list.append(MemoryLoggerCallback())
        
        if "debug_optimizer" in config["special"]["mode"]:
            callbacks_list.append(GradientLogger())
            callbacks_list.append(LayerRowStats(layer_name="encoder_variable"))
            callbacks_list.append(LayerRowStats(layer_name="encoder_position"))
            callbacks_list.append(LayerRowStats(layer_name="final_ff"))
            callbacks_list.append(LearningRateMonitor(logging_interval="epoch"))
            # callbacks_list.append(MetricsAggregator())
        
        # update ds
        dm.update_idx(train_idx=train_local_idx, val_idx=val_local_idx, test_idx=test_idx)

        # Hard-example mining: shared per-sample loss memory between the batch
        # sampler (dm) and the forecaster (model); reset for every fold.
        # See proT/training/hard_mining.py.
        hm_state = None
        if config["training"].get("loss_hard_mining", False):
            hm_state = HardMiningState(
                n_samples=len(train_local_idx),
                frac=config["training"].get("loss_hm_frac", 0.25),
                pool_frac=config["training"].get("loss_hm_pool", 0.10),
                start_epoch=config["training"].get("loss_hm_start_epoch", 0),
                momentum=config["training"].get("loss_hm_momentum", 0.9),
            )
            model.hard_mining_state = hm_state
        dm.hard_mining_state = hm_state

        trainer = pl.Trainer(
            callbacks=callbacks_list,
            logger=logger_csv, #[logger, logger_csv],
            accelerator="gpu" if torch.cuda.is_available() else "auto",
            devices=1 if cluster else "auto",
            #overfit_batches=1 if debug else 0,
            max_epochs=config["training"]["max_epochs"],
            log_every_n_steps= 1,
            # validate every epoch: best-checkpoint selection can only pick
            # epochs it actually sees. A sparse validation grid biases the
            # selection towards whichever epochs happened to be evaluated.
            check_val_every_n_epoch=config["training"].get("check_val_every_n_epoch", 1),
            deterministic=True,
            # hard mining: rebuild the train dataloader every epoch so the
            # batch sampler sees the updated per-sample loss memory
            reload_dataloaders_every_n_epochs=1 if hm_state is not None else 0,
            enable_progress_bar=False if cluster else True,  # Disables the progress bar
            enable_model_summary=False if cluster else True,
            detect_anomaly=True if debug else False,
            # gradient clipping (default 0.0 = disabled, Lightning default);
            # protects long AR-rollout runs from gradient explosions
            gradient_clip_val=config["training"].get("gradient_clip_val", 0.0),
            gradient_clip_algorithm="norm",
        )
        # * other stuff we can do
        # trainer.tune() to find optimal hyperparameters

        # Record start time
        start_time = time.time()

        # training
        trainer.fit(
            model,
            dm,
            ckpt_path=resume_ckpt, # resume training from checkpoint
        )

        # Calculate training time
        training_time = time.time() - start_time
        num_epochs = trainer.current_epoch + 1
        avg_time_per_epoch = training_time / num_epochs if num_epochs > 0 else 0

        # Determine evaluation checkpoint: with best=True, both validation and
        # test are evaluated on the best (val_mae) checkpoint, so that
        # kfold_summary.json and best_metrics.json are consistent. With
        # best=False, final-epoch weights are used.
        best_ckpt = best_checkpoint_callback.best_checkpoint_path
        use_best = best and best_ckpt is not None and os.path.exists(best_ckpt)
        eval_ckpt = best_ckpt if use_best else None

        # validation (only if validation dataset exists)
        if dm.val_ds is not None:
            trainer.validate(model, dm, ckpt_path=eval_ckpt)
            val_metrics = trainer.callback_metrics.copy()
        else:
            val_metrics = {}
        
        # test (on the same checkpoint as validation)
        trainer.test(model, dm, ckpt_path=eval_ckpt)
        test_metrics = trainer.callback_metrics.copy()
        
        # Determine which metrics to report
        if use_best:
            # on_test_end has written best_metrics.json with consistent
            # val/train (best epoch) and test (best checkpoint) metrics
            best_metrics_file = join(save_dir_k, "best_metrics.json")
            with open(best_metrics_file, 'r') as f:
                best_metrics_data = json.load(f)
            
            # Extract metrics (keep best_epoch as diagnostics, drop the path)
            fold_metrics = {k: v for k, v in best_metrics_data.items() 
                           if k != 'best_checkpoint_path'}
            best_checkpoint_path = best_metrics_data.get('best_checkpoint_path')
        else:
            # Final-epoch metrics (also the fallback if no best checkpoint exists)
            fold_metrics = {**val_metrics, **test_metrics}
            best_checkpoint_path = None
        
        # Add model size and timing metrics
        fold_metrics['trainable_params'] = trainable_params
        fold_metrics['total_training_time'] = training_time
        fold_metrics['avg_time_per_epoch'] = avg_time_per_epoch
        
        # Update metrics dictionary
        metrics_dict[fold] = fold_metrics
        
        # Add fold result to tracker
        kfold_tracker.add_fold_result(fold, fold_metrics, best_checkpoint_path)
        
        # Track checkpoint for plotting (from last fold)
        last_save_dir_k = save_dir_k
        if best and best_checkpoint_path is not None:
            last_checkpoint_path = best_checkpoint_path
        else:
            # Use last checkpoint from this fold
            last_checkpoint_path = join(save_dir_k, "checkpoints", "best_checkpoint.ckpt")
    
    # Convert the dictionary to a pandas DataFrame
    df_metric = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df_metric = df_metric.map(lambda x: x.item() if isinstance(x, torch.Tensor) else x) # Convert tensor values to floats
    
    if plot_pred_check:
        # Use tracked checkpoint from last fold
        if last_checkpoint_path and os.path.exists(last_checkpoint_path):
            try:
                mk_quick_pred_plot(
                    config=config,
                    checkpoint_path=last_checkpoint_path,
                    datadir_path=data_dir,
                    val_idx=config["data"]["val_idx"],
                    save_dir=save_dir
                )
            except Exception as e:
                # Log error to file
                error_log_path = join(save_dir, "plot_error.log")
                with open(error_log_path, 'w') as f:
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Error: {type(e).__name__}: {str(e)}\n\n")
                    f.write(traceback.format_exc())
                
                # Print warning to console
                print(f"Warning: Failed to create prediction plot. Error logged to {error_log_path}")
        else:
            print(f"Warning: Could not create prediction plot. Checkpoint not found at {last_checkpoint_path}")

    return df_metric



def get_model_object(config: dict)->pl.LightningModule:
    
    model_obj = config["model"]["model_object"]
    available_models = ["proT", "proT_sim", "proT_adaptive", "proT_enc",
                        "LSTM", "GRU", "TCN", "MLP", "S6"]
    
    assert model_obj in available_models, AssertionError(f"{model_obj} unavailable! Choose between {available_models}")

    MODEL_REGISTRY = {
        # PRO-T
        "proT"          : EntropyRegularizedForecaster,
        "proT_sim"      : SimulatorForecaster,
        "proT_adaptive" : OnlineTargetForecaster,
        "proT_enc"      : EncoderForecaster,

        # BASELINE MODELS
        "GRU"           : RNNForecaster,
        "LSTM"          : RNNForecaster,
        "TCN"           : RNNForecaster,
        "MLP"           : RNNForecaster,
        "S6"            : RNNForecaster,
    }
    return MODEL_REGISTRY[model_obj]






if __name__ == "__main__":
    
    """
    Run a quick test
    """
    import re
    
    ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
    exp_dir = join(ROOT_DIR, "experiments/fusion/fusion_proT_ishigami_noisy_cat")
    data_dir = join(ROOT_DIR, "data/input/")
    
    # look for config file
    pattern_config = re.compile(r'config_.*\.yaml')
    config_matching_files = [file for file in os.listdir(exp_dir) if pattern_config.match(file)]
    if len(config_matching_files) == 1:
        config = OmegaConf.load(join(exp_dir,config_matching_files[0]))
    else:
        raise ValueError(f"None or more than one config file found in {exp_dir}")
    
    
    config_updated = update_config(config)
        
    save_dir = exp_dir
    
    trainer(
        config = config_updated,
        data_dir = data_dir, 
        save_dir = save_dir,
        experiment_tag = "test", 
        cluster = False, 
        resume_ckpt = None,
        plot_pred_check = True,
        debug = True)
