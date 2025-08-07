import numpy as np
import logging
from os.path import dirname, abspath, join
import os
import sys
from os.path import abspath, join
sys.path.append(dirname(dirname(abspath(__file__))))
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torch
from proT.dataloader import ProcessDataModule
from callbacks import early_stopping_callbacks, get_checkpoint_callback, MemoryLoggerCallback, GradientLogger, LayerRowStats, MetricsAggregator
from forecaster import TransformerForecaster
from labels import *
from subroutines.sub_utils import mk_missing_folders
from pytorch_lightning import seed_everything
from sklearn.model_selection import KFold
from proT.predict import mk_quick_pred_plot
from proT.experiment_control import update_config
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def kfold_train(
    config: dict, 
    data_dir: str, 
    save_dir: str,
    cluster: bool, 
    resume_ckpt: str=None,
    plot_pred_check: bool=False,
    debug: bool=False,
    )->None:
    """
    Training function with k-fold cross-validation

    Args:
        config (dict): configuration file
        data_dir (str): data directory
        save_dir (str): saving directory
        cluster (bool): cluster used?
        resume_ckpt (str, optional): checkpoint to resume training. Defaults to None.
        debug (bool, optional): turn on debug options. Defaults to False.
    """
    
    # set logging
    logger_info = logging.getLogger("logger_info")
    
    # set seed
    seed = config["training"]["seed"]
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    
    dm = ProcessDataModule(
        data_dir = join(data_dir,config["data"]["dataset"]),
        input_file =  config["data"]["filename_input"],
        target_file = config["data"]["filename_target"],
        batch_size =  config["training"]["batch_size"],
        num_workers = 1 if cluster else 20,
        data_format = "float32",
        max_data_size = config["data"]["max_data_size"],
        seed = seed,
    )
    
    
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
    
    
    for fold, (train_local_idx, val_local_idx) in enumerate(kfold.split(train_val_idx)):
        
        # re-initialize the model at any fold
        model = TransformerForecaster(config)
        
        print(f"Fold {fold + 1}/{k_folds}")
        logger_info.info(f"Fold {fold + 1}/{k_folds}")
        
        
        save_dir_k = join(save_dir, f"k_{fold}") # make subfolder for the given fold
        logs_dir = join(save_dir_k, "logs")      # save dir for Tensorboard/CSV logs and checkpoints
        mk_missing_folders([logs_dir])

        # define loggers and callbacks
        logger = TensorBoardLogger(save_dir=logs_dir, name="tensorboard")
        logger_csv = CSVLogger(save_dir=logs_dir, name="csv")
        checkpoint_callback = get_checkpoint_callback(save_dir_k,config["training"]["save_ckpt_every_n_epochs"])        
        
        
        callbacks_list = [
            #early_stopping_callbacks, 
            cb for cb in checkpoint_callback
            ]
        
        
        if debug:
            callbacks_list.append(MemoryLoggerCallback())
            
            
        if config["special"]["mode"] == "debug_optimizer":
            callbacks_list.append(GradientLogger())
            callbacks_list.append(LayerRowStats(layer_name="encoder_variable"))
            callbacks_list.append(LayerRowStats(layer_name="encoder_position"))
            callbacks_list.append(LayerRowStats(layer_name="final_ff"))
            callbacks_list.append(LearningRateMonitor(logging_interval="epoch"))
            # callbacks_list.append(MetricsAggregator())
        
        
        # update ds
        dm.update_idx(train_idx=train_local_idx, val_idx=val_local_idx, test_idx=test_idx)

        trainer = pl.Trainer(
            callbacks=callbacks_list,
            logger=logger_csv, #[logger, logger_csv],
            accelerator="gpu" if torch.cuda.is_available() else "auto",
            devices=1 if cluster else "auto",
            #overfit_batches=1 if debug else 0,
            max_epochs=config["training"]["max_epochs"],
            log_every_n_steps= 1,
            deterministic=True,
            enable_progress_bar=False if cluster else True,  # Disables the progress bar
            enable_model_summary=False if cluster else True,
            detect_anomaly=True if debug else False,
        )
        # * other stuff we can do
        # trainer.tune() to find optimal hyperparameters

        trainer.fit(
            model,
            dm,
            ckpt_path=resume_ckpt, # resume training from checkpoint
        )

        trainer.validate(model, dm)
        trainer.test(model, dm)
    
    if plot_pred_check:
        mk_quick_pred_plot(model=model, dm=dm, val_idx=config["data"]["val_idx"], save_dir=save_dir)



if __name__ == "__main__":
    
    """
    Run a quick test
    """
    
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    exp_dir = join(ROOT_DIR, "experiments/training/test/")
    data_dir = join(ROOT_DIR, "data/input/")
    
    
    config = OmegaConf.load(join(exp_dir,"config.yaml"))
    config_updated = update_config(config)
    
    
    save_dir = exp_dir
    
    kfold_train(
        config = config_updated, 
        data_dir = data_dir, 
        save_dir = save_dir, 
        cluster = False, 
        resume_ckpt = None,
        plot_pred_check = True,
        debug = True)
