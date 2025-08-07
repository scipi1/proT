import numpy as np
from os.path import dirname, abspath, join, exists
import os
import sys
from argparse import ArgumentParser
import yaml


from os.path import abspath, join
sys.path.append(dirname(dirname(abspath(__file__))))
# from lightning.pytorch import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torch
from proT.dataloader import ProcessDataModule
from callbacks import early_stopping_callbacks, get_checkpoint_callback
from config import load_config
from forecaster import TransformerForecaster
from labels import *
from generate_report import generate_notebook
from predict import predict
from subroutines.sub_utils import mk_missing_folders
from modules.utils import set_seed
from pytorch_lightning import seed_everything

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train(
    config: dict, 
    data_dir: str, 
    save_dir: str, 
    cluster: bool, 
    resume_ckpt: str=None, 
    debug: bool=False,
    )->None:
    """
    Training function

    Args:
        config (dict): configuration file
        data_dir (str): data directory
        save_dir (str): saving directory
        cluster (bool): cluster used?
        resume_ckpt (str, optional): checkpoint to resume training. Defaults to None.
        debug (bool, optional): turn on debug options. Defaults to False.
    """

    
    logs_dir = join(save_dir, "logs") # save dir for Tensorboard/CSV logs and checkpoints
    mk_missing_folders([logs_dir])
    
    # define loggers and callbacks
    logger = TensorBoardLogger(save_dir=logs_dir, name="tensorboard")
    logger_csv = CSVLogger(save_dir=logs_dir, name="csv")
    checkpoint_callback = get_checkpoint_callback(save_dir)

    # set seed (TODO in config)
    seed_everything(42)
    torch.set_float32_matmul_precision("high")

    dm = ProcessDataModule(
        data_dir = join(data_dir,config["data"]["dataset"]),
        input_file =  config["data"]["filename_input"],
        target_file = config["data"]["filename_target"],
        batch_size =  config["training"]["batch_size"],
        num_workers = 1 if cluster else 20,
        data_format = "float32",
    )
    
    dm.prepare_data()
    

    model = TransformerForecaster(config)
    
    
    trainer = pl.Trainer(
        callbacks=[early_stopping_callbacks, checkpoint_callback],
        logger=[logger, logger_csv],
        accelerator="auto", #if args.gpu else "auto",
        devices=1 if cluster else "auto",
        overfit_batches=1 if debug else 0,
        max_epochs=config["training"]["max_epochs"],
        log_every_n_steps=1,  # if args.debug else 10,
        deterministic=True,
        gradient_clip_val=0.5,
        detect_anomaly=True,
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



if __name__ == "__main__":
    
    """
    Run a quick test
    """
    
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    exp_dir = join(ROOT_DIR, "experiments/test/")
    data_dir = join(ROOT_DIR, "data/input/")
    
    with open(join(exp_dir,"config.yaml"), 'r') as file:
        config = yaml.safe_load(file)
    
    save_dir = exp_dir
    
    train(
        config = config, 
        data_dir = data_dir, 
        save_dir = save_dir, 
        cluster = False, 
        resume_ckpt = None, 
        debug = True)
