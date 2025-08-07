import numpy as np
from os.path import dirname, abspath, join, exists
import os
import sys
from argparse import ArgumentParser

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


def main(args):

    exp_id = args.exp

    print(f"Performing experiment: {exp_id}")

    ROOT_DIR = dirname(dirname(abspath(__file__)))
    sys.path.append(ROOT_DIR)
    INPUT_DIR, _, _, EXPERIMENTS_DIR = get_dirs(ROOT_DIR)
    EXP_DIR = join(EXPERIMENTS_DIR, exp_id)

    assert exists(EXP_DIR), AssertionError("The experiment folder doesn't exist!")

    output_path = join(EXP_DIR, "output")
    logs_dir = join(EXP_DIR, "logs")
    mk_missing_folders([logs_dir])

    logger = TensorBoardLogger(save_dir=logs_dir, name="tensorboard")
    logger_csv = CSVLogger(save_dir=logs_dir, name="csv")
    config = load_config(EXP_DIR)
    dataset = config["data"]["dataset"]
    checkpoint_callback = get_checkpoint_callback(EXP_DIR)

    seed_everything(42)
    torch.set_float32_matmul_precision("high")

    dm = ProcessDataModule(
        ds_flag = config["data"]["ds_flag"],
        data_dir=join(INPUT_DIR, dataset),
        input_file =  config["data"]["filename_input"],
        target_file = config["data"]["filename_target"],
        batch_size =  config["training"]["batch_size"],
        num_workers =1 if args.cluster else 20,
        data_format = "float32" 
    )

    model = TransformerForecaster(config)

    # dynamic_kwargs = {
    #         "enc_mask_flag"         : False,
    #         "enc_mask"              : None,
    #         "dec_self_mask_flag"    : False,
    #         "dec_self_mask"         : None,
    #         "dec_cross_mask_flag"   : False,
    #         "dec_cross_mask"        : None,
    #         "enc_output_attn"       : False,
    #         "dec_self_output_attn"  : False,
    #         "dec_cross_output_attn" : False,}

    # model.set_kwargs(dynamic_kwargs)

    trainer = pl.Trainer(
        callbacks=[early_stopping_callbacks, checkpoint_callback],
        logger=[logger, logger_csv],
        accelerator="gpu" if args.gpu else "auto",
        devices=1 if args.cluster else "auto",
        overfit_batches=1 if args.debug else 0,
        fast_dev_run=True if args.devrun else False,
        max_epochs=config["training"]["max_epochs"],
        log_every_n_steps=1,  # if args.debug else 10,
        deterministic=True,
        gradient_clip_val=0.5,
        detect_anomaly=True,
    )
    # * other stuff we can do
    # trainer.tune() to find optimal hyperparameters

    checkpoint_path = None
    if args.checkpoint is not None:
        print(f"Resuming training from: {args.checkpoint}")
        checkpoint_path = join(EXP_DIR, "checkpoints", args.checkpoint)

    trainer.fit(
        model,
        dm,
        ckpt_path=checkpoint_path,
    )
    trainer.validate(model, dm)
    trainer.test(model, dm)

    plots_path = join(EXP_DIR, r"logs", r"csv")
    template_path = join(ROOT_DIR, r"notebooks/nb_experiment_template.ipynb")
    output_path = join(EXP_DIR, r"report.ipynb")

    generate_notebook(config, plots_path, template_path, output_path)
    print("Notebook report generated")
    print("END")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="overfit one batch for sanity check",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU acceleration in Lightning Trainer",
    )

    parser.add_argument(
        "--devrun", action="store_true", help="Run a quick test for debugging purpose"
    )

    parser.add_argument("--cluster", action="store_true", help="running on cluster")

    parser.add_argument(
        "--workers", action="store", type=int, default=1, help="running on cluster"
    )

    parser.add_argument("--exp", action="store", type=str, help="experiment folder")

    parser.add_argument(
        "--checkpoint", action="store", type=str, help="resume training from checkpoint"
    )

    args = parser.parse_args()

    main(args)
