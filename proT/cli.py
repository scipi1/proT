# Standard library imports
import logging
import sys
from os import makedirs
from os.path import abspath, join, exists, dirname
from pathlib import Path

# Third-party imports
import click
from omegaconf import OmegaConf

# Local imports
sys.path.append(dirname(dirname(abspath(__file__))))
from proT.experiment_control import combination_sweep
from proT.modules.utils import mk_fname, find_last_checkpoint
from proT.old_.kfold_train import kfold_train
from proT.optuna_opt import OptunaStudy
from proT.predict import predict_test_from_ckpt
from proT.subroutines.sub_utils import save_output
from proT.trainer import trainer


@click.group()
def cli():
    pass

# TRAINING
@click.command()
@click.option("--exp_id", help="Experiment folder containing the config file")
@click.option("--debug", default=False, help="Debug mode")
@click.option("--cluster", default=False, help="On the cluster?")
@click.option("--exp_tag", default="NA", help="Tag for model manifest")
@click.option("--scratch_path", default=None, help="SCRATCH path") # for the cluster
@click.option("--resume_checkpoint", default=None, help="Resume training from checkpoint")
@click.option("--plot_pred_check", default=True, help="Set to True for a quick prediction plot after training")
@click.option("--sweep_mode", default="combination", help= "sweep mode, either 'independent' or 'combination'")
def train(exp_id, debug, cluster, exp_tag, scratch_path, resume_checkpoint, plot_pred_check,  sweep_mode:str):
    
    # Get folders
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    
    print(exp_id)
    print(scratch_path)
    
    if scratch_path is None:
        exp_dir = join(ROOT_DIR, "experiments/training", exp_id)
    else:
        exp_dir = join(scratch_path)
        
    data_dir = join(ROOT_DIR, "data/input/")
    
    
    # Create loggers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger_info = logging.getLogger("logger_info")
    info_handler = logging.FileHandler(join(ROOT_DIR,  mk_fname(filename="log", label="train", suffix="log")))
    logger_info.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger_info.addHandler(info_handler)
    
    if debug:
        # memory logger
        logger_memory = logging.getLogger("logger_memory")
        memory_handler = logging.FileHandler(join(ROOT_DIR, mk_fname(filename="log", label="memory", suffix="log")))
        logger_memory.setLevel(logging.INFO)
        memory_handler.setFormatter(formatter)
        logger_memory.addHandler(memory_handler)
    
    
    
    @combination_sweep(exp_dir, mode=sweep_mode)
    #@independent_sweep(exp_dir)
    def run_sweep(config,save_dir):
        trainer(
            config=config, 
            data_dir=data_dir, 
            save_dir=save_dir, 
            cluster=cluster,
            experiment_tag=exp_tag,
            resume_ckpt=resume_checkpoint,
            plot_pred_check=plot_pred_check,
            debug=debug)
    
    run_sweep()
    
    
# PARAMETER OPTIMIZATION
@click.command()
@click.option("--exp_id", help="Experiment folder containing the config file")
@click.option("--cluster", default=False, help="On the cluster?")
@click.option("--study_name", default="NA", help="Tag for model manifest")
@click.option("--exp_tag", default="NA", help="Tag for model manifest")
@click.option("--mode", help="select between [`create`, `resume`, `summary`]") # for the cluster
@click.option("--scratch_path", default=None, help="SCRATCH path") # for the cluster
@click.option("--study_path", default=None, help="study path") # for the cluster
def paramsopt(exp_id, cluster, study_name, exp_tag, mode, scratch_path, study_path=None):
    
    # Get folders
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    
    print(exp_id)
    print(scratch_path)
    
    if scratch_path is None:
        exp_dir = join(ROOT_DIR, "experiments/training", exp_id)
    else:
        exp_dir = join(scratch_path)
        
    data_dir = join(ROOT_DIR, "data/input/")
    
    # Create loggers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger_info = logging.getLogger("logger_info")
    info_handler = logging.FileHandler(join(ROOT_DIR,  mk_fname(filename="log", label="train", suffix="log")))
    logger_info.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger_info.addHandler(info_handler)
    
    
    # run the optuna study
    optuna_study = OptunaStudy(
        exp_dir=exp_dir,
        data_dir=data_dir,
        cluster=cluster,
        study_name=study_name,
        manifest_tag = exp_tag,
        study_path= study_path
        )
    
    if mode == "create":
        optuna_study.create()
    
    elif mode == "resume":
        optuna_study.resume()
    
    elif mode == "summary":
        optuna_study.summary()
    


cli.add_command(train)
cli.add_command(paramsopt)

if __name__ == "__main__":
    cli()
