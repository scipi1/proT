from pathlib import Path
import click
import logging
from omegaconf import OmegaConf
from os import makedirs
import sys
from os.path import abspath, join, exists, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from prochain_transformer.kfold_train import kfold_train
from prochain_transformer.trainer import trainer
from prochain_transformer.experiment_control import combination_sweep, independent_sweep
from prochain_transformer.predict import predict_test_from_ckpt
from prochain_transformer.modules.utils import mk_fname, find_last_checkpoint
from prochain_transformer.subroutines.sub_utils import save_output
from prochain_transformer.optuna_opt import optuna_study


@click.group()
def cli():
    pass

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
    
    
    
@click.command()
@click.option("--exp_id", help="Experiment folder containing the config file")
@click.option("--debug", default=False, help="Debug mode")
@click.option("--cluster", default=False, help="On the cluster?")
@click.option("--study_name", default="NA", help="Tag for model manifest")
@click.option("--exp_tag", default="NA", help="Tag for model manifest")
@click.option("--scratch_path", default=None, help="SCRATCH path") # for the cluster
def paramsopt(exp_id, debug, cluster, study_name, exp_tag, scratch_path):
    
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
    
    # run the optuna study
    optuna_study(
        exp_dir=exp_dir,
        data_dir=data_dir,
        cluster=cluster,
        study_name=study_name,
        manifest_tag = exp_tag
    )
    



#____________________________________________________________________________________________________________________________________
    
# @click.command()
# @click.option("--exp_id", help="Experiment folder containing the config file")
# @click.option("--debug", default=False, help="Debug mode")
# @click.option("--cluster", default=False, help="On the cluster?")
# @click.option("--scratch_path", default=None, help="SCRATCH path")
# @click.option("--resume_checkpoint", default=None, help="Resume training from checkpoint")
# @click.option("--plot_pred_check", default=True, help="Set to True for a quick prediction plot after training")
# @click.option("--sweep_mode", default="combination", help= "sweep mode, either 'independent' or 'combination'")
# def global_train(exp_id, debug, cluster, scratch_path, resume_checkpoint, plot_pred_check,  sweep_mode:str):
    
#     # Get folders
#     ROOT_DIR = dirname(dirname(abspath(__file__)))
    
#     print(exp_id)
#     print(scratch_path)
    
    
#     # directory pointer for cluster
#     if scratch_path is None:
#         exp_dir = join(ROOT_DIR, "experiments/training", exp_id)
#     else:
#         exp_dir = join(scratch_path)
        
#     data_dir = join(ROOT_DIR, "data/input/")
    
#     # Create loggers
#     formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#     logger_info = logging.getLogger("logger_info")
#     info_handler = logging.FileHandler(join(ROOT_DIR,  mk_fname(filename="log", label="train", suffix="log")))
#     logger_info.setLevel(logging.INFO)
#     info_handler.setFormatter(formatter)
#     logger_info.addHandler(info_handler)
    
#     if debug:
#         # memory logger
#         logger_memory = logging.getLogger("logger_memory")
#         memory_handler = logging.FileHandler(join(ROOT_DIR, mk_fname(filename="log", label="memory", suffix="log")))
#         logger_memory.setLevel(logging.INFO)
#         memory_handler.setFormatter(formatter)
#         logger_memory.addHandler(memory_handler)
    
    
    
#     @combination_sweep(exp_dir, mode=sweep_mode)
#     #@independent_sweep(exp_dir)
#     def run_sweep(config,save_dir):
#         trainer(
#             config = config,
#             data_dir = data_dir, 
#             save_dir = save_dir, 
#             cluster = cluster, 
#             resume_ckpt = resume_checkpoint,
#             plot_pred_check = plot_pred_check, # set to False for all models but proT
#             debug = debug)
    
#     run_sweep()
    

cli.add_command(train)
cli.add_command(paramsopt)

if __name__ == "__main__":
    cli()