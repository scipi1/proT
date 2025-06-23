from pathlib import Path
import click
import logging
from omegaconf import OmegaConf
from os import makedirs
import sys
from os.path import abspath, join, exists, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from prochain_transformer.kfold_train import kfold_train
from prochain_transformer.experiment_control import combination_sweep, independent_sweep
from prochain_transformer.predict import predict_test_from_ckpt
from prochain_transformer.modules.utils import mk_fname, find_last_checkpoint
from prochain_transformer.subroutines.sub_utils import save_output


@click.group()
def cli():
    pass

@click.command()
@click.option("--exp_id", help="Experiment folder containing the config file")
@click.option("--debug", default=False, help="Debug mode")
@click.option("--cluster", default=False, help="On the cluster?")
@click.option("--scratch_path", default=None, help="SCRATCH path")
@click.option("--resume_checkpoint", default=None, help="Resume training from checkpoint")
@click.option("--plot_pred_check", default=True, help="Set to True for a quick prediction plot after training")
@click.option("--sweep_mode", default="combination", help= "sweep mode, either 'independent' or 'combination'")
def train(exp_id, debug, cluster, scratch_path, resume_checkpoint, plot_pred_check,  sweep_mode:str):
    
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
        kfold_train(
            config = config, 
            data_dir = data_dir, 
            save_dir = save_dir, 
            cluster = cluster, 
            resume_ckpt = resume_checkpoint,
            plot_pred_check = plot_pred_check,
            debug = debug)
    
    run_sweep()
    
    
    
    
@click.command()
@click.option("--exp_id", default=None, help="Relative path to experiment from experiment/training")
@click.option("--out_id", default=None, help="Relative path to experiment from experiment/evaluations")
@click.option("--checkpoint", default=None, help="Relative checkpoint path from exp_id path")
@click.option("--cluster", default=False, help="On the cluster?")
@click.option("--debug", default=False, help="Debug mode")
def predict(exp_id:Path, out_id, checkpoint, cluster, debug)->None:
    
    # will be passed
    if exp_id is None:
        exp_id = r"dx_250406_epoch_sweeps\sweeps\sweep_max_epochs\sweep_max_epochs_1000"
    
    if out_id is None:
        out_id = r"dx_250406_1000_epoch\predictions"
    
    checkpoint = r"k_1\checkpoints\epoch=999-train_loss=0.02.ckpt"
    
    # Get folders #TODO make function in labels
    root_path = dirname(dirname(abspath(__file__)))
    exp_path = join(root_path, "experiments", "training", exp_id)
    output_path = join(root_path, "experiments", "evaluations", out_id)
    datadir_path = join(root_path,"data","input")
    
    if not(exists(output_path)):
        makedirs(output_path)
    
    # hardcoded for now but can be improved
    checkpoint_dir_relpath = "k_0\checkpoints" # relative path wrt exp_path
    
    config_path = join(exp_path,"config.yaml")
    config = OmegaConf.load(config_path)
        
    if checkpoint is None:
        checkpoint_path = find_last_checkpoint(checkpoint_dir=join(exp_path,checkpoint_dir_relpath))
        print(f"Loading last checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = join(exp_path,checkpoint)
    
    
    input_array, output_array, target_array, cross_att_array = predict_test_from_ckpt(config, datadir_path, checkpoint_path, cluster)
    
    save_output(
        src_trg_list=[
            (input_array, "input.npy"),
            (output_array, "output.npy"),
            (target_array, "target.npy"),
            (cross_att_array, "cross_att.npy")
        ],
        output_path = output_path)
    

cli.add_command(train)
cli.add_command(predict)

if __name__ == "__main__":
    cli()