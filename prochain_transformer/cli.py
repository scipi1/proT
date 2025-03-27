from os.path import dirname, abspath, join
import sys
from os.path import abspath, join
sys.path.append(dirname(dirname(abspath(__file__))))
from prochain_transformer.kfold_train import kfold_train
from prochain_transformer.exp_control import independent_sweep
from modules.utils import mk_fname
import click
import logging


@click.group()
def cli():
    pass

@click.command()
@click.option("--exp_id", help="Experiment folder containing the config file")
@click.option("--debug", default=False, help="Debug mode")
@click.option("--cluster", default=False, help="On the cluster?")
@click.option("--resume_checkpoint", default=None, help="Resume training from checkpoint")
def train(exp_id, debug, cluster, resume_checkpoint):
    
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    exp_dir = join(ROOT_DIR, "experiments/training", exp_id)
    data_dir = join(ROOT_DIR, "data/input/")
    #save_dir = exp_dir
    
    
    # Create loggers
    logger_memory = logging.getLogger("logger_memory")
    logger_info = logging.getLogger("logger_info")

    # Create handlers
    memory_handler = logging.FileHandler(join(ROOT_DIR, mk_fname(filename="log", label="memory", suffix="log")))
    info_handler = logging.FileHandler(join(ROOT_DIR,  mk_fname(filename="log", label="train", suffix="log")))

    # Set different logging levels
    logger_info.setLevel(logging.INFO)
    logger_memory.setLevel(logging.INFO)

    # Define a log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Apply formatters to handlers
    memory_handler.setFormatter(formatter)
    info_handler.setFormatter(formatter)

    # Attach handlers to loggers
    logger_memory.addHandler(memory_handler)
    logger_info.addHandler(info_handler)
    
    
    
    @independent_sweep(exp_dir)
    def run_sweep(config,save_dir):
        kfold_train(
            config = config, 
            data_dir = data_dir, 
            save_dir = save_dir, 
            cluster = cluster, 
            resume_ckpt = resume_checkpoint, 
            debug = debug)
    
    run_sweep()


cli.add_command(train)

if __name__ == "__main__":
    cli()