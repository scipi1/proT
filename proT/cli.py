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
from proT.training.experiment_control import find_yml_files
from proT.core.modules.utils import mk_fname, find_last_checkpoint
from proT.training.trainer import trainer


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
@click.option("--best/--no-best", default=True, help="Evaluate val/test on the best (val_mae) checkpoint instead of the final epoch")
@click.option("--fold", "folds", type=int, multiple=True,
              help="Run only this fold (repeatable, e.g. --fold 0 --fold 3); "
                   "omit for all folds. Use with SLURM arrays: --fold $SLURM_ARRAY_TASK_ID")
def train(exp_id, debug, cluster, exp_tag, scratch_path, resume_checkpoint, plot_pred_check, best, folds):
    
    # Get folders
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    
    print(exp_id)
    print(scratch_path)
    
    if scratch_path is None:
        exp_dir = join(ROOT_DIR, "experiments/", exp_id)
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
    
    # Load config file (ignoring sweep config)
    config, _ = find_yml_files(dir=exp_dir)
    
    # Run training once with the loaded config
    trainer(
        config=config, 
        data_dir=data_dir, 
        save_dir=exp_dir, 
        cluster=cluster,
        experiment_tag=exp_tag,
        resume_ckpt=resume_checkpoint,
        plot_pred_check=plot_pred_check,
        debug=debug,
        best=best,
        folds=list(folds) if folds else None)


# AGGREGATION (parallel folds)
@click.command()
@click.option("--exp_id", default=None, help="Experiment folder containing the config file")
@click.option("--scratch_path", default=None, help="Run folder (overrides exp_id)")
def aggregate(exp_id, scratch_path):
    """Rebuild kfold_summary.json from all k_<n>/fold_result.json files."""
    from proT.training.callbacks.training_callbacks import KFoldResultsTracker
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    exp_dir = scratch_path if scratch_path else join(ROOT_DIR, "experiments", exp_id)
    config, _ = find_yml_files(dir=exp_dir)
    tracker = KFoldResultsTracker(
        exp_dir, config["training"]["k_fold"],
        best_metric=config["training"].get("best_checkpoint_metric", "val_mae"))
    tracker._update_summary()
    s = tracker.get_summary()
    print(f"folds completed: {s.get('completed_folds', 0)}/{config['training']['k_fold']}, "
          f"best fold: {s.get('best_fold', {}).get('fold_number')}")


cli.add_command(train)
cli.add_command(aggregate)

if __name__ == "__main__":
    cli()
