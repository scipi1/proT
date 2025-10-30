# Standard library imports
import datetime
import json
import logging
import os
import sys
import time
from os.path import join, dirname, abspath
from pathlib import Path

# Third-party imports
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# Local imports
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)
from proT.utils.entropy_utils import get_entropy_registry



class PerRunManifest(pl.Callback):
    def __init__(self, config, path, tag=""):
        self.config = config
        self.tag    = tag
        self.path   = path
        self.manifest_f = Path(ROOT_DIR) / "logs" / "manifest.ldjson"
        self.record = None
    # helpers
    def _gather_common(self):
        return {
            "timestamp" : datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z",
            "model"     : self.config["model"]["model_object"],
            "dataset"   : self.config["data"]["dataset"],    
            "tag"       : self.tag,
            "path"      : self.path
        }

    # def _write_manifest(self, new_fields: dict):
    #     log_dir = join(ROOT_DIR,"logs")
    #     info_f  = Path(join(log_dir,"manifest.json"))

    #     # load previous if exists
    #     if info_f.exists():
    #         with open(info_f) as f:
    #             entry = json.load(f)
    #     else:
    #         entry = {}
    #     entry.update(self._gather_common())  
    #     entry.update(new_fields)                    

    #     with open(info_f, "w") as f:
    #         json.dump(entry, f, indent=2)
            
    def _append(self, fields: dict):
        if self.record is None:
            self.record = {**self._gather_common(), **fields}
        else:
            self.record.update(fields)
        self.manifest_f.parent.mkdir(parents=True, exist_ok=True)
        
    def _write_manifest(self):
        with open(self.manifest_f, "a") as f:               # APPEND mode
            f.write(json.dumps(self.record, default=str) + "\n")
    
    
    def _elapsed(self):
        return time.time() - getattr(self, "_fit_start_time", time.time())

    
    # lightning hooks
    def on_fit_start(self,trainer, pl_module):
        self._fit_start_time = time.time()
    
    def on_fit_end(self, trainer, pl_module):
        m = trainer.logged_metrics
        epochs_run = trainer.current_epoch
        self._append({
            "val_loss"      : float(m.get("val_loss", float("nan"))),
            "val_mae"       : float(m.get("val_mae",  float("nan"))),
            "val_r2"        : float(m.get("val_r2",   float("nan"))),
            "val_rmse"      : float(m.get("val_rmse", float("nan"))),
            "train_seconds" : round(self._elapsed(), 2),
            "epochs"        : epochs_run,
        })

    def on_test_end(self, trainer, pl_module):
        m = trainer.logged_metrics
        self._append({
            "test_loss" : float(m.get("test_loss", float("nan"))),
            "test_mae"  : float(m.get("test_mae", float("nan"))),
            "test_r2"   : float(m.get("test_r2", float("nan"))),
            "test_rmse" : float(m.get("test_rmse", float("nan")))
        })
        self._write_manifest()


early_stopping_callbacks = EarlyStopping(
    monitor="val_mae", # what to watch
    min_delta=1E-5,    # improvement threshold
    patience=50,       # epochs to wait
    verbose=True, 
    mode="min"         # lower is better
    )         


def get_checkpoint_callback(experiment_dir: str, save_ckpt_every_n_epochs: int):

    checkpoint_dir = join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── 1. periodic checkpoints ────────────────────────────────────────────────
    periodic_ckpt = ModelCheckpoint(
        dirpath     = checkpoint_dir,
        filename    = "{epoch}-{train_loss:.2f}",
        every_n_epochs = save_ckpt_every_n_epochs,  # unchanged
        save_top_k = -1,
        monitor    = "val_loss",
        mode       = "min",
    )

    # ── 2. one-off checkpoint right at the start ──────────────────────────────
    class SaveInitial(Callback):
        """Dump weights before the first optimization step."""
        @rank_zero_only             # avoid DDP duplicates
        def on_fit_start(self, trainer, pl_module):
            trainer.save_checkpoint(join(checkpoint_dir, "epoch0-initial.ckpt"))

    # return both callbacks so the Trainer can register them
    return [SaveInitial(), periodic_ckpt]


class MemoryLoggerCallback(Callback):
    
    def log_memory(self, stage):
        """Logs CPU & GPU memory usage."""
        allocated_gpu = torch.cuda.memory_allocated() / 1e9  # GB
        reserved_gpu = torch.cuda.memory_reserved() / 1e9  # GB
        # ram_usage = psutil.virtual_memory().used / 1e9  # GB
        # ram_total = psutil.virtual_memory().total / 1e9  # GB
        # ram_percent = psutil.virtual_memory().percent  # %
        logger_memory = logging.getLogger("logger_memory")
        logger_memory.info(
            f"[{stage}] GPU Allocated: {allocated_gpu:.2f} GB | GPU Reserved: {reserved_gpu:.2f} GB | "
            # f"CPU Used: {ram_usage:.2f}/{ram_total:.2f} GB ({ram_percent}%)"
        )

    def on_train_start(self, trainer, pl_module):
        self.log_memory("TRAIN START")

    def on_train_epoch_start(self, trainer, pl_module):
        self.log_memory(f"EPOCH {trainer.current_epoch} START")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.log_memory(f"BATCH {batch_idx} START")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_memory(f"BATCH {batch_idx} END")

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_memory(f"EPOCH {trainer.current_epoch} END")

    def on_train_end(self, trainer, pl_module):
        self.log_memory("TRAIN END")
        # Log max memory usage at the end
        logger_memory = logging.getLogger("logger_memory")
        logger_memory.info(f"Max allocated GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        logger_memory.info(f"Max reserved GPU: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
        
        
        
class GradientLogger(Callback):
    """
    Logs ‖∇θ‖₂ and variance layer‑by‑layer.
    Optionally stores raw gradients as .pt files.
    """
    def __init__(self):
        super().__init__()        

    @staticmethod
    def _stats(t: torch.Tensor):
        return dict(
            grad_norm = t.norm().item(),
            #grad_var  = t.var(unbiased=False).item()
        )

    # # Lightning fires this hook *after* every backward pass
    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        
        
    #     for name, p in pl_module.named_parameters():
    #         if p.grad is None:                       # frozen param
    #             continue
    #         s = self._stats(p.grad.detach())
    #         self.metrics[f"grad_norm/{name}"] = s["grad_norm"]
    #         #metrics[f"grad_var/{name}"]  = s["grad_var"]
    
    
    #     pl_module.log_dict(metrics = {}
        self.metrics = {}
        for name, p in pl_module.named_parameters():
            if p.grad is None:                       # frozen param
                continue
            s = self._stats(p.grad.detach())
            self.metrics[f"grad_norm/{name}"] = s["grad_norm"]
    
    def on_train_epoch_end(self, trainer, pl_module):
        
        # metrics = {}
        # for name, p in pl_module.named_parameters():
        #     if p.grad is None:                       # frozen param
        #         continue
        #     s = self._stats(p.grad.detach())
        #     metrics[f"grad_norm/{name}"] = s["grad_norm"]
        
        pl_module.log_dict(
                self.metrics,
                on_step=False,
                on_epoch=True
            )
        

class MetricsAggregator(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        trainer.logger.log_metrics(
            trainer.callback_metrics,           # one dense dict
            step=trainer.current_epoch
        )


class AttentionEntropyLogger(Callback):
    """
    Logs attention entropy statistics collected during forward passes.
    """
    
    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.entropy_registry = get_entropy_registry()
    
    def on_train_start(self, trainer, pl_module):
        """Enable entropy collection at training start."""
        if self.enabled:
            self.entropy_registry.enable()
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log entropy statistics and clear registry."""
        if not self.enabled:
            return
            
        # Get aggregated entropy statistics
        entropy_stats = self.entropy_registry.get_aggregated_entropy()

        if entropy_stats:
            # Log all entropy metrics
            pl_module.log_dict(
                entropy_stats,
                on_step=False,
                on_epoch=True
            )
        
        # Clear registry for next epoch
        self.entropy_registry.clear()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log entropy statistics for validation and clear registry."""
        if not self.enabled:
            return
            
        # Get aggregated entropy statistics
        entropy_stats = self.entropy_registry.get_aggregated_entropy()
        
        if entropy_stats:
            # Add 'val_' prefix to distinguish from training entropy
            val_entropy_stats = {f"val_{k}": v for k, v in entropy_stats.items()}
            pl_module.log_dict(
                val_entropy_stats,
                on_step=False,
                on_epoch=True
            )
        
        # Clear registry for next epoch
        self.entropy_registry.clear()
    
    def on_test_epoch_end(self, trainer, pl_module):
        """Log entropy statistics for testing and clear registry."""
        if not self.enabled:
            return
            
        # Get aggregated entropy statistics
        entropy_stats = self.entropy_registry.get_aggregated_entropy()
        
        if entropy_stats:
            # Add 'test_' prefix to distinguish from training entropy
            test_entropy_stats = {f"test_{k}": v for k, v in entropy_stats.items()}
            pl_module.log_dict(
                test_entropy_stats,
                on_step=False,
                on_epoch=True
            )
        
        # Clear registry
        self.entropy_registry.clear()


class LayerRowStats(Callback):
    
    def __init__(self,layer_name: str=None):
        super().__init__()
        
        self.layer_name = layer_name
        self.layer_index = None
        supported_layers = ["encoder_variable", "encoder_position", "final_ff", None]
        
        assert layer_name in supported_layers, AssertionError(f"layer_name must be one of {supported_layers}, but got {layer_name}")
        
        if layer_name == "encoder_variable" or layer_name is None:
            self.layer_fn = lambda m: m.model.enc_embedding.embed_modules_list[0].embedding.embedding.weight
            self.layer_index = 2
            
        elif layer_name == "encoder_position":
            self.layer_fn = lambda m: m.model.enc_embedding.embed_modules_list[1].embedding.embedding.weight
            self.layer_index = 3
            
        elif layer_name == "final_ff":
            self.layer_fn = lambda m: m.model.forecaster.weight
    
    
    def on_train_start(self, trainer, pl_module):
        # save the initial weight
        embedding = self.layer_fn(pl_module)
        self.E0 = embedding.detach().clone()
    
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.seen_rows = set()
        self.miss_val_rows = set()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        
        X,_ = batch
        
        if self.layer_index is not None:
            
            value_index = 4
            val_nan_ids = X[:,:,value_index].isnan()           # index of the rows with missing values
            layer_ids = X[:,:,self.layer_index]                # lookup table index
            layer_ids_val_nan = layer_ids[val_nan_ids]         # index of the lookup table with missing values
            
            self.seen_rows.update(torch.nan_to_num(layer_ids).unique().tolist())
            self.miss_val_rows.update(torch.nan_to_num(layer_ids_val_nan).unique().tolist())


    def on_train_epoch_end(self, trainer, pl_module):
        
        def _mean_abs(grad_subset)-> float:
            """
            Calculates the mean of the absolute values of the gradient.
            If the gradient is sparse, it returns the mean of the non-zero entries.
            """
            if grad_subset.is_sparse:
                # sparse coo: values() are the non-zero entries
                return grad_subset.coalesce().values().abs().mean()
            else:
                return grad_subset.abs().mean()
            
        def _make_label(label):
            return self.layer_name + "_" + label if self.layer_name is not None else label
        
        
        # get the embedding layer
        E = self.layer_fn(pl_module)
                
        # whole embedding
        dF = (E - self.E0).norm() / self.E0.norm()
        
        # lookup tables
        if self.layer_index is not None:
            seen_rows = torch.tensor(list(self.seen_rows), device=E.device, dtype=torch.long)
            miss_val_rows = torch.tensor(list(self.miss_val_rows), device=E.device, dtype=torch.long)

            # active rows
            if seen_rows.numel() > 0:
                drift_act = (E[seen_rows] - self.E0[seen_rows]).norm() / self.E0[seen_rows].norm()

            else:
                drift_act = torch.tensor(0., device=E.device)
                

            # inactive rows
            mask_inactive = torch.ones(E.size(0), dtype=torch.bool, device=E.device)
            mask_inactive[seen_rows] = False

            if seen_rows.numel() > 0:
                inactive = E[mask_inactive]
                E0_inact = self.E0[mask_inactive.cpu()].to(E.device)
                drift_inact = (inactive - E0_inact).norm() / E0_inact.norm()

            else:
                drift_inact = torch.tensor(0., device=E.device)

            # rows with missing values
            mask_miss_val = torch.zeros(E.size(0), dtype=torch.bool, device=E.device)
            mask_miss_val[miss_val_rows] = True

            if miss_val_rows.numel() > 0:
                miss_val = E[mask_miss_val]
                E0_miss_val = self.E0[miss_val_rows.cpu()].to(E.device)
                drift_miss_val = (miss_val - E0_miss_val).norm() / E0_miss_val.norm()

            else:
                drift_miss_val = torch.tensor(0., device=E.device)

            # log the metrics
            pl_module.log_dict(
                {
                    _make_label("whole_drift")       : dF.item(),
                    _make_label("row_drift_active")  : drift_act.item(),
                    _make_label("row_drift_inactive"): drift_inact.item(),
                    _make_label("row_drift_miss_val"): drift_miss_val.item(),
                },
                on_step=False,
                on_epoch=True
            )
            
        else:
            pl_module.log_dict(
                {
                    _make_label("whole_drift")       : dF.item(),
                },
                on_step=False,
                on_epoch=True
            )


class BestCheckpointCallback(Callback):
    """
    Callback to save the best checkpoint based on val_mae and store associated metrics.
    """
    
    def __init__(self, save_dir: str, monitor: str = "val_mae", mode: str = "min"):
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_metric_value = float('inf') if mode == 'min' else float('-inf')
        self.best_metrics = {}
        self.best_epoch = 0
        self.best_checkpoint_path = None
        
        # Create checkpoints directory
        self.checkpoint_dir = join(save_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _is_better(self, current_value):
        """Check if current metric value is better than the best so far."""
        if self.mode == 'min':
            return current_value < self.best_metric_value
        else:
            return current_value > self.best_metric_value
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Check if current epoch has the best validation metric and save if so."""
        current_metrics = trainer.logged_metrics
        
        if self.monitor in current_metrics:
            current_value = float(current_metrics[self.monitor])
            
            if self._is_better(current_value):
                self.best_metric_value = current_value
                self.best_epoch = trainer.current_epoch
                
                # Store all current metrics (validation only at this point)
                self.best_metrics = {
                    key: float(value) if isinstance(value, torch.Tensor) else value
                    for key, value in current_metrics.items()
                }
                
                # Save the best checkpoint
                self.best_checkpoint_path = join(self.checkpoint_dir, "best_checkpoint.ckpt")
                trainer.save_checkpoint(self.best_checkpoint_path)
    
    def on_test_end(self, trainer, pl_module):
        """Save the final best metrics including test metrics after testing is complete."""
        if self.best_metrics:  # Only save if we have best metrics
            # Get current test metrics and add them to best metrics
            current_metrics = trainer.logged_metrics
            test_metrics = {k: float(v) for k, v in current_metrics.items() if k.startswith('test_')}
            
            # Update best metrics with test results
            final_best_metrics = {**self.best_metrics, **test_metrics}
            
            # Save best metrics to JSON file
            best_metrics_path = join(self.save_dir, "best_metrics.json")
            metrics_to_save = {
                **final_best_metrics,
                "best_epoch": self.best_epoch,
                "best_checkpoint_path": self.best_checkpoint_path
            }
            
            with open(best_metrics_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)


class DataIndexTracker(Callback):
    """
    Callback to save train/validation/test data indices for each fold.
    """
    
    def __init__(self, save_dir: str, fold_num: int, train_idx, val_idx, test_idx):
        super().__init__()
        self.save_dir = save_dir
        self.fold_num = fold_num
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
    
    def on_fit_start(self, trainer, pl_module):
        """Save data indices at the start of training."""
        # Save train indices
        train_indices_path = join(self.save_dir, f"fold_{self.fold_num}_train_indices.npy")
        np.save(train_indices_path, self.train_idx)
        
        # Save validation indices
        val_indices_path = join(self.save_dir, f"fold_{self.fold_num}_val_indices.npy")
        np.save(val_indices_path, self.val_idx)
        
        # Save test indices
        test_indices_path = join(self.save_dir, f"fold_{self.fold_num}_test_indices.npy")
        np.save(test_indices_path, self.test_idx)


class KFoldResultsTracker:
    """
    Class to track and aggregate results across all k-folds.
    """
    
    def __init__(self, save_dir: str, k_folds: int):
        self.save_dir = save_dir
        self.k_folds = k_folds
        self.fold_results = {}
        self.summary_file = join(save_dir, "kfold_summary.json")
    
    def add_fold_result(self, fold_num: int, metrics: dict, best_checkpoint_path: str = None):
        """Add results for a specific fold."""
        self.fold_results[fold_num] = {
            "metrics": metrics,
            "best_checkpoint_path": best_checkpoint_path,
            "fold_dir": join(self.save_dir, f"k_{fold_num}")
        }
        
        # Update summary file after each fold
        self._update_summary()
    
    def _update_summary(self):
        """Update the k-fold summary file."""
        if not self.fold_results:
            return
        
        # Calculate statistics across folds
        metric_names = list(next(iter(self.fold_results.values()))["metrics"].keys())
        summary = {
            "total_folds": self.k_folds,
            "completed_folds": len(self.fold_results),
            "fold_results": self.fold_results,
            "statistics": {}
        }
        
        # Calculate mean, std, min, max for each metric
        for metric_name in metric_names:
            values = [self.fold_results[fold]["metrics"][metric_name] 
                     for fold in self.fold_results.keys()]
            
            if values and all(isinstance(v, (int, float)) for v in values):
                summary["statistics"][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        # Find best fold based on val_mae (lower is better)
        if "val_mae" in metric_names:
            best_fold = min(self.fold_results.keys(), 
                           key=lambda x: self.fold_results[x]["metrics"]["val_mae"])
            summary["best_fold"] = {
                "fold_number": best_fold,
                "val_mae": self.fold_results[best_fold]["metrics"]["val_mae"],
                "metrics": self.fold_results[best_fold]["metrics"],
                "checkpoint_path": self.fold_results[best_fold]["best_checkpoint_path"]
            }
        
        # Save summary to file
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def get_summary(self):
        """Get the current summary of all folds."""
        if os.path.exists(self.summary_file):
            with open(self.summary_file, 'r') as f:
                return json.load(f)
        return {}
