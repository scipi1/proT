import os
from os.path import join, dirname, abspath
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch
import logging
from pathlib import Path
import json, datetime, pytorch_lightning as pl
import time
import sys
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)



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