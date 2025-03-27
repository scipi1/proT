import os
from os.path import join, dirname, abspath
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from pytorch_lightning import Callback
import torch
import logging

early_stopping_callbacks = EarlyStopping(
    monitor="val_loss", 
    min_delta=0.00, 
    patience=100, 
    verbose=False, 
    mode="min")


def get_checkpoint_callback(experiment_dir):
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=5, 
        save_top_k=-1,
        dirpath = checkpoint_dir,
        filename = "{epoch}-{train_loss:.2f}",
        monitor = "val_loss",
        mode = "min")
    
    return checkpoint_callback


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