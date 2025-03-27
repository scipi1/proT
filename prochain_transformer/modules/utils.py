import torch
import logging
import os
from datetime import datetime
from pytorch_lightning import seed_everything

def set_seed(seed=42):
    """
    Sets the random seed across various libraries and enforces deterministic behavior.
    
    Parameters:
    seed (int): The random seed to use. Default is 42.
    """
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # For GPUs
    
    seed_everything(seed, workers=True)

    # Enforce deterministic operations in PyTorch
    torch.backends.cudnn.deterministic = True  # Ensures reproducible behavior in cuDNN
    torch.backends.cudnn.benchmark = False     # Disables benchmarking to avoid nondeterminism

    # Set environment variable to control other sources of randomness
    os.environ["PYTHONHASHSEED"] = str(seed)   # Controls hashing randomness in Python
    
    
    
def log_memory(stage):
    # GPU memory usage
    allocated_gpu = torch.cuda.memory_allocated() / 1e9  # GB
    reserved_gpu = torch.cuda.memory_reserved() / 1e9  # GB
    
    # # CPU memory usage
    # ram_usage = psutil.virtual_memory().used / 1e9  # GB
    # ram_total = psutil.virtual_memory().total / 1e9  # GB
    # ram_percent = psutil.virtual_memory().percent  # %

    logging.info(
        f"[{stage}] GPU Allocated: {allocated_gpu:.2f} GB | GPU Reserved: {reserved_gpu:.2f} GB | "
        # f"CPU Used: {ram_usage:.2f}/{ram_total:.2f} GB ({ram_percent}%)"
    )
    
    
def mk_fname(filename: str,label: str,suffix: str):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") # format YYYYMMDD_HHMMSS
    return filename+"_"+str(label)+f"_{timestamp}"+suffix