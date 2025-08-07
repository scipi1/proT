import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from os.path import join, isdir, dirname, abspath, exists
from os import listdir, makedirs
from typing import List
from pathlib import Path
import boto3
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import sys
import torch
from pytorch_lightning import seed_everything
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)
from proT.subroutines.eval_sweeps import get_df_recursive, has_logs_subfolder, get_df_kfold_loss
from proT.predict import get_features_gradients
from proT.subroutines.feat_grad import get_emb_gradients
from proT.modules.utils import find_last_checkpoint
root_path = dirname(dirname(abspath(__file__)))
from proT.forecaster import TransformerForecaster
from proT.labels import *
from proT.experiment_control import update_config


root_path = ROOT_DIR

def get_df_params_count(filepath: str, level_folders: List[str], s3: bool = False, bucket: str = None) -> pd.DataFrame:
    """
    Loads models and count its parameters. 
    Selects the first k_fold folder (usually k0) and makes the count.
    All k-folder have the same parameter numbers
    Args:
        filepath (str): path or prefix to k_fold folders
        level_folders (List[str]): names of subfolders (e.g. k0, k1, ...)
        s3 (bool): if True, read from S3
        bucket (str): S3 bucket name

    Returns:
        pd.DataFrame: dataframe with parameters count for any of the k folder
    """
    print(filepath)
    
    config_path = join(filepath, "config.yaml")

    subpath = "checkpoints"

    

    s3_client = boto3.client("s3") if s3 else None
    
    k0 = level_folders[0]
    checkpoint_path = find_last_checkpoint(join(filepath, k0 , subpath).replace("\\", "/"))

    # for case in level_folders:
    #     print(case)
    #     checkpoint_path = find_last_checkpoint(join(filepath, case, subpath).replace("\\", "/"))
        
        
    #     # if s3:
    #     #     pass
    #     #     # try:
    #     #     #     obj = s3_client.get_object(Bucket=bucket, Key=full_path)
    #     #     #     df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
    #     #     # except Exception as e:
    #     #     #     print(f"Failed to read {full_path} from S3: {e}")
    #     #     #     continue
    #     # else:
    #     #     try:
    #     #         df = pd.read_csv(full_path)
    #     #     except Exception as e:
    #     #         print(f"Failed to read {full_path} from disk: {e}")
    #     #         continue
            
            
    cols = get_params_count(
        config_path,
        checkpoint_path, 
        )
        
        # # initialize cols
        # if cols is None:
        #     cols = {key:[] for key in S.keys()}
        #     cols["k"] = []
            

        # for key in cols.keys():
        #     if key != "k":
        #         cols[key].append(S[key])

        # cols["k"].append(case)

    return pd.DataFrame([cols])



def get_params_count(
    config_path: Path,
    checkpoint_path: Path,
    )->dict:
    config = OmegaConf.load(config_path) # load config
    
    config_updated = update_config(config)                      # update config
    model = TransformerForecaster(config_updated)               # load model
    forecaster = model.load_from_checkpoint(checkpoint_path)    # load checkpoint
    
    
    # Count total parameters
    total_params = sum(p.numel() for p in forecaster.parameters())
    trainable_params = sum(p.numel() for p in forecaster.parameters() if p.requires_grad)
    
    # Count parameters using the model's split_params method
    try:
        group_1, group_2 = forecaster.split_params()
        group_1_params = sum(p.numel() for p in group_1)
        group_2_params = sum(p.numel() for p in group_2)
    except Exception as e:
        # Fallback if split_params fails
        print(f"Warning: split_params failed with error: {e}")
        group_1_params = 0
        group_2_params = total_params
    
    # Count embedding parameters separately for additional insight
    try:
        enc_emb_params = sum(p.numel() for p in forecaster.model.enc_embedding.embed_modules_list.parameters())
        dec_emb_params = sum(p.numel() for p in forecaster.model.dec_embedding.embed_modules_list.parameters())
        total_emb_params = enc_emb_params + dec_emb_params
    except Exception as e:
        print(f"Warning: embedding parameter counting failed with error: {e}")
        enc_emb_params = 0
        dec_emb_params = 0
        total_emb_params = 0
    
    # Return parameter counts as dictionary
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "group_1_params": group_1_params,
        "group_2_params": group_2_params,
        "enc_emb_params": enc_emb_params,
        "dec_emb_params": dec_emb_params,
        "total_emb_params": total_emb_params
    }







if __name__ == "__main__":

    dirpath = join(ROOT_DIR,"experiments/training/dx_250618_stformer")
    save_dirpath = join(ROOT_DIR,"experiments/evaluations/dx_250618_stformer/model_size")
    
    if not exists(save_dirpath):
        makedirs(save_dirpath)
    
    df = get_df_recursive(filepath=dirpath, bottom_action=get_df_params_count, is_bottom= has_logs_subfolder)
    
    df.to_csv(join(save_dirpath,"df_params_count"))
