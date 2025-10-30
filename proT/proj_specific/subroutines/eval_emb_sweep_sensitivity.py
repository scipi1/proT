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
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)
from proT.subroutines.eval_sweeps import get_df_recursive, has_logs_subfolder, get_df_kfold_loss
from proT.predict import get_features_gradients
from proT.subroutines.feat_grad import get_emb_gradients
from proT.modules.utils import find_last_checkpoint


root_path = ROOT_DIR

def get_df_kfold_sensitivity(filepath: str, level_folders: List[str], s3: bool = False, bucket: str = None) -> pd.DataFrame:
    """
    Loop over k_fold folders, looking for the final loss in logs/csv/version_0/metrics.csv

    Args:
        filepath (str): path or prefix to k_fold folders
        level_folders (List[str]): names of subfolders (e.g. k0, k1, ...)
        s3 (bool): if True, read from S3
        bucket (str): S3 bucket name

    Returns:
        pd.DataFrame: dataframe with val/train/test losses per k-fold
    """
    print(filepath)
    
    config_path = join(filepath, "config.yaml")
    datadir_path = join(root_path,"data","input")

    subpath = "checkpoints"

    cols = None

    s3_client = boto3.client("s3") if s3 else None

    for case in level_folders:
        print(case)
        checkpoint_path = find_last_checkpoint(join(filepath, case, subpath).replace("\\", "/"))
        
        
        # if s3:
        #     pass
        #     # try:
        #     #     obj = s3_client.get_object(Bucket=bucket, Key=full_path)
        #     #     df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
        #     # except Exception as e:
        #     #     print(f"Failed to read {full_path} from S3: {e}")
        #     #     continue
        # else:
        #     try:
        #         df = pd.read_csv(full_path)
        #     except Exception as e:
        #         print(f"Failed to read {full_path} from disk: {e}")
        #         continue
            
            
        S = get_emb_gradients(
            config_path,
            datadir_path, 
            checkpoint_path, 
            )
        
        # initialize cols
        if cols is None:
            cols = {key:[] for key in S.keys()}
            cols["k"] = []
            

        for key in cols.keys():
            if key != "k":
                cols[key].append(S[key])

        cols["k"].append(case)

    return pd.DataFrame(cols)


if __name__ == "__main__":

    dirpath = join(ROOT_DIR,"experiments/training/dx_250618_sum")
    save_dirpath = join(ROOT_DIR,"experiments/evaluations/dx_250618_sum/sensitivity")
    
    if not exists(save_dirpath):
        makedirs(save_dirpath)
    
    df_S = get_df_recursive(filepath=dirpath, bottom_action=get_df_kfold_sensitivity, is_bottom= has_logs_subfolder)
    df_S.to_csv(join(save_dirpath,"df_S"))