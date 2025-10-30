from os import listdir, makedirs
from os.path import join, isdir, abspath, dirname, exists
from typing import List, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import boto3
from io import StringIO



# def has_logs_subfolder(directory: str, s3: bool=False, bucket: str=None) -> bool:
#     """
#     Check if the given directory contains at least one logs subfolder
#     """
#     directory_path = Path(directory)
#     target_folder = "logs"

#     return any(subdir.name == target_folder for subdir in directory_path.iterdir() if subdir.is_dir())


def has_logs_subfolder(directory: str, s3: bool = False, bucket: str = None) -> bool:
    """
    Check if the given directory contains at least one logs subfolder
    """
    target_folder = "logs"

    if s3:
        s3_client = boto3.client("s3")
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=directory, Delimiter='/')

        for page in pages:
            for prefix in page.get("CommonPrefixes", []):
                if prefix["Prefix"].rstrip('/').endswith(target_folder):
                    return True
        return False

    else:
        directory_path = Path(directory)
        return any(subdir.name == target_folder for subdir in directory_path.iterdir() if subdir.is_dir())



def get_df_recursive(filepath: str, bottom_action: Callable, is_bottom: Callable, s3: bool=False, bucket: str=None, lev: int=0)->pd.DataFrame:
    """
    Loops recursively inside folders, keeping track of the various levels, until the bottom is reached
    At the bottom, performs the bottom_action.

    Args:
        filepath (str): level path, if user input, starting level
        bottom_action (Callable): function to perform at the bottom level
        is_bottom (Callable): function to check if a given layer is the bottom one
        s3 (bool): AWS s3 flag
        lev (int, optional): Current level, leave default value. Defaults to 0.

    Returns:
        pd.DataFrame: multi-level dataframe
    """
    
    
    if s3:
        s3_client = boto3.client("s3")
        # List all "directories" one level under the current prefix
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=filepath, Delimiter='/')

        level_folders = []
        for page in pages:
            for prefix in page.get("CommonPrefixes", []):
                level_folders.append(prefix["Prefix"].rstrip('/').split('/')[-1])
        
        if not level_folders:
            return pd.DataFrame()  # empty folder

        # Construct full path for the first to check if it's bottom
        first_subfolder = f"{filepath.rstrip('/')}/{level_folders[0]}/"
        if is_bottom(first_subfolder, s3=s3, bucket=bucket):
            df = bottom_action(filepath, level_folders, s3=s3, bucket=bucket)
            
        else:
            df = None
            for case in level_folders:
                subpath = f"{filepath.rstrip('/')}/{case}/"
                df_temp = get_df_recursive(subpath, bottom_action, is_bottom, s3=s3, bucket=bucket, lev=lev+1)
                if df_temp is not None:
                    df_temp[f"level_{lev}"] = case
                    df = df_temp if df is None else pd.concat([df, df_temp])
    
    
    else:
        
        # get all folders of current level
        level_folders = [d for d in listdir(filepath) if isdir(join(filepath,d))]

        # check if bottom level is reached, condition might change for other applications
        if is_bottom(join(filepath,level_folders[0])):
            df = bottom_action(filepath, level_folders, s3=s3, bucket=bucket)

        else:

            # init dataframe
            df = None

            # loop over the sweep folders
            for case in level_folders:

                # update internal filepath
                filepath_ = join(filepath,case)

                # recursive call
                df_temp = get_df_recursive(filepath_, bottom_action, is_bottom, s3=s3, bucket=bucket, lev = lev+1)

                # update sweep colums
                df_temp[f"level_{lev}"] = case

                # append to df
                df = df_temp if df is None else pd.concat([df,df_temp])

    return df





def get_df_kfold_loss(filepath: str, level_folders: List[str], s3: bool = False, bucket: str = None) -> pd.DataFrame:
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
    def get_last_value(df: pd.DataFrame, col: str):
        if col in df.columns:
            return df[col].dropna().to_numpy()[-1]
        else:
            return np.nan

    subpath = "logs/csv/version_0/metrics.csv"

    cols = {
        "k": [],
        "val_loss": [],
        "train_loss": [],
        "test_loss": [],
    }

    s3_client = boto3.client("s3") if s3 else None

    for case in level_folders:
        full_path = join(filepath, case, subpath).replace("\\", "/")

        if s3:
            try:
                obj = s3_client.get_object(Bucket=bucket, Key=full_path)
                df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
            except Exception as e:
                print(f"Failed to read {full_path} from S3: {e}")
                continue
        else:
            try:
                df = pd.read_csv(full_path)
            except Exception as e:
                print(f"Failed to read {full_path} from disk: {e}")
                continue

        for col in cols.keys():
            if col != "k":
                cols[col].append(get_last_value(df, col))

        cols["k"].append(case)

    return pd.DataFrame(cols)


    
    
    
if __name__ == "__main__":
    
    # exp_id = "paper"
    # eval_id = "paper/full_field_sweep"
    # ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
    # filepath = join(ROOT_DIR,"experiments","training",exp_id)
    # outpath = join(ROOT_DIR,"experiments","evaluations",eval_id)
    
    # if not(exists(outpath)):
    #     makedirs(outpath)
    
    filepath = "proT/optimizer_experiments/test_opt1_sum/"
    df = get_df_recursive(
        filepath=filepath, 
        bottom_action=get_df_kfold_loss,
        is_bottom=has_logs_subfolder,
        s3=True, bucket="experiments-private")
    
    
    