from os import listdir, makedirs
from os.path import join, isdir, abspath, dirname, exists
from typing import List, Callable
import pandas as pd
from pathlib import Path



def has_logs_subfolder(directory: str) -> bool:
    """
    Check if the given directory contains at least one logs subfolder
    """
    directory_path = Path(directory)
    target_folder = "logs"

    return any(subdir.name == target_folder for subdir in directory_path.iterdir() if subdir.is_dir())




def get_df_recursive(filepath: str, bottom_action: Callable, is_bottom: Callable, lev: int=0)->pd.DataFrame:
    """
    Loops recursively inside folders, keeping track of the various levels, until the bottom is reached
    At the bottom, performs the bottom_action.
    
    N.B. The condition for the bottom is hard-coded

    Args:
        filepath (str): level path, if user input, starting level
        bottom_action (Callable): function to perform at the bottom level
        lev (int, optional): Current level, leave default value. Defaults to 0.

    Returns:
        pd.DataFrame: multi-level dataframe
    """
    
    
    level_folders = [d for d in listdir(filepath) if isdir(join(filepath,d))]
    
    # check if bottom level is reached, condition might change for other applications
    if is_bottom(join(filepath,level_folders[0])):
        df = bottom_action(filepath, level_folders)
    
    else:
        
        # init dataframe
        df = None
        
        # loop over the sweep folders
        for case in level_folders:
            
            # update internal filepath
            filepath_ = join(filepath,case)
            
            # recursive call
            df_temp = get_df_recursive(filepath_, bottom_action, is_bottom, lev+1)
            
            # update sweep colums
            df_temp[f"level_{lev}"] = case

            # append to df
            df = df_temp if df is None else pd.concat([df,df_temp])

    return df



def get_df_kfold_loss(filepath: str, level_folders: List[str])->pd.DataFrame:
    """
    Loop over k_fold folders, looking for the final loss

    Args:
        filepath (str): path to k_fold folders
        level_folders (List[str]): folder names to loop for

    Returns:
        pd.DataFrame: _description_
    """
    
    # needed later
    def get_last_value(df: pd.DataFrame, col: str):
        return df[df[col].notna()][col].to_numpy()[-1]
    
    # subpath to the cvs log file
    subpath = "logs/csv/version_0/metrics.csv"
    
    cols = {
        "k": [],
        "val_loss": [],
        "train_loss": [],
        # "test_loss": []
        }
    
    # loop over the bottom folders
    for case in level_folders:
        
        # update internal filepath
        filepath_ = join(filepath,case,subpath)
    
        df = pd.read_csv(filepath_)
        
        # save losses in the cols dictionary
        for col in cols.keys():
            if col != "k":
                cols[col].append(get_last_value(df,col))
    
        # save current k to cols dictionary
        cols["k"].append(case)
    
    return pd.DataFrame().from_dict(cols)


    
    
    
if __name__ == "__main__":
    
    exp_id = "paper"
    eval_id = "paper/full_field_sweep"
    ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
    filepath = join(ROOT_DIR,"experiments","training",exp_id)
    outpath = join(ROOT_DIR,"experiments","evaluations",eval_id)
    
    if not(exists(outpath)):
        makedirs(outpath)
        
    df = get_df_recursive(
        filepath=filepath, 
        bottom_action=get_df_kfold_loss,
        is_bottom=has_logs_subfolder)
    
    df.to_csv(join(outpath,"eval_sweeps.csv"))
    
    
    