import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime
import pytorch_lightning as pl
from os.path import dirname, abspath, join,exists
from os import makedirs
import sys
from typing import List

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)


from proT.forecaster import TransformerForecaster
from proT.dataloader import ProcessDataModule
from proT.predict import predict
from proT.labels import *
from proT.old_.config import load_config


# # enforce deterministic behavior
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def mk_missing_folders(folders):
    for folder in folders:
        if not exists(folder):
            makedirs(folder)
            print(f"Created folder: {folder}")



def mk_fname(filename: str,label: str,suffix: str):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") # format YYYYMMDD_HHMMSS
    return filename+"_"+str(label)+f"_{timestamp}"+suffix




def save_output(src_trg_list: List[tuple], output_path: str)->None:
    """
    saves multiple files in a common destination
    Args:
        src_trg_list List(tuple): list of tuples with (file,file_name)
        output_path (str): output destination where the files are saved
    """

    for tuple in src_trg_list:
        source = tuple[0]
        trg_filename = tuple[1]

        with open(join(output_path, trg_filename), "wb") as f:
            np.save(f, source)




def load_model(exp_id: str, checkpoint: str, standard_path: bool=True):

    if standard_path:
        ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
        _, _, _, EXPERIMENTS_DIR = get_dirs(ROOT_DIR)
        EXP_DIR = join(EXPERIMENTS_DIR, exp_id)
        checkpoint_path = join(EXP_DIR, "checkpoints", checkpoint)
        config = load_config(EXP_DIR)
    else:
        checkpoint_path = checkpoint
        config = load_config(exp_id)

    model = TransformerForecaster(config)
    model_resumed = model.load_from_checkpoint(checkpoint_path)

    return model_resumed




def load_dataset(exp_id: str, standard_path: bool = True):

    if standard_path:
        ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
        INPUT_DIR, _, _, EXPERIMENTS_DIR = get_dirs(ROOT_DIR)
        EXP_DIR = join(EXPERIMENTS_DIR, exp_id)
        config = load_config(EXP_DIR)
    else:
        config = load_config(exp_id)

    dm = ProcessDataModule(
        ds_flag=config["data"]["ds_flag"],
        data_dir=join(INPUT_DIR, config["data"]["dataset"]),
        files=("X_np.npy", "Y_np.npy"),  # move to config
        batch_size=config["training"]["batch_size"],
        num_workers=20,
    )

    dm.prepare_data()
    dm.setup(stage=None)

    return dm




def predict_save(
    model: pl.LightningModule,
    dm: pl.LightningDataModule,
    dataset_label: str,
    input_mask: torch.Tensor,
    debug_flag: bool,
    zero_flag: bool,
    dynamic_kwargs: dict,
    save_label: str,
    output_path:str):

    input_array, pred_val, true_val, cross_att, _ = predict(
        model=model,
        dm=dm,
        dataset_label=dataset_label,
        input_mask=input_mask,
        debug_flag=debug_flag,
        kwargs=dynamic_kwargs)
    
    save_output(
        src_trg_list=[
            (input_array, mk_fname("input",save_label,".npy")),
            (pred_val, mk_fname("output",save_label,".npy")),
            (true_val, mk_fname("trg",save_label,".npy")),
            (cross_att, mk_fname("att",save_label,".npy")),
        ],
    output_path=output_path
    )




def get_masks_from_template(template_path=None,exp_id=None):

    if template_path is None and exp_id is not None:
        ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
        INPUT_DIR,_,_,EXPERIMENTS_DIR = get_dirs(ROOT_DIR)
        EXP_DIR = join(EXPERIMENTS_DIR,exp_id)
        config = load_config(EXP_DIR)
        dataset = config["data"]["dataset"]
        template_path = join(INPUT_DIR,dataset,"templates.csv")

    df = pd.read_csv(template_path)

    col_dict = {}
    for col in df.columns:
        items_dict = {}

        items = df[col].unique()
        for item in items:

            mask = df[col]==item
            items_dict[item]=list(mask)

        col_dict[col] = items_dict

    return col_dict




def get_masks_len(masks: dict)-> dict:

        unsorted_dict = {}

        for k in masks.keys():
            unsorted_dict[k]=sum(masks[k])

        return dict(sorted(unsorted_dict.items()))

# def main():

#     exp_id = "ishigami_test"
#     sub_id = "test"
#     checkpoint = "epoch=29-val_loss=0.51.ckpt"

#     ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
#     _, _, _, EXPERIMENTS_DIR = get_dirs(ROOT_DIR)
#     EXP_DIR = join(EXPERIMENTS_DIR, exp_id)
#     SUB_DIR = join(EXP_DIR, sub_id)

#     mask = torch.tensor([False, True, False, False])
#     output_path = join(SUB_DIR, "output")


#     dm = load_dataset(exp_id=exp_id)
#     print("Data Loaded")

#     model = load_model(exp_id=exp_id, checkpoint=checkpoint)

#     print("Model loaded.")

#     dynamic_kwargs = {
#         "enc_mask_flag": False,
#         "enc_mask": None,
#         "dec_self_mask_flag": False,
#         "dec_self_mask": None,
#         "dec_cross_mask_flag": False,
#         "dec_cross_mask": None,
#         "enc_output_attn": False,
#         "dec_self_output_attn": False,
#         "dec_cross_output_attn": False,
#     }

#     # model.set_kwargs(dynamic_kwargs_mask)
#     print("Predict 1...Full...")

#     predict_save(
#         model=model,
#         dm=dm,
#         dataset_label="train",
#         input_mask=mask,
#         debug_flag=True,
#         dynamic_kwargs=dynamic_kwargs,
#         save_label="",
#         output_path=output_path)



# if __name__ == "__main__":
#     main()
