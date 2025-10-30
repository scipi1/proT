from os.path import dirname, abspath, join,exists
import sys

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)

from proT.subroutines.sub_utils import *
from proT.labels import *


def predict_mask_input(
    exp_id: str,
    sub_id: str,
    checkpoint: str,
    dataset_label:str,
    mask: torch.Tensor,
    save_label: str,
    debug_flag: bool,
    zero_flag: bool
    ):
    
    # directories
    ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
    _, _, _, EXPERIMENTS_DIR = get_dirs(ROOT_DIR)
    EXP_DIR = join(EXPERIMENTS_DIR, exp_id)
    assert exists(EXP_DIR), f"Folder does not exist: {EXP_DIR}"
    SUB_DIR = join(EXP_DIR, "sub", sub_id)
    output_path = join(SUB_DIR, "output")
    mk_missing_folders([SUB_DIR,output_path])
    
    # dataset
    dm = load_dataset(exp_id=exp_id)
    print("Data Loaded...")

    # model
    model = load_model(exp_id=exp_id, checkpoint=checkpoint)
    print("Model loaded...")
    
    # prediction options for masked input
    dynamic_kwargs = {
        "enc_mask_flag": False,
        "enc_mask": None,
        "dec_self_mask_flag": False,
        "dec_self_mask": None,
        "dec_cross_mask_flag": False,
        "dec_cross_mask": None,
        "enc_output_attn": False,
        "dec_self_output_attn": False,
        "dec_cross_output_attn": False,
    }

    # model.set_kwargs(dynamic_kwargs_mask)
    print("Predict...")
    predict_save(
        model=model,
        dm=dm,
        dataset_label=dataset_label,
        input_mask=mask,
        debug_flag=debug_flag,
        zero_flag=zero_flag,
        dynamic_kwargs=dynamic_kwargs,
        save_label=save_label,
        output_path=output_path)
    
    print("Predictions saved!")



if __name__ == "__main__":
    predict_mask_input(
        exp_id = "ishigami_test",
        sub_id = "test",
        checkpoint = "epoch=29-val_loss=0.51.ckpt",
        mask = torch.tensor([False, True, False, False]),
        save_label="test",
        debug_flag=True)