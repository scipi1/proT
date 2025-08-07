from os.path import dirname, abspath, join,exists
import sys

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)
from proT.modules.utils import set_seed
from proT.subroutines.predict_many_masks_input import predict_many_masks_input
from proT.subroutines.predict_mask_input import predict_mask_input
from proT.subroutines.sub_utils import *

def main():
    exp_id="dyconex_453828B_241102_500dmodel_200dkq"
    sub_id="full_prediction"
    checkpoint = "epoch=199-train_loss=0.00.ckpt"
    column = "PaPos"
    debug_flag = False
    zero_flag= True
    dataset_label = "test"
    
    
    
    masks_dict = get_masks_from_template(exp_id=exp_id)
    
    assert column in masks_dict.keys(), AssertionError(f"{column} not valid! Choose from {masks_dict.keys()}")
    
    masks = masks_dict[column]
    
    #get_partial_length
    masks_len_dict = get_masks_len(masks=masks)
    pd.DataFrame.from_dict(data=masks_len_dict, orient='index').to_csv('partial_seq_lens.csv')
    
    
    set_seed(42)
    
    #unconditioned
    predict_mask_input(
        exp_id = exp_id,
        sub_id = sub_id,
        checkpoint = checkpoint,
        dataset_label =dataset_label,
        mask = None,
        save_label="full",
        debug_flag=debug_flag,
        zero_flag= zero_flag)
    
    breakpoint()
    
    
    #conditioned
    predict_many_masks_input(
            exp_id=exp_id,
            sub_id=sub_id,
            checkpoint=checkpoint,
            masks=[torch.tensor(masks[k]) for k in masks.keys()],
            save_labels=list((masks.keys())),
            debug_flag=debug_flag,
            sparsity_flag= zero_flag)
    
if __name__ == "__main__":
    main()