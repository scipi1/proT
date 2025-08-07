from os.path import dirname, abspath, join,exists
import sys

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)
from proT.modules.utils import set_seed
from proT.subroutines.predict_many_masks_input import predict_many_masks_input
from proT.subroutines.predict_mask_input import predict_mask_input
from proT.subroutines.sub_utils import *


def main():
    exp_id="markov_ishigami_a1b1c015d1"
    sub_id="test_predictions"
    checkpoint = "epoch=129-train_loss=0.01.ckpt"
    debug_flag = True
    sparsity_flag = False
    
    
    set_seed(42)
    
    #unconditioned
    predict_mask_input(
        exp_id = exp_id,
        sub_id = sub_id,
        checkpoint = checkpoint,
        mask = None,
        save_label="full",
        debug_flag=debug_flag,
        zero_flag= sparsity_flag)
    
    
    
    #conditioned
    masks = [
        [True,True,True,False,False,False],
        [False,False,False,True,True,True],
        [False,False,False,False,False,False],
        [True,False,False,False,False,False],
        [False,True,False,False,False,False],
        [False,False,True,False,False,False],
        [False,False,False,True,False,False],
        [False,False,False,False,True,False],
        [False,False,False,False,False,True],
    ]
    
    save_labels=[
        "first_half",
        "second_half",
        "all_zero",
        "step_1",
        "step_2",
        "step_3",
        "step_4",
        "step_5",
        "step_6",
    ]
    
    predict_many_masks_input(
            exp_id=exp_id,
            sub_id=sub_id,
            checkpoint=checkpoint,
            masks=torch.tensor(masks),
            save_labels=save_labels,
            debug_flag=debug_flag,
            sparsity_flag= sparsity_flag)
    
    
if __name__ == "__main__":
    main()