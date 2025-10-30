from os.path import dirname, abspath, join,exists
import sys

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(ROOT_DIR)

from proT.subroutines.sub_utils import *
from proT.subroutines.predict_mask_input import predict_mask_input



def predict_many_masks_input(
    exp_id: str = "ishigami_test",
    sub_id: str = "test",
    checkpoint: str = "epoch=29-val_loss=0.51.ckpt",
    dataset_label: str = "train",
    masks: list = [torch.tensor([False, True, False, False])],
    save_labels: list=["test"],
    debug_flag=False,
    sparsity_flag=False,
    ) -> None:
    
    assert len(masks)==len(save_labels), AssertionError(f"Got {len(masks)} masks, but {len(save_labels)} labels!")
    
    for i,mask in enumerate(masks):
        predict_mask_input(
                exp_id=exp_id,
                sub_id=sub_id,
                checkpoint=checkpoint,
                dataset_label=dataset_label,
                mask = mask,
                save_label=save_labels[i],
                debug_flag=debug_flag,
                zero_flag=sparsity_flag)



if __name__ == "__main__":
    
    predict_many_masks_input(
        exp_id="ishigami_test",
        sub_id="test_parameters_gsa",
        checkpoint="epoch=29-val_loss=0.51.ckpt",
        masks=[
            None,
            torch.tensor([True, False, True, False]),
            torch.tensor([False, True, False, False]),
            torch.tensor([False, False, False, True]),
            ],
        save_labels=[
            "full",
            "X1",
            "X2",
            "X3"
            ],
        debug_flag=False)