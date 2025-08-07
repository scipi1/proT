
import numpy as np
import torch
import random
from os.path import dirname, abspath, join
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

from proT.forecaster import TransformerForecaster
from proT.dataloader import ProcessDataModule
from proT.predict import predict
from proT.labels import *
from proT.old_.config import load_config

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def save_output(src_trg_list:dict,output_path):
        
        for tuple in src_trg_list:

                source = tuple[0]
                trg_filename = tuple[1]

                with open(join(output_path,trg_filename), 'wb') as f:
                        np.save(f, source)


def main():
        

        exp_id = "ishigami_test"
        checkpoint = "epoch=29-val_loss=0.51.ckpt"

        ROOT_DIR = dirname(dirname(abspath(__file__)))
        INPUT_DIR,_,_, EXPERIMENTS_DIR = get_dirs(ROOT_DIR)
        EXP_DIR = join(EXPERIMENTS_DIR,exp_id)

        mask = torch.tensor([False, True, False, False])
        output_path = join(EXP_DIR,"output")
        checkpoint_path = join(EXP_DIR,"checkpoints",checkpoint)
        config = load_config(EXP_DIR)

        print("Loading data...")
        dm = ProcessDataModule(
                ds_flag=config["data"]["ds_flag"],
                data_dir=join(INPUT_DIR,config["data"]["dataset"]),
                files=("X_np.npy","Y_np.npy"),              #move to config
                batch_size=config["training"]["batch_size"],
                num_workers = 20)

        dm.prepare_data()
        dm.setup(stage=None)
        
        print("Loading model...")
        model = TransformerForecaster(config)
        model_resumed = model.load_from_checkpoint(checkpoint_path) 
        

        dynamic_kwargs_smask = {
                "enc_mask_flag"         : False,
                "enc_mask"              : None,
                "dec_self_mask_flag"    : False,
                "dec_self_mask"         : None,
                "dec_cross_mask_flag"   : False,
                "dec_cross_mask"        : None,
                "enc_output_attn"       : False,
                "dec_self_output_attn"  : False,
                "dec_cross_output_attn" : False,}
        
        # model.set_kwargs(dynamic_kwargs_mask)
        print("Predict 1...Full...")
        pred_val, true_val, cross_att, vs, H = predict(
                exp_id=exp_id,
                checkpoint=None,
                model_given=True,
                dm=dm,
                model=model_resumed,
                input_mask=None,
                kwargs=dynamic_kwargs_smask)
        
        save_output(
                src_trg_list=[
                        (pred_val,"output.npy"),
                        (true_val,"trg.npy"),
                        (cross_att,"att.npy"),
                        (vs,"V.npy"),],
                output_path=output_path)
        
        
        print("Predict 2...Masking the input...")
        pred_val, true_val, cross_att, vs, H = predict(
                exp_id=exp_id,
                checkpoint=None,
                model_given=True,
                dm=dm,
                model=model_resumed,
                input_mask=mask,
                kwargs=dynamic_kwargs_smask)
        
        save_output(
                src_trg_list=[
                        (pred_val,"output_2.npy"),
                        (true_val,"trg_2.npy"),
                        (cross_att,"att_2.npy"),
                        (vs,"V_2.npy"),],
                output_path=output_path)
        
        print("Predict 3...Masking the attention...")
        
        
        dynamic_kwargs_mask = {
                "enc_mask_flag"         : False,
                "enc_mask"              : None,
                "dec_self_mask_flag"    : False,
                "dec_self_mask"         : None,
                "dec_cross_mask_flag"   : True,
                "dec_cross_mask"        : torch.logical_not(mask),
                "enc_output_attn"       : False,
                "dec_self_output_attn"  : False,
                "dec_cross_output_attn" : False,}
        
        
        pred_val, true_val, cross_att, vs, H = predict(
                exp_id=exp_id,
                checkpoint=None,
                model_given=True,
                dm=dm,
                model=model_resumed,
                input_mask=None,
                kwargs=dynamic_kwargs_mask)
        
        save_output(
                src_trg_list=[
                        (pred_val,"output_3.npy"),
                        (true_val,"trg_3.npy"),
                        (cross_att,"att_3.npy"),
                        (vs,"V_3.npy"),],
                output_path=output_path)

if __name__ == "__main__":
    main()