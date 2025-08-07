import torch
import numpy as np

from os.path import dirname, abspath, join
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

from proT.proT_model import ProT
from proT.old_.config import get_folders

DATA_DIR,INPUT_DIR,_,_ = get_folders(key="local")

# general
BATCH_SIZE = 5
seq_len = 10
d_model = 12
time_comp = 5
use_val = False
# attention
d_queries_keys = 8
d_values = 8
n_heads = 3
dropout_qkv = 0.1
# encoder
e_layers = 3
norm = "layer"
dropout_attn_out = 0.1
activation = "relu"
use_final_norm = True
dropout_emb = 0.1
#d_ff???
#decoder
d_layers = 3
d_ff = 4*d_model
dropout_ff = 0.1
data = "real"


# dummy data
if data == "dummy":
    enc_x = torch.rand(BATCH_SIZE,seq_len,time_comp)
    enc_y = torch.rand(BATCH_SIZE,seq_len,1) 
    enc_p = torch.arange(0,seq_len).view(BATCH_SIZE,seq_len)
    dec_x = torch.rand(BATCH_SIZE,seq_len,time_comp)
    dec_y = torch.rand(BATCH_SIZE,seq_len,1) 
    dec_p = torch.arange(0,seq_len).view(BATCH_SIZE,seq_len)

# real data
if data == "real":
    X_np = np.load(join(INPUT_DIR,"X_np.npy"))
    Y_np = np.load(join(INPUT_DIR,"Y_np.npy"))
    enc_y = torch.Tensor(X_np[:BATCH_SIZE,0,:])
    enc_y = enc_y.reshape(enc_y.shape[0],enc_y.shape[1],1)
    enc_p = torch.Tensor(X_np[:BATCH_SIZE,1,:])
    enc_x = torch.Tensor(X_np[:BATCH_SIZE,2:,:]).permute(0,2,1)
    dec_y = torch.Tensor(Y_np[:BATCH_SIZE,0,:])
    dec_y = dec_y.reshape(dec_y.shape[0],dec_y.shape[1],1)
    dec_p = torch.Tensor(Y_np[:BATCH_SIZE,1,:])
    dec_x = torch.Tensor(Y_np[:BATCH_SIZE,2:,:]).permute(0,2,1)
    


model = ProT(d_time=time_comp)


print(f"Shapes enc_x, dec_x: {enc_x.shape}, {dec_x.shape}")
print(f"Shapes enc_y, dec_y: {enc_y.shape}, {dec_y.shape}")
print(f"Shapes enc_p, dec_p: {enc_p.shape}, {dec_p.shape}")

model.forward(
    enc_x,
    enc_y,
    enc_p,
    dec_x,
    dec_y,
    dec_p,
    output_attention=True,
)