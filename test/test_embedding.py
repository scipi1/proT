from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

from proT.modules.embedding import Embedding
import torch

BATCH_SIZE = 1
time_comp = 6
seq_len = 10
d_model = 50
use_val = True


d_value = 1,
d_time = 6,
d_model = 100,
time_emb_dim = 6, # hidden dimension of the time to vec, not d_time!
d_var_emb = 10,
var_vocab_siz = 1000,
is_encoder = True,
# position_emb : str = "abs",
embed_method = "spatio-temporal",
dropout_data = None,
max_seq_len = 1600,
use_given = True,
use_val = True



def main(*args, **kwargs):
    
    # quick test
    x = torch.rand(BATCH_SIZE,seq_len,time_comp)
    y = torch.rand(BATCH_SIZE,seq_len,1) 
    p = torch.arange(0,seq_len).view(BATCH_SIZE,seq_len).unsqueeze(-1)
    print(p.shape)
    
    
    embed = Embedding(d_value=1, d_time=time_comp, d_model=d_model,use_val=use_val)
    
    embed = Embedding(
        d_value = d_value,
        d_time = d_time,
        d_model = d_model,
        time_emb_dim = time_emb_dim, # hidden dimension of the time to vec, not d_time!
        d_var_emb = d_var_emb,
        var_vocab_siz = var_vocab_siz,
        is_encoder = True,
        # position_emb  = "abs",
        embed_method = "spatio-temporal",
        dropout_data = None,
        max_seq_len = 1600,
        use_given = True,
        use_val = True)
    
    embed_out = embed(x=x, y=y, p=p)
    print(f"Check embedding output shape: BATCH_SIZE ({BATCH_SIZE}), seq_len ({seq_len}), d_model({d_model}) --> {embed_out.shape}")
    
    return embed_out

if __name__ == "__main__":
    main()
    