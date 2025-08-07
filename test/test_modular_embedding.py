from os.path import dirname, abspath, join
import sys
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)

from proT.modules.embedding import ModularEmbedding
import torch
import numpy as np




def main(*args, **kwargs):
    
    # X = torch.rand(5,10,3)
    # X[:,:,0]=torch.arange(10)
    # X[:,:,1]=torch.zeros(10)
    
    # real data
    filepath = "data/input/dyconex_252901/X.npy"
    X = np.load(join(ROOT_DIR,filepath),allow_pickle=True)
    X = torch.tensor(X[:5].astype("float"))

    pos_idx = 2
    time_idx = slice(5,None)
    time_dim = 5
    
    d_model = 4
    vocab_size = int(X.nan_to_num().max())

    ds_embed = [
        {
            "idx":pos_idx,
            "embed":"nn_embedding",
            "kwargs": {"num_embeddings":vocab_size, "embedding_dim":d_model}
            },
        {
            "idx": time_idx,
            "embed":"time2vec",
            "kwargs": {"input_dim":time_dim, "embed_dim":time_dim*5}
            },
    ]

    
    embed = ModularEmbedding(ds_embed=ds_embed, comps="concat")
    
    
    embed_out = embed(X)
    print(f"{embed_out.shape}")
    
    return embed_out

if __name__ == "__main__":
    main()
    