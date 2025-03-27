import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange, repeat

from os.path import dirname, abspath
import sys
sys.path.append(dirname(abspath(__file__)))
from prochain_transformer.modules.embedding_layers import Time2Vec, SinusoidalPosition, identity_emb, nn_embedding


class EmbeddingMap(nn.Module):
    """Mapping between dataset variable index and embedding
    """
    def __init__(self,var_idx:int ,embedding:nn.Module, kwargs:dict, device):
        super().__init__()
        self.var_idx = var_idx
        self.embedding = embedding(**kwargs,device=device) 
        
        
        
    def __call__(self,X: torch.Tensor):
        
        X_ = torch.nan_to_num(X[:,:,self.var_idx])
        
        try:
            out = self.embedding(X_)
        except:
            # in case the embedding is a lookup table, the input must be int
            X_ = X_.type(torch.int)
            out = self.embedding(X_)
        return out




class ModularEmbedding(nn.Module):
    """
    It builds the embedding specific for the dataset 
    though the instructions in "ds_embed" 'dictionary argument.
    
    Dictionary example:
    ds_embed = [
        {
            "idx":0,
            "embed":"nn_embedding",
            "kwargs": {"num_embeddings":vocab_size, "embedding_dim":d_model}
            },
        {
            "idx": 1,
            "embed":"nn_embedding",
            "kwargs": {"num_embeddings":vocab_size, "embedding_dim":d_model}
            },
    ]

    Options
    - embed: "mask", "nn_embedding", "time2vec", "identity", "pass"
    
    """
    def __init__(
        self,
        ds_embed:dict,
        comps:str,
        device,
        
    ):
        super().__init__()
        
        assert comps in ["concat"]
        
        self.comps = comps
        
        # assemble list of embeddings according to "ds_embed"
        self.embed_list = []
        self.pass_idx_list = None
        self.mask_idx = None
        
        for var in ds_embed:
            idx_, embed_, kwargs = var["idx"], var["embed"], var["kwargs"]
            
            # assign embedding layers
            assert embed_ in ["mask", "nn_embedding", "sinusoidal","time2vec","identity","pass"], AssertionError("Invalid embedding selected!")
            
            if var["embed"] == "mask":
                self.mask_idx = idx_
                emb_module = None
                
            if var["embed"] == "nn_embedding":
                emb_module = nn_embedding
                
            if var["embed"] == "sinusoidal":
                emb_module = SinusoidalPosition
                
            if var["embed"] == "time2vec":
                emb_module = Time2Vec
                
            if var["embed"] == "identity":
                emb_module = identity_emb
                
            if var["embed"] == "pass":
                emb_module = None
                
                if self.pass_idx_list is None:
                    self.pass_idx_list = []
                    
                self.pass_idx_list.append(var["idx"])
                
            if emb_module is not None:
                self.embed_list.append(EmbeddingMap(var_idx=idx_ ,embedding=emb_module, kwargs=kwargs, device=device))


    def __call__(self, X: torch.Tensor):
        if self.comps == "concat":
            return self.concat(X)
        
        
    def concat(self, X: torch.Tensor):
        """
        Returns all concat embeddings on the last dimension
        out tensor shape (BS, seq_length, sum(d_embed))   
        """     
        return torch.cat([embed(X) for embed in self.embed_list],dim=-1)
    
    
    def get_mask_tensor(self, X: torch.Tensor):
        assert self.mask_idx is not None, AssertionError("Mask index not found in the embedding settings")
        return X[:,:,self.mask_idx]
    
    def get_mask(self, X: torch.Tensor):
        X_mask = self.get_mask_tensor(X)
        return X_mask.isnan().unsqueeze(-1)
    
    def pass_var(self, X: torch.Tensor):
        return X[:,:,self.pass_idx_list] if self.pass_idx_list is not None else None




def main():
    # quick test
    
    X = torch.rand(5,10,3)
    X[:,:,0]=torch.arange(10)
    X[:,:,1]=torch.zeros(10)
    d_model = 4
    vocab_size = 15

    ds_embed = [
        {
            "idx":0,
            "embed":"nn_embedding",
            "kwargs": {"num_embeddings":vocab_size, "embedding_dim":d_model}
            },
        {
            "idx": 1,
            "embed":"nn_embedding",
            "kwargs": {"num_embeddings":vocab_size, "embedding_dim":d_model}
            },
        # {
        #     "idx": 0,
        #     "embed": "pass",
        #     "kwargs": None
        #     },
    ]
    
    embed = ModularEmbedding(ds_embed,"concat",device="cpu")
    X_emb = embed(X)
    print(X_emb.shape)
    print(embed.pass_idx_list)
    print(embed.pass_var(X))
    

if __name__ == "__main__":
    main()