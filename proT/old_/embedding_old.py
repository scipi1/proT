import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from os.path import dirname, abspath
import sys
sys.path.append(dirname(abspath(__file__)))
from proT.modules.embedding_layers import Time2Vec

class Embedding(nn.Module):
    def __init__(
        self,
        d_value: int = 1,
        d_time: int = 6,
        d_model: int = 100,
        time_emb_dim: int = 6, # hidden dimension of the time to vec, not d_time!
        d_var_emb: int = 10,
        var_vocab_size:int = 1000,
        is_encoder: bool = True,
        # position_emb : str = "abs",
        embed_method: str = "spatio-temporal",
        dropout_data = None,
        max_seq_len: int = 1600,
        use_given: bool = True,
        use_val: bool = True
        
    ):
        super().__init__()
        
        if dropout_data is None:
            self.data_drop = lambda y: y
        else:
            self.data_drop = dropout_data
        
        embed_opts = ["spatio-temporal","positional"]
        assert embed_method in embed_opts, AssertionError(f"Embed Type invalid! Choose from {embed_opts}")
        
        self.d_y = d_value
        self.d_time = d_time
        self.d_model = d_model
        self.embed_method = embed_method
        self.is_encoder = is_encoder
        self.max_seq_len = max_seq_len
        self.use_given = use_given
        self.use_val = use_val
        
        # for space-time embed
        self.given_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)
        self.position_emb = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=d_model) 
        # self.space_emb = nn.Embedding(num_embeddings=d_y, embedding_dim=d_model) 
        # time_dim = time_emb_dim * d_time
        self.time_emb = Time2Vec(d_time, embed_dim=time_emb_dim*d_time)
        
        # projects back the value+time embedding into d_model so it can be added with the other embeddings
        self.val_time_emb = nn.Linear((time_emb_dim*d_time)+1, d_model) #dimensions: time components + value (1)
        
        # for pos_embed
        self.val_emb = nn.Linear(1, d_model)
        
        # MODULAR MULTIVARIATE EMBED
        self.var_emb = nn.Embedding(var_vocab_size,d_var_emb)
        
    def __call__(self, y:torch.Tensor): #(self, x:torch.Tensor, y:torch.Tensor,p:torch.Tensor)
        
        if self.embed_method == "spatio-temporal":
            embed = self.spatio_temporal_embed(X=y) #(x=x, y=y,p=p)
        
        elif self.embed_method == "positional":
            embed = self.pos_embed(y=y)
            
        return embed
    
    def pos_embed(self,y: torch.Tensor):
        
        y = y.unsqueeze(1)
        
        seq_len=y.shape[-1]
        
        p = torch.arange(seq_len, device=y.get_device()).view(1, 1, seq_len)
        p = p.expand_as(y)
        
        pos_emb = self.position_emb(p)
        
        value_emb = self.val_emb(y.unsqueeze(-1))
        combined = pos_emb+value_emb
        combined = combined.squeeze(1)
        
        return combined
    
    def spatio_temporal_embed(self, X:torch.Tensor):
        
        
        y,p,x = X[:,0,:].unsqueeze(-1),X[:,1,:],X[:,2:,:].permute(0,2,1)#.permute(0,1,3,2)
        
        batch, length, dy = y.shape
        
        pos = p
        
        # given embedding
        
        true_null = torch.isnan(y).squeeze(-1)
        y = torch.nan_to_num(y)
        
        if not self.use_val:
            y = torch.zeros_like(y)

        # keep track of pre-dropout y for given emb
        y_original = y.clone()
        #y = self.data_drop(y)        
        
        if self.use_given:
            
            given = torch.ones((batch, length)).long().to(x.device)  # start with all given
            
            if not self.is_encoder:
                # mask missing values that need prediction...
                given=torch.zeros((batch, length)).long().to(x.device)  # (False)
                
            given *= ~true_null                                          # if y was NaN, set Given = False
            given *= (y == y_original).squeeze(-1)                                  # update Given with dropout
            
            given_emb = self.given_emb(given)
        
        else:
            given_emb = 0.0
        
        # positional embedding ("pos_emb")
        pos_emb = self.position_emb(pos.long())
        
        
        ## temporal embedding
        # X (time)
        x = torch.nan_to_num(x) # set eventual nan to zero
        time_emb = self.time_emb(x) # apply the time_embed function, shape (len, d_model)
        
        # Y (value)
        y = torch.nan_to_num(y)
        
        # concat time_emb, y --> FF --> val_time_emb
        val_time_inp = torch.cat((time_emb, y), dim=-1) 
        val_time_emb = self.val_time_emb(val_time_inp) # linear layer
        
        # dropout
        ...
        # mask
        ...
        
        given_emb = self.given_emb(given)
        
        # put everything together: value+time, variable, position, given
        val_time_emb =  val_time_emb + pos_emb + given_emb
        
        # space (variables) embedding
        # in the original code, they are generated
        
        # var_idx = var
        # var_idx_true = var_idx.clone()
        # space_emb = self.space_emb(var_idx)
        
        return val_time_emb #space_emb, var_idx_true, mask
    
    # def spatio_temporal_embed(self, x:torch.Tensor, y:torch.Tensor, p:torch.Tensor):
    #     """_summary_

    #     Args:
    #         x (torch.Tensor): time vector
    #         y (torch.Tensor): value vector
    #         p (torch.Tensor): position vector
    #         g (torch.Tensor): given vector
            
    #     Returns:
    #         _type_: _description_
    #     """
    #     batch, length, dy = y.shape
        
    #     local_pos = p
        
    #     # given embedding
        
    #     true_null = torch.isnan(y).squeeze(-1)
    #     y = torch.nan_to_num(y)
        
    #     if not self.use_val:
    #         y = torch.zeros_like(y)

    #     # keep track of pre-dropout y for given emb
    #     y_original = y.clone()
    #     y = self.data_drop(y)        
        
    #     if self.use_given:
            
    #         given = torch.ones((batch, length)).long().to(x.device)  # start with all given
            
    #         if not self.is_encoder:
    #             # mask missing values that need prediction...
    #             given=torch.zeros((batch, length)).long().to(x.device)  # (False)
                
    #         given *= ~true_null                                          # if y was NaN, set Given = False
    #         given *= (y == y_original).squeeze(-1)                                  # update Given with dropout
            
    #         given_emb = self.given_emb(given)
        
    #     else:
    #         given_emb = 0.0
        
    #     # positional embedding ("local_emb")
    #     local_emb = self.local_emb(local_pos.long())
        
        
    #     ## temporal embedding
    #     # X (time)
    #     x = torch.nan_to_num(x) # set eventual nan to zero
    #     time_emb = self.time_emb(x) # apply the time_embed function, shape (len, d_model)
        
    #     # Y (value)
    #     y = torch.nan_to_num(y)
        
    #     # concat time_emb, y --> FF --> val_time_emb
    #     val_time_inp = torch.cat((time_emb, y), dim=-1) # not very clear
    #     val_time_emb = self.val_time_emb(val_time_inp) # linear layer
        
    #     # dropout
    #     ...
    #     # mask
    #     ...
        
    #     given_emb = self.given_emb(given)
        
    #     # put everything togrther: value+time, variable, position, given
    #     val_time_emb =  val_time_emb + local_emb + given_emb
        
    #     # space (variables) embedding
    #     # in the original code, they are generated
        
    #     # var_idx = var
    #     # var_idx_true = var_idx.clone()
    #     # space_emb = self.space_emb(var_idx)
        
    #     return val_time_emb #space_emb, var_idx_true, mask

    
    def modular_multivariate_embed(self, X:torch.Tensor):
        
        # TODO: pass idx and embed layer to concat
        
        id_idx = 0
        process_idx = 1
        var_idx = 2
        pos_idx = 3
        val_idx = 4
        time_idx = 5
        
        
        # get the indicator function to nullify missing data embedding
        #Ind = torch.logical_not(torch.isnan(X[:,:,val_idx])).unsqueeze(-1)

        # convert missing data to zero
        # (this is needed to avoid that False X nan = nan instead of zero)
        #X = torch.nan_to_num(X)
        
        # value
        v = X[:,:,val_idx]
        
        # variable embeddings
        var_embed_idx = torch.nan_to_num(X[:,:,var_idx]).type(torch.int)
        phi_var = self.var_emb(var_embed_idx)
        
        # positional embeddings
        phi_pos = self.position_emb(X[:,:,pos_idx])
        
        # temporal embed
        phi_tem = self.time_emb(X[:,:,time_idx])
        
        return torch.cat((v,phi_var,phi_pos,phi_tem),dim=-1)


def main():
    # quick test
    BATCH_SIZE = 1
    time_comp = 6
    seq_len = 10
    d_model = 50
    
    x = torch.rand(BATCH_SIZE,seq_len,time_comp)
    y = torch.rand(BATCH_SIZE,seq_len,1) 
    p = torch.arange(0,seq_len).view(BATCH_SIZE,seq_len)
    
    
    embed = Embedding(d_time=time_comp, d_model=d_model,embed_method="positional")
    embed_out = embed(x=None, y=y, p=None)#embed(x=x, y=y, p=p)
    print(f"Check embedding output shape {embed_out.shape} <--> {BATCH_SIZE}, {seq_len}, {d_model}")
    

if __name__ == "__main__":
    main()