import torch, torch.nn as nn
from os.path import dirname, abspath, join
import sys
from os.path import abspath, join
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)
from prochain_transformer.modules.embedding import ModularEmbedding

# TODO: delete
# class GRUBaseline(nn.Module):
#     def __init__(self, d_in, d_hidden, n_layers, d_out, comps_embed, ds_embed):
#         super().__init__()
        
#         device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         self.embedding = ModularEmbedding(
#             ds_embed=ds_embed,
#             comps=comps_embed,
#             device=device)
        
#         self.gru = nn.GRU(
#             d_in, 
#             d_hidden, 
#             n_layers,
#             batch_first=True
#             )
#         self.head = nn.Linear(d_hidden, d_out)
        
        
#     def forward(self, x):           
#         x = self.embedding(x)
#         _, h_n = self.gru(x)        # h_n: (n_layers,N,d_hidden)
#         return self.head(h_n[-1])   # (N,d_out)
    
    
class RNN(nn.Module):
    def __init__(
        self, 
        model, 
        d_in, 
        d_emb, 
        d_hidden, 
        n_layers, 
        comps_embed, 
        ds_embed_in, 
        ds_embed_trg):
        
        super().__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # define embeddings
        self.input_embedding = ModularEmbedding(
            ds_embed=ds_embed_in,
            comps=comps_embed,
            device=device)
        
        self.target_embedding = ModularEmbedding(
            ds_embed=ds_embed_trg,
            comps=comps_embed,
            device=device)
        
        # select RNN model
        if model == "GRU":
            nn_module = nn.GRU
        elif model == "LSTM":
            nn_module = nn.LSTM
            
        self.rnn = nn_module(
            input_size=d_in, 
            hidden_size=d_hidden, 
            num_layers=n_layers,
            batch_first=True
            )
        
        # define final projection layer
        self.head = nn.Linear(d_emb+n_layers*d_hidden, 1)
        
        
    def forward(self, X, y):           
        
        # input and target embeddings
        X = self.input_embedding(X)
        y = self.target_embedding(y)
        _, L_out, _ = y.shape
        
        # RNN last hidden state (context)
        _, h_n = self.rnn(X) if isinstance(self.rnn, nn.GRU) else self.rnn(X)[1]
        ctx = h_n.reshape(h_n.size(1), -1)    # (B, n_layers*d_hidden)
        
        # expand context to match the target embeddings
        ctx_expanded = ctx.unsqueeze(1).expand(-1, L_out, -1)  # (B, L, n_layers*d_hidden)
        
        # concatenate target embeddings with context
        z = torch.cat((y, ctx_expanded), dim=-1)
        
        # final projection
        out = self.head(z).squeeze(-1)     
        
        return out




# class TCNBaseline(nn.Module):
#     def __init__(self, d_in, channels, kernel=3, d_out=1):
#         super().__init__()
#         layers = []
#         for i, ch in enumerate(channels):
#             dilation = 2**i
#             layers += [nn.Conv1d(d_in if i==0 else channels[i-1],
#                                  ch, kernel,
#                                  padding=(kernel-1)*dilation,
#                                  dilation=dilation),
#                        nn.ReLU()]
#         self.tcn = nn.Sequential(*layers)
#         self.head = nn.Linear(channels[-1], d_out)

#     def forward(self, x):               # x: (N,L,D)
#         x = x.permute(0,2,1)            # (N,D,L) → Conv1d
#         out = self.tcn(x)               # (N,C,L)
#         out = out.mean(-1)              # global avg pool over time
#         return self.head(out)           # (N,d_out)
    
    
    
    
class TCN(nn.Module):
    def __init__(self,
                 model: str,
                 d_in: int,           # input embedding dim
                 d_emb: int,          # target embedding dim
                 channels: list[int], # conv widths e.g. [64, 64, 128]
                 comps_embed,
                 ds_embed_in,
                 ds_embed_trg,
                 kernel: int = 3,
                 d_out: int = 1):
        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.input_embedding  = ModularEmbedding(
            ds_embed_in,  
            comps_embed,
            device=device
            )
        self.target_embedding = ModularEmbedding(
            ds_embed_trg, 
            comps_embed,
            device=device
            )

        layers = []
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            pad = (kernel - 1) * dilation        # causal padding
            layers += [
                nn.Conv1d(
                    in_channels=d_in if i == 0 else channels[i - 1],
                    out_channels=ch,
                    kernel_size=kernel,
                    dilation=dilation,
                    padding=pad
                ),
                nn.ReLU()
            ]
        self.tcn = nn.Sequential(*layers)

        in_feats = channels[-1] + d_emb         # ctx + tgt_emb per step
        self.head = nn.Linear(in_feats, d_out)  # shared across time steps

    
    def forward(self, X, y):

        # embed input & run causal TCN
        x_emb   = self.input_embedding(X)       # B × L_in × d_in_emb
        x_conv  = x_emb.permute(0, 2, 1)        # B × d_in_emb × L_in
        conv_out = self.tcn(x_conv)             # B × C × (L_in + pad)
        conv_out = conv_out[..., :X.size(1)]    # trim future‐leak padding
        ctx = conv_out[..., -1]                 # B × C (last time step)

        # embed target-side features
        y_emb = self.target_embedding(y)
        _, L_out, _ = y_emb.shape

        # broadcast context and concat
        ctx_exp = ctx.unsqueeze(1).expand(-1, L_out, -1)  # B × L_out × C
        z = torch.cat([y_emb, ctx_exp], dim=-1)           # B × L_out × (C+d_emb)

        # 4) time-distributed prediction
        out = self.head(z).squeeze(-1)  # B × L_out
        return out