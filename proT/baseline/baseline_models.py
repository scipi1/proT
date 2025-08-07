import torch, torch.nn as nn
from os.path import dirname, abspath, join
import sys
from os.path import abspath, join
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)
from proT.modules.embedding import ModularEmbedding



class RNN(nn.Module):
    def __init__(
        self, 
        model, 
        d_in, 
        d_emb, 
        d_hidden, 
        n_layers,
        dropout,
        comps_embed, 
        ds_embed_in, 
        ds_embed_trg
        ):
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
            dropout=dropout if n_layers > 1 else 0.0,
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



class TCN(nn.Module):
    def __init__(
        self,
        model: str,
        d_in: int,           # input embedding dim
        d_emb: int,          # target embedding dim
        channels: list[int], # conv widths e.g. [64, 64, 128]
        dropout: float,
        comps_embed: str,
        ds_embed_in: dict,
        ds_embed_trg: dict,
        kernel: int = 3,
        d_out: int = 1
        ):
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
                nn.ReLU(),
                nn.Dropout(p=dropout)
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



class MLP(nn.Module):
    def __init__(
        self,
        model,
        d_in: int,
        d_emb: int,
        dropout: float,
        comps_embed: str,
        ds_embed_in: dict,
        ds_embed_trg: dict,
        hidden: list[int] = [256, 256],
        d_out: int = 1
        ):
        super().__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.input_embedding  = ModularEmbedding(
            ds_embed=ds_embed_in,  
            comps=comps_embed, 
            device=device)
        
        self.target_embedding = ModularEmbedding(
            ds_embed=ds_embed_trg, 
            comps=comps_embed, 
            device=device)

        # mean-pool along the sequence length L_in
        self.pool = lambda x: x.mean(dim=1)           # B × d_in

        #  MLP over pooled context
        layers, in_dim = [], d_in
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(p=dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)             # ctx: B × h_last

        # final projection layer
        self.head = nn.Linear(hidden[-1] + d_emb, d_out)

    def forward(self, X, y):
        
        # input and target embeddings
        X = self.input_embedding(X)                         # input emb --> B × L_in × d_in_emb
        yemb = self.target_embedding(y)                     # target emb --> B × L_out × d_emb
        _, L_out, _ = yemb.shape

        # MLP forward pass
        ctx  = self.mlp(self.pool(X))                       # input emb --> B × h_last
        
        # expand context to match the target embeddings
        ctx_exp = ctx.unsqueeze(1).expand(-1, L_out, -1)    # input emb --> B × L_out × h_last
        
        # concatenate target embeddings with context
        z   = torch.cat([yemb, ctx_exp], -1)                # input emb --> B × L_out × (h_last+d_emb)
        
        # final projection
        out = self.head(z).squeeze(-1)
        
        return out

