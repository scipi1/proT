from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

from proT.modules.encoder import Encoder,EncoderLayer

from proT.modules.decoder import Decoder,DecoderLayer
from proT.modules.extra_layers import Normalization
from test_attention import main as get_test_attention
from test_embedding import main as get_test_embedding

import torch

# general
BATCH_SIZE = 1
seq_len = 10
d_model = 12
time_comp = 6
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


x_emb = get_test_embedding(BATCH_SIZE=BATCH_SIZE,
                           time_comp=time_comp,
                           seq_len=seq_len,
                           d_model=d_model,
                           use_val=use_val)


_,_,attention = get_test_attention(BATCH_SIZE=BATCH_SIZE,
                                   seq_len=seq_len,
                                   d_model=d_model,
                                   d_queries_keys=d_queries_keys,
                                   d_values=d_values,
                                   n_heads=n_heads,
                                   dropout_qkv=dropout_qkv)



encoder = Encoder(encoder_layers=[EncoderLayer(global_attention=attention,
                                            d_model_enc=d_model,
                                            #d_yc=d_yc if embed_method == "spatio-temporal" else 1,
                                            #time_windows=attn_time_windows,
                                            #time_window_offset=2 if use_shifted_time_windows and (l % 2 == 1) else 0,
                                            #d_ff=d_ff,dropout_ff=dropout_ff,
                                            dropout_attn_out=dropout_attn_out,
                                            activation=activation,
                                            norm=norm)

                               for l in range(e_layers)],

                  #conv_layers = [ConvBlock(split_length_into=split_length_into, d_model=d_model) for l in range(intermediate_downsample_convs)],
                  norm_layer = Normalization(norm, d_model=d_model) if use_final_norm else None,
                  emb_dropout = dropout_emb)


decoder = Decoder(
        decoder_layers=[
            DecoderLayer(
                global_self_attention=attention,
                local_self_attention=None,
                global_cross_attention=attention,
                local_cross_attention=None,
                d_model_dec=d_model,
                time_windows=None,
                # decoder layers alternate using shifted windows, if applicable
                time_window_offset=None,
                d_ff=d_ff,
                # temporal embedding effectively has 1 variable
                # for the purposes of time windowing.
                d_yt=None, #d_yt if embed_method == "spatio-temporal" else 1,
                d_yc=None, #d_yc if embed_method == "spatio-temporal" else 1,
                dropout_ff=dropout_ff,
                dropout_attn_out=dropout_attn_out,
                activation=activation,
                norm=norm,
            ) for l in range(d_layers)],

        norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
        emb_dropout=dropout_emb)

val_time_emb = torch.rand(BATCH_SIZE, seq_len, d_model)
space_emb = torch.rand(BATCH_SIZE, seq_len, d_model)

x,attn = encoder(val_time_emb=val_time_emb, space_emb=space_emb)

y,attns = decoder(val_time_emb=val_time_emb, space_emb=space_emb,cross=x)

print(f"x shape: {x.shape}")
print(f"Attention is a list with {len(attn)} tensors with shape: {attn[-1].shape}")

print(f"y shape: {y.shape}")
print(f"Attention is a list with {len(attns)} tensors with shape: {attns[-1].shape}")