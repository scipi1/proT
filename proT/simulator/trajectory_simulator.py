import torch
import torch.nn as nn
import torch.func as F
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import itertools




class ISTSimulator(nn.Module):
    def __init__(self, model, L=200, num_vars=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_vars = num_vars    # number of variables, generally 2 (SenseA, SenseB)
        if model == "F":
            self.simulator_model = FractureIST(L)
        else:
            self.simulator_model = DuctileIST(L)
        
        # they depend on the dataset, hard-coded for now, might read from config
        self.val_idx = 0
        self.pos_idx = 1
        self.var_idx = 2
        
        self.vars = torch.tensor([1,2]) # senseA or senseB
        
    def forward(self, transformer_out: torch.Tensor):
        """_summary_

        Args:
            transformer_out (torch.Tensor): output of the transformer with shape B x num_parameters*num_vars x 1

        Returns:
            _type_: _description_
        """
        
        # the output of the transformer is divided into chunks belonging to the same
        # output variable (e.g. Sense A, Sense B) and collected in a list
        transformer_out_list = self._prepare_transformer_out(transformer_out)
        
        # the simulator is called for every chunk, feeding all B samples and all parameters
        return torch.cat([self.simulator_model(t[:,:,0]) for t in transformer_out_list],dim=-1)
    
    def get_decoder_input(self,batch_size: int, device: str):
        
        pos = self.simulator_model.get_pos()
        
        # Generate all combinations of pos and vars
        combinations = list(itertools.product(self.vars, pos))

        # Create a tensor for the combinations
        comb_tensor = torch.tensor(combinations, dtype=torch.float32)
        
        # Create the tensor for a single batch element
        single_batch_tensor = torch.zeros((len(combinations), 3), device=device)
        single_batch_tensor[:, self.val_idx] = 0                   # value set 0
        single_batch_tensor[:, self.pos_idx] = comb_tensor[:, 1]   # position
        single_batch_tensor[:, self.var_idx] = comb_tensor[:, 0]   # variable

        # Expand the tensor to the desired batch size B
        return single_batch_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    
    def _prepare_transformer_out(self, transformer_out: torch.Tensor):
        
        # Splits the output tensor in v tensors, where v is the number of variables
        # usually v=2, (Sense A and Sense B)
        # results is a list[tensors] to feed the forward method
        # transformer_out is [bs x num_vars*num_pos x D] 
        # (D=3: 0:group_id, 1: position, 2: value)
        return  list(transformer_out.chunk(self.num_vars,1)) # splits the positions having the same variable 





class DuctileIST(nn.Module):
    
    def __init__(self, L, eps=1e-12,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.L = L
        self.kappa=1
        self.eps = eps
        
    def forward(self, model_out: torch.Tensor):
        
        # assign params
        lam     = 1e-3*torch.relu(model_out[:,0].unsqueeze(-1))
        beta    = 1.0 + 10.0*torch.relu(model_out[:,1].unsqueeze(-1))
        D       = torch.zeros_like(lam, device=model_out.get_device())
        
        out = []

        for _ in range(self.L):
            one_minus_D = torch.clamp(1.0 - D, min=self.eps)
            update = lam * torch.exp(beta * torch.log(one_minus_D)) # implemented a^x = exp(x log(a))
            D = torch.clamp(D + update, max=1.0 - self.eps)
            out.append(self.kappa * D)
            
        return torch.nan_to_num(torch.cat(out, dim=-1))             # shape [batch, L]
    
    def get_pos(self):
        return torch.tensor([1,2])
    
    
    
class FractureIST(nn.Module):
    def __init__(self, L, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.L = L

    def forward(self, transformer_out: torch.Tensor):
        
        # assign params
        lam     =   1e-3*torch.relu(transformer_out[:,0])           # [bs, 1]
        beta    =   1.0 + 10.0*torch.relu(transformer_out[:,1])     # [bs, 1]
        k_d     =   transformer_out[:,2]
        mu      =   transformer_out[:,3]  
        k_f     =   transformer_out[:,4] 
        n_c     =   transformer_out[:,5] 
        w_t     =   transformer_out[:,6]
        L       =   self.L
        
        eps = self.eps
        
        # initialise states
        #D = lam.new_zeros(batch)
        D   = torch.zeros_like(lam, device=transformer_out.get_device())
        #F = lam.new_full((batch,), eps)      # tiny seed
        F   = torch.ones_like(lam, device=transformer_out.get_device())*eps
        out = []

        for n in range(int(L)):
            # ductile channel 
            one_minus_D = torch.clamp(1. - D, min=eps)
            update = lam * torch.exp(beta * torch.log(one_minus_D)) # implemented a^x = exp(x log(a))
            D = torch.clamp(D + update, max=1.0 - self.eps)
            
            # fracture channel
            g = torch.sigmoid((n - n_c) / w_t)     # pole
            F = (1 + mu * g) * F + eps * g         
            F = torch.clamp(F, 0., 1. - eps)       # stay in domain

            deltaR = k_d * D + k_f * F
            out.append(deltaR)

        return torch.stack(out, dim=-1)
    
    def get_pos(self):
        num_params = 7
        return torch.arange(1,num_params+1)
        

    
    
if __name__ == "__main__":
    
    simulator = ISTSimulator()
    
    decoder_input = simulator.get_decoder_input(batch_size=10)
    
    print(f"decoder_input shape: {decoder_input.shape}")
    print(f"example value index [0,:,0]: {decoder_input[0,:,0]}")
    print(f"example position index [0,:,1]: {decoder_input[0,:,1]}")
    print(f"example variable [0,:,2]: {decoder_input[0,:,2]}")
    
    simulator_out = simulator(decoder_input)
    print(f"forecast output shape: {simulator_out.shape}")
    
    