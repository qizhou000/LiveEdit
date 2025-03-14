#%%
from typing import Dict, List, Tuple
from torch import nn
import numpy as np
import torch

class Attention(nn.Module):
    def __init__(self, inp1_dim:int, inp2_dim:int, qk_dim:int, v_dim:int, head_n:int,
                 add_bias_q = True, add_bias_k = True, add_bias_v = True) -> None:
        super().__init__()
        assert qk_dim % head_n == 0
        self.head_n = head_n
        self.qk_head_dim = qk_dim // head_n
        self.v_head_dim = v_dim // head_n
        self.scale_factor = 1 / (self.qk_head_dim ** 0.5)
        self.q_mlp = nn.Linear(inp1_dim, qk_dim, add_bias_q)
        self.k_mlp = nn.Linear(inp2_dim, qk_dim, add_bias_k)
        self.v_mlp = nn.Linear(inp2_dim, v_dim, add_bias_v)
        self.reset_parameters()
    
    def forward(self, inp1:torch.Tensor, inp2:torch.Tensor, rescale_with_score = False):
        # inp1: [b, l1, d1]; inp2: [b, l2, d2];
        b, l1, _ = inp1.shape
        b, l2, _ = inp2.shape
        q = self.q_mlp(inp1).reshape(b, l1, self.head_n, self.qk_head_dim)
        k = self.k_mlp(inp2).reshape(b, l2, self.head_n, self.qk_head_dim)
        v = self.v_mlp(inp2).reshape(b, l2, self.head_n, self.v_head_dim)
        s = torch.softmax(torch.einsum('blhd,bmhd->blmh', q, k) * self.scale_factor, 2) # [b,l,m,h]
        res = torch.einsum('blmh,bmhd->blhd', s, v) # [b,l,h,d]
        if not rescale_with_score:
            return res.reshape(b, l1, -1) # [b,l,h*d]
        # rescale variance back to v's original variance
        res = res / torch.sum(s**2, 2).unsqueeze(-1)**0.5 # [b,l,h,d]
        return res.reshape(b, l1, -1)

    def reset_parameters(self):
        self.q_mlp.reset_parameters()
        self.k_mlp.reset_parameters()
        self.v_mlp.reset_parameters()

class QVExtractor(nn.Module):
    def __init__(self, eqe_n, inpt_reps_dim, module_dim, cross_att_head_n, vision_tok_n, vis_prot = False) -> None:
        super().__init__()
        # extract query for extract vision
        self.layer_norm1 = nn.LayerNorm(inpt_reps_dim)
        self.eqe1 = nn.Parameter(torch.zeros(1, eqe_n, module_dim)) # vision edit query extractor
        self.ca_query_info_ext1 = Attention(module_dim, inpt_reps_dim, module_dim, module_dim, cross_att_head_n)
        self.ca_vision_info_ext = Attention(module_dim, inpt_reps_dim, module_dim, module_dim, cross_att_head_n)
        # extract only for query
        self.layer_norm2 = nn.LayerNorm(inpt_reps_dim)
        self.eqe2 = nn.Parameter(torch.zeros(1, eqe_n, module_dim)) # vision edit query extractor
        self.ca_query_info_ext2 = Attention(module_dim, inpt_reps_dim, module_dim, module_dim, cross_att_head_n)
        # prototype vision representation
        if vis_prot:
            self.vis_rep_prot = nn.Parameter(torch.zeros(1, vision_tok_n, inpt_reps_dim)) # used for relative similarity comparison
        self.reset_parameters()

    def extract_vision(self, query_reps:torch.Tensor, vision_reps:torch.Tensor)->torch.Tensor:
        # vision_reps: [b, l1, d], query_reps: [b, l2, d]
        assert len(query_reps) == len(vision_reps) == 1
        query_reps, vision_reps = self.layer_norm1(query_reps), self.layer_norm1(vision_reps)
        eqr = self.ca_query_info_ext1(self.eqe1, query_reps) # [b, eqe_n, module_dim]
        evr = self.ca_vision_info_ext(eqr, vision_reps) # [b, eqe_n, module_dim]
        return evr # [b, eqe_n, module_dim]
    
    def extract_query(self, query_reps:torch.Tensor)->torch.Tensor:
        # query_reps: [b,l,d]
        assert len(query_reps) == 1
        query_reps = self.layer_norm2(query_reps)
        eqr = self.ca_query_info_ext2(self.eqe2, query_reps) # [b, eqe_n, module_dim]
        return eqr # [b, eqe_n, module_dim]

    def extract_from_visprot(self, query_reps:torch.Tensor):
        return self.extract_vision(query_reps, self.vis_rep_prot)

    def forward(self):
        raise
    
    def reset_parameters(self):
        self.layer_norm1.reset_parameters()
        nn.init.kaiming_normal_(self.eqe1)
        self.ca_query_info_ext1.reset_parameters()
        self.ca_vision_info_ext.reset_parameters()
        self.layer_norm2.reset_parameters()
        nn.init.kaiming_normal_(self.eqe2)
        self.ca_query_info_ext2.reset_parameters()
        if hasattr(self, 'vis_rep_prot'):
            nn.init.kaiming_normal_(self.vis_rep_prot)

class LowRankGenerator(nn.Module):
    def __init__(self, lora_dim, lora_rank, lora_scale, inpt_reps_dim, module_dim, cross_att_head_n) -> None:
        super().__init__()
        self.phi = nn.Parameter(torch.zeros(1, lora_rank, module_dim)) # lora phi
        self.ca_lora = Attention(module_dim, inpt_reps_dim, module_dim, lora_dim, cross_att_head_n)
        self.layer_norm = nn.LayerNorm(inpt_reps_dim) 
        self.scale = 1 / (lora_scale * lora_rank ** 0.5)
        self.reset_parameters()

    def forward(self, inpt_reps:torch.Tensor):
        # vision_reps: [b, l, d]
        assert len(inpt_reps) == 1
        inpt_reps = self.layer_norm(inpt_reps)
        lora = self.ca_lora(self.phi, inpt_reps) * self.scale  # [b, lora_rank, hidden_size]
        return lora

    def reset_parameters(self):
        # nn.init.normal_(self.phi, 0, 1)
        nn.init.kaiming_normal_(self.phi)
        self.ca_lora.reset_parameters()
        self.layer_norm.reset_parameters()

