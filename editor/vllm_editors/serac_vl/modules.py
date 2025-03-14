from transformers import AutoModelForCausalLM, BertModel
from torch.nn.utils.rnn import pad_sequence
from utils.GLOBAL import model_path_map
from utils import get_full_model_name
import torch.nn.functional as F
from typing import List, Dict
from torch import nn
import torch

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_layer = nn.ReLU):
        super().__init__()
        self.l_in_norm = nn.LayerNorm(in_dim)
        self.l_in = nn.Linear(in_dim, out_dim)
        self.l_mid = nn.Linear(out_dim, out_dim)
        self.act = act_layer()
        self.l_out = nn.Linear(out_dim, out_dim)
        
    def reset_parameters(self):
        self.l_in_norm.reset_parameters()
        self.l_in.reset_parameters()
        self.l_mid.reset_parameters()
        self.l_out.reset_parameters()

    def forward(self, x):
        x = self.l_in(self.l_in_norm(x))
        x = self.l_out(self.act(self.l_mid(x))) + x
        return x

class Classifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert 'bert' in cfg.classifier_path
        self.classifier = BertModel.from_pretrained(cfg.classifier_path)
        self.classifier_proj = ResBlock(cfg.llm_hidden_size, cfg.classifier_rep_dim)
        self.ln = nn.LayerNorm( cfg.classifier_rep_dim)

    def forward(self):
        raise

    def get_sim_reps(self, inputs_embeds:torch.Tensor, attention_mask:torch.Tensor):
        # inputs_embeds: [b, l, d], attention_mask: [b, l]
        assert len(inputs_embeds) == 1 and len(inputs_embeds.shape) == 3
        inputs_embeds = self.classifier_proj(inputs_embeds[:, -512:])
        inputs_embeds = self.ln(inputs_embeds) / 32
        pooler_output = self.classifier(inputs_embeds = inputs_embeds, 
            attention_mask = attention_mask[:, -512:]).last_hidden_state # [b, l, d]
        pooler_output = torch.cat([pooler_output.mean(1), 
            pooler_output.max(1).values, pooler_output.min(1).values], 1)
        return pooler_output

    def get_sim(self, rep1:torch.Tensor, rep2:torch.Tensor):
        # rep1: [b1, d], rep2: [b2, d]
        # rep1 = F.normalize(rep1, 2, 1)
        # rep2 = F.normalize(rep2, 2, 1)
        sim = torch.einsum('bd,td->bt', rep1, rep2) / (rep2.shape[1] ** 0.5)  # [b1, b2]
        # sim = torch.sigmoid(sim)
        return sim
 
class CounterfactModel(nn.Module):
    def __init__(self, cfg, map_to_edit_llm_logits):
        super().__init__()
        self.counterfact_model = AutoModelForCausalLM.from_pretrained(cfg.counterfact_model_path)
        self.counterfact_model_proj_in = ResBlock(cfg.llm_hidden_size, cfg.counterfact_model_rep_dim)
        self.counterfact_model_proj_out = ResBlock(cfg.counterfact_model_rep_dim, cfg.llm_hidden_size)
        self.map_to_edit_llm_logits = map_to_edit_llm_logits

    def forward(self):
        raise
    
    def forward_with_request_embd(self, edit_embeds:List[torch.Tensor], input_embeds:List[torch.Tensor]):
        #  edit_embeds: b * [1, l1?, D]; input_embeds: b * [1, l2?, D]
        logits = []
        for ee, ie in zip(edit_embeds, input_embeds):
            inpt = self.counterfact_model_proj_in(torch.cat([ee, ie], 1)) # [1, l1 + l2, d]
            mask = torch.ones([1, inpt.shape[1]], dtype=torch.long, device = inpt.device) # [1, l1 + l2]
            last_layer_outpt = self.counterfact_model(inputs_embeds = inpt, 
                attention_mask = mask, output_hidden_states = True).hidden_states[-1] # [1, l1 + l2, d]
            l = self.map_to_edit_llm_logits(self.counterfact_model_proj_out(last_layer_outpt)) # [1, l1 + l2, V]
            l = l[:, ee.shape[1]:] # # [1, l2, V]
            logits.append(l)
        return logits

    