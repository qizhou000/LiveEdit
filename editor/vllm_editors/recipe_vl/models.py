#%%
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from torch import nn
import torch

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Knowledge Representation Model
class KnowledgeRepModel(nn.Module):
    def __init__(self, input_dim, base_model_dim = 768, reps_dim = 2048, 
            prot_token_n = 10, device = 'cuda:0', base_path = 'models/roberta-base', 
            truncation = 512) -> None:
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(base_path)
        self.base_model = RobertaModel.from_pretrained(base_path)
        self.knowl_trans_mlp1 = nn.Linear(4 * 768, reps_dim, True)
        self.knowl_trans_mlp2 = nn.Sequential(nn.Linear(reps_dim, reps_dim, True) , nn.ReLU())
        self.query_trans_mlp1 = nn.Linear(4 * 768, reps_dim, True)
        self.query_trans_mlp2 = nn.Sequential(nn.Linear(reps_dim, reps_dim, True) , nn.ReLU())
        self.prot_tokens = nn.Parameter((torch.rand([1, prot_token_n, 768]) - 0.5) * 0.1)
        self.truncation = truncation
        self.device = device
        self.to(device)

    def forward(self): 
        raise 

    def get_inpt_reps(self, edit_sentences:List[str], knowl_or_query:str):
        # edit_sentences: [sent1, sent2, ..., sentn]
        inpt = self.tokenizer(edit_sentences, return_tensors='pt', padding=True,
                truncation=True).to(self.device)
        bert_out = self.base_model(**inpt) 
        reps = self.compute_reps(bert_out.last_hidden_state, bert_out.pooler_output, 
                                 inpt.attention_mask, knowl_or_query)
        return reps # [n, rep_n]
    
    def compute_reps(self, lhs, po, attention_mask, knowl_or_query):
        # hs, po = bert_out.last_hidden_state, bert_out.pooler_output # [n, l, 768], [n, 768]
        mask = attention_mask.unsqueeze(-1)
        ave_lhs = torch.sum(lhs * mask, 1) / torch.sum(mask, 1) # [n, 768]
        max_lhs = torch.max(lhs + (mask - 1) * 999999, 1).values # [n, 768]
        min_lhs = torch.min(lhs + (1 - mask) * 999999, 1).values # [n, 768]
        x = torch.cat([po, ave_lhs, max_lhs, min_lhs], 1) # [n, 4 * 768]
        if knowl_or_query == 'k':
            x = self.knowl_trans_mlp1(x)
            x = self.knowl_trans_mlp2(x) + x
        elif knowl_or_query == 'q':
            x = self.query_trans_mlp1(x)
            x = self.query_trans_mlp2(x) + x
        else:
            raise ValueError
        return x # [n, rep_n]
    
    def get_knowl_rep_prot(self):
        lhs = self.base_model.encoder(self.prot_tokens).last_hidden_state
        po = self.base_model.pooler(lhs)
        attention_mask = torch.ones(self.prot_tokens.shape[:-1], dtype=torch.long, device=self.device)
        rep = self.compute_reps(lhs, po, attention_mask, 'k')
        return rep # [1, rep_n]


    # def get_inpt_reps(self, inpt_states:Dict, knowl_or_query:str):
    #     # inpt_states: {'inputs_embeds': torch.Tensor([b, l, d]), 'attention_mask': ..}
    #     ipt = {}
    #     ipt['inputs_embeds'] = pad_sequence([inpt_states['inputs_embeds'][i][m.sum()-self.truncation:m.sum()] 
    #         for i, m in enumerate(inpt_states['attention_mask'])], True, 0)
    #     ipt['inputs_embeds'] = self.transform_mlp(ipt['inputs_embeds'])
    #     ipt['attention_mask'] = pad_sequence([inpt_states['attention_mask'][i][m.sum()-self.truncation:m.sum()] 
    #         for i, m in enumerate(inpt_states['attention_mask'])], True, 0)
    #     bert_out = self.base_model(**ipt) 
    #     reps = self.compute_reps(bert_out.last_hidden_state, bert_out.pooler_output, 
    #                              ipt['attention_mask'], knowl_or_query)
    #     return reps # [n, rep_n]
     
# model = KnowledgeRepModel()
# edit_sentences = ['2 a']
# model.get_knowledge_rep_prototype().shape
# #%%
    
class PromptTransformer(nn.Module):
    def __init__(self, in_dim = 2048, out_dim = 1600, prompt_token_n = 3, device = 'cuda:0') -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, in_dim, True), nn.ReLU()
        )
        self.l2 = nn.Linear(in_dim, out_dim * prompt_token_n, True)
        self.prompt_token_n = prompt_token_n
        self.to(device)
    def forward(self, knowledge_reps:torch.Tensor):
        # knowledge_reps: [batch_size, in_dim]
        x = self.l1(knowledge_reps) + knowledge_reps
        x = self.l2(x).reshape(knowledge_reps.shape[0], self.prompt_token_n, -1)
        return x # [batch_size, prompt_token_n, out_dim]
    
