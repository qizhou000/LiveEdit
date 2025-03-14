from .models import KnowledgeRepModel, PromptTransformer
from ...vllms_for_edit.base import BaseVLLMForEdit
from ..base import VLLMBaseEditorWithTraining
from torch.nn.utils.rnn import pad_sequence
from dataset.vllm import BaseVLLMEditData
from PIL.Image import Image as PILImage
from typing import Dict, List, Tuple
from types import SimpleNamespace
from dataclasses import dataclass
from ...base import BaseConfig
from torch.optim import Adam
import numpy as np
import torch, yaml


@dataclass
class RECIPEvlConfig(BaseConfig):
    @dataclass
    class TrainConfig():
        krm_lr: float
        pt_lr: float
        relia_lambda: float
        gen_lambda: float
        loc_lambda: float
        contra_lambda: float
        query_knowledge_t: float
        query_prototype_t: float
        constra_hinge_scale: float # w/hinge >= 1, w/o hinge== 999999 
        edit_hinge_scale: float # w/hinge >= 1, w/o hinge== 999999 
        # set in train_init
        batch_size:int = None
        sample_count:int = None
        random_seed:int = None
        eps:float = 1e-8
    @dataclass
    class KRMConfig():
        krm_base_path: str
        krm_base_dim: int
        prompt_token_n: int
        knowledge_rep_dim: int
        knowl_rep_prot_token_n: int
    edit_model_name: str
    model_hidden_size: int
    retr_top_k: int
    train: TrainConfig
    krm: KRMConfig

    @classmethod
    def from_yaml(self, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        data['train'] = self.TrainConfig(**data['train'])
        data['krm'] = self.KRMConfig(**data['krm'])
        return self(**data)
    @classmethod
    def from_json(self, fpath):
        raise
    
class RECIPEvl(VLLMBaseEditorWithTraining):
    def __init__(self, vllm: BaseVLLMForEdit, config: RECIPEvlConfig, device:str = 'cuda:0'):
        super().__init__(vllm, config, device)
        self.cfg = config
        # initialize parameters
        self.knowl_rep_model = KnowledgeRepModel(config.model_hidden_size, config.krm.krm_base_dim,
            config.krm.knowledge_rep_dim, config.krm.knowl_rep_prot_token_n, self.device, 
            config.krm.krm_base_path).to(self.device)
        self.prompt_transformer = PromptTransformer(config.krm.knowledge_rep_dim, 
            config.model_hidden_size, config.krm.prompt_token_n, self.device).to(self.device)
        # initialize vllm
        self.wrap_vllm_get_llm_input_embeds(self.vllm)
        self.wrap_vllm_get_llm_outpt(self.vllm)
        # initialize editing
        self.restore_to_original_model()
        self.set_train(False)
        self.is_editing = False
    
    ############################################################################
    ############################# Initialize ###################################
    ############################################################################
    def wrap_vllm_get_llm_input_embeds(self, vllm:BaseVLLMForEdit):
        '''Get retrieved knowledge ids.'''
        def wrap(get_llm_input_embeds):
            def wrapped_get_llm_input_embeds(texts: List[str], imgs: List[PILImage]):
                if self.is_train or self.is_editing:
                    return get_llm_input_embeds(texts, imgs)
                query_reps = self.knowl_rep_model.get_inpt_reps(texts, knowl_or_query = 'q') # [b, knowledge_rep_dim] 
                sim_matrx = (query_reps @ self.knowl_reps_pool.T) / self.cfg.krm.knowledge_rep_dim**0.5
                sim_with_prototype = sim_matrx[:, :1] 
                sorted_sim, order = torch.sort(sim_matrx, 1, True) # [b, edit_n]
                mask = sorted_sim[:, :self.cfg.retr_top_k] > sim_with_prototype
                retrieved_ids = torch.masked_select(order[:, :self.cfg.retr_top_k], mask)
                retrieved_ids = torch.split(retrieved_ids, mask.sum(1).tolist()) # retrieved indexes
                input_embeds, vt_range = get_llm_input_embeds(texts, None) 
                input_embeds['retrieved_ids'] = retrieved_ids # [retr_ids_1, retr_ids_2, ..., retr_ids_n]
                input_embeds['sorted_sim_order'] = (sorted_sim, order)
                return input_embeds, vt_range
            return wrapped_get_llm_input_embeds
        if not hasattr(vllm, 'original_get_llm_input_embeds'):
            vllm.original_get_llm_input_embeds = vllm.get_llm_input_embeds
            vllm.get_llm_input_embeds = wrap(vllm.get_llm_input_embeds)
        else:
            vllm.get_llm_input_embeds = wrap(vllm.original_get_llm_input_embeds)

    def wrap_vllm_get_llm_outpt(self, vllm:BaseVLLMForEdit):
        def wrap(get_llm_outpt):
            def wrapped_get_llm_outpt(input_embeds, vt_range):
                if self.is_train:
                    return get_llm_outpt(input_embeds, vt_range)
                adopted_prompts = [self.prompts_pool[ids].reshape(
                    len(ids)*self.cfg.krm.prompt_token_n, self.cfg.model_hidden_size) 
                    for ids in input_embeds['retrieved_ids']]
                ipt = {
                    'attention_mask': pad_sequence([torch.ones(len(ap) + torch.sum(m), device=self.device) 
                        for m, ap in zip(input_embeds['attention_mask'], adopted_prompts)], True, 0), 
                    'inputs_embeds': pad_sequence([torch.cat([ap, e[:torch.sum(m)]])
                    for e, m, ap in zip(input_embeds['inputs_embeds'], input_embeds['attention_mask'], adopted_prompts)], True, 0)
                }
                logits = get_llm_outpt(ipt, vt_range).logits
                logits = pad_sequence([l[len(ap):int(torch.sum(m))] for l, ap, m in 
                    zip(logits, adopted_prompts, ipt['attention_mask'])], True, 0)
                otpt = SimpleNamespace()
                otpt.logits = logits
                return otpt
            return wrapped_get_llm_outpt
        if not hasattr(vllm, 'original_get_llm_outpt'):
            vllm.original_get_llm_outpt = vllm.get_llm_outpt
            vllm.get_llm_outpt = wrap(vllm.get_llm_outpt)
        else:
            vllm.get_llm_outpt = wrap(vllm.original_get_llm_outpt)

    ############################################################################
    ############################# Editor Basic Functions #######################
    ############################################################################
    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'recipe_vl', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return True
 
    def restore_to_original_model(self):
        self.request_pool = ['<Knowledge_Representation_Prototype>'] # [edit_n, knowledge_rep_dim]
        self.knowl_reps_pool = self.knowl_rep_model.get_knowl_rep_prot() # [edit_n, knowledge_rep_dim]
        self.prompts_pool = torch.zeros([1, self.cfg.krm.prompt_token_n, self.cfg.model_hidden_size], 
            device = self.device) # [edit_n, prompt_token_n, model_hidden_size]

    def edit_batch(self, requests: List[Dict]):
        '''requests = [
            {'image': PILImage, 'prompt': str, 'target_new': str, ...},
            {'image': PILImage, 'prompt': str, 'target_new': str, ...}, ...] '''
        for r in requests:
            self.edit_one_piece(r)

    def edit_one_piece(self, request: Dict) -> None:
        """request = {'image': PILImage, 'prompt': str, 'target_new': str, ...}"""
        self.is_editing = True
        self.request_pool.append(request)
        if request['prompt'][-1] != ' ' and request['target_new'][0] != ' ':
            t = request['prompt'] + ' ' + request['target_new']
        else:
            t = request['prompt'] + request['target_new']
        knowl_reps = self.knowl_rep_model.get_inpt_reps([t], knowl_or_query = 'k') # [b, knowledge_rep_dim] 
        prompt = self.prompt_transformer(knowl_reps)
        self.knowl_reps_pool = torch.cat([self.knowl_reps_pool, knowl_reps], 0)
        self.prompts_pool = torch.cat([self.prompts_pool, prompt], 0)
        self.is_editing = False

    ############################################################################
    ######################### RECIPEvl Training ################################
    ############################################################################
    def train_a_batch(self, a_batch_samples):
        knowl_reps, rg_q1, rg_q2, loc_q, rel_xyms, gen_xyms, loc_xyms = a_batch_samples
        loss = 0 
        bsz = len(knowl_reps)
        eps = self.cfg.train.eps
        q1_reps = torch.cat([self.knowl_rep_model.get_inpt_reps([q], 'q') for q in rg_q1])  # [bsz, rep_dim]
        q2_reps = torch.cat([self.knowl_rep_model.get_inpt_reps([q], 'q') for q in rg_q2]) # [bsz, rep_dim]
        knowl_reps = torch.cat([self.knowl_rep_model.get_inpt_reps([r], 'k') for r in knowl_reps]) # [bsz, rep_dim]
        prot_reps = self.knowl_rep_model.get_knowl_rep_prot() # [1, rep_dim]
        knowl_reps_with_proto = torch.cat([knowl_reps, prot_reps]) # [bsz + 1, rep_dim]
        scale_factor = 1 / self.cfg.krm.knowledge_rep_dim**0.5
        chs = self.cfg.train.constra_hinge_scale
        # reliability/generality loss_contra_q1
        sim_q1 = (q1_reps @ knowl_reps_with_proto.T) * scale_factor # [bsz, bsz+1]
        sim_q1 = torch.softmax(sim_q1 * self.cfg.train.query_knowledge_t, 1)
        loss_contra_q1 = - torch.log(torch.diag(sim_q1) + eps).mean(0)
        # reliability/generality loss_contra_q2
        sim_q2 = (q2_reps @ knowl_reps.T) * scale_factor # [bsz, bsz]
        sim_q2 = sim_q2 * (1 - torch.eye(bsz, device=self.device))
        sim_q2 = sim_q2 + torch.diag((q2_reps @ prot_reps.T)[:, 0] * scale_factor)
        sim_q2 = torch.softmax(sim_q2 * self.cfg.train.query_prototype_t, 1)
        second_sim_q2 = torch.topk(sim_q2, 2, 1).values[:, 1]
        sim_q2 = torch.diag(sim_q2) 
        sim_q2 = torch.masked_select(sim_q2, sim_q2 < second_sim_q2 * chs)
        if len(sim_q2) == 0:
            loss_contra_q2 = 0
        else:
            loss_contra_q2 = - torch.log(sim_q2 + eps).mean(0) 
        # locality loss_contra_q3 
        loss_contra_q3 = 0
        q3_reps = torch.cat([self.knowl_rep_model.get_inpt_reps([q], 'q') for q in loc_q]) # [bsz, rep_dim]
        sim_q3 = (q3_reps @ knowl_reps_with_proto.T) * scale_factor # [bsz, bsz+1]
        sim_q3 = torch.softmax(sim_q3 * self.cfg.train.query_prototype_t, 1)
        second_sim_q3 = torch.topk(sim_q3, 2, 1).values[:, 1]
        sim_q3 = sim_q3[:, -1]
        sim_q3 = torch.masked_select(sim_q3, sim_q3 < second_sim_q3 * chs)
        if len(sim_q3) == 0:
            l = 0
        else:
            l = - torch.log(sim_q3 + eps).mean(0) 
        loss_contra_q3 += l
        # sum
        loss_contra = loss_contra_q1 + loss_contra_q2 + loss_contra_q3
        loss += loss_contra * self.cfg.train.contra_lambda
        # edit loss
        adopted_prompts = [i.unsqueeze(0) for i in self.prompt_transformer(knowl_reps)]
        # compute reliability loss
        rel_losses = []
        for ((input_embeds, vt_range), label_ids, label_masks), ap in zip(rel_xyms, adopted_prompts):
            ipt_embd = {
                'attention_mask': torch.ones([1, ap.shape[1] + torch.sum(input_embeds['attention_mask'])], device=self.device), # [1, 3+l]
                'inputs_embeds': torch.cat([ap, input_embeds['inputs_embeds']], 1)# [1, 3+l, d]
            }
            logits = self.vllm.get_llm_outpt(ipt_embd, vt_range).logits[:, ap.shape[1]:]
            rel_losses.append(self.vllm.label_loss(logits, label_ids, label_masks))
        rel_loss = sum(rel_losses)/len(rel_losses)
        loss += rel_loss * self.cfg.train.relia_lambda
        # compute generality loss
        gen_losses = []
        for ((input_embeds, vt_range), label_ids, label_masks), ap in zip(gen_xyms, adopted_prompts):
            ipt_embd = {
                'attention_mask': torch.ones([1, ap.shape[1] + torch.sum(input_embeds['attention_mask'])], device=self.device), # [1, 3+l]
                'inputs_embeds': torch.cat([ap, input_embeds['inputs_embeds']], 1)# [1, 3+l, d]
            }
            logits = self.vllm.get_llm_outpt(ipt_embd, vt_range).logits[:, ap.shape[1]:]
            gen_losses.append(self.vllm.label_loss(logits, label_ids, label_masks))
        gen_loss = sum(gen_losses)/len(gen_losses)
        loss += gen_loss * self.cfg.train.gen_lambda
        # compute locality loss
        loc_losses = []
        for ((input_embeds, vt_range), pre_logits, label_masks), ap in zip(loc_xyms, adopted_prompts):
            ipt_embd = {
                'attention_mask': torch.ones([1, ap.shape[1] + torch.sum(input_embeds['attention_mask'])], device=self.device), # [1, 3+l]
                'inputs_embeds': torch.cat([ap, input_embeds['inputs_embeds']], 1)# [1, 3+l, d]
            }
            logits = self.vllm.get_llm_outpt(ipt_embd, vt_range).logits[:, ap.shape[1]:]
            loc_losses.append(self.vllm.logit_KL_loss(logits, pre_logits, label_masks))
        loc_loss = sum(loc_losses)/len(loc_losses)
        loss += loc_loss * self.cfg.train.loc_lambda
        # update
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        log_dict = {
            'Contrastive loss': float(loss_contra),
            'Reliability loss': float(rel_loss),
            'Generality loss': float(gen_loss),
            'Locality loss': float(loc_loss)
        }
        return float(loss), log_dict

    def preprocess_train_data(self, vllm_edit_data:BaseVLLMEditData)->List:
        return vllm_edit_data.data_with_img
        
    def organize_batch_data(self, a_batch_of_training_data:List):
        if not hasattr(self, 'rng_data_proc'):
            self.rng_data_proc = np.random.default_rng(self.random_seed)
        def add_space(p, a):
            return '%s %s'%(p, a) if p[-1] != ' ' and a[0] != ' ' else '%s%s'%(p, a)
        def random_select_rg(d):
            if self.rng_data_proc.integers(0, 2) == 0: # rel
                t = d['requests'][0]['prompt']
            else: # gen
                gen_name = list(d['generality'].keys())[self.rng_data_proc.integers(0, len(d['generality'].keys()))]
                gen_d = d['generality'][gen_name][self.rng_data_proc.integers(0, len(d['generality'][gen_name]))]
                t = gen_d['prompt']
            return t
        def random_select_l(d):
            loc_name = list(d['locality'].keys())[self.rng_data_proc.integers(0, len(d['locality'].keys()))]
            loc_d = d['locality'][loc_name][self.rng_data_proc.integers(0, len(d['locality'][loc_name]))]
            return loc_d['prompt']
        knowl_reps, rg_q1, rg_q2, loc_q = [], [], [], []
        rel_xyms, gen_xyms, loc_xyms = [], [], []
        for d in a_batch_of_training_data:
            knowl_reps.append(add_space(d['requests'][0]['prompt'], d['requests'][0]['target_new']))
            rg_q1.append(random_select_rg(d))
            rg_q2.append(random_select_rg(d))
            loc_q.append(random_select_l(d))
            rel_xyms.append(self.vllm.prompts_imgs_target_to_xym([d['requests'][0]['prompt']], 
                [d['requests'][0]['image']], [d['requests'][0]['target_new']]))
            gen_name = list(d['generality'].keys())[self.rng_data_proc.integers(0, len(d['generality'].keys()))]
            gen_d = d['generality'][gen_name][self.rng_data_proc.integers(0, len(d['generality'][gen_name]))]
            gen_xyms.append(self.vllm.prompts_imgs_target_to_xym(
                [gen_d['prompt']], [gen_d['image']], [gen_d['target']]))
            loc_name = list(d['locality'].keys())[self.rng_data_proc.integers(0, len(d['locality'].keys()))]
            loc_d = d['locality'][loc_name][self.rng_data_proc.integers(0, len(d['locality'][loc_name]))]
            (input_embeds, vt_range), label_ids, label_masks = self.vllm.prompts_imgs_target_to_xym(
                [loc_d['prompt']], [loc_d['image']], [loc_d['target']])
            pre_logits = self.vllm.get_llm_outpt(input_embeds, vt_range).logits 
            loc_xyms.append(((input_embeds, vt_range), pre_logits, label_masks))
        return knowl_reps, rg_q1, rg_q2, loc_q, rel_xyms, gen_xyms, loc_xyms
        
    def get_modules_for_training(self):
        return {'knowl_rep_model': self.knowl_rep_model, 'prompt_transformer': self.prompt_transformer}

    def get_a_new_optimizer(self):
        return Adam([{'params': self.knowl_rep_model.parameters(), 'lr': self.cfg.train.krm_lr},
            {'params': self.prompt_transformer.parameters(), 'lr': self.cfg.train.krm_lr}])

    def set_train(self, is_train = False):
        self.is_train = is_train
        self.vllm.model.train(False)
        self.vllm.model.requires_grad_(False)
        self.knowl_rep_model.train(is_train)
        self.knowl_rep_model.requires_grad_(is_train)
        self.prompt_transformer.train(is_train)
        self.prompt_transformer.requires_grad_(is_train)

    def other_train_init_final(self):
        pass

    def other_train_init_begin(self):
        self.set_train(True)

    def reinit_train_parameters(self):
        pass
