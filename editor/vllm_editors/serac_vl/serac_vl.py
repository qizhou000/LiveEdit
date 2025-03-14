from ...vllms_for_edit.base import BaseVLLMForEdit
from utils import find_module, move_to_device
from ..base import VLLMBaseEditorWithTraining
from torch.nn.utils.rnn import pad_sequence
from dataset.vllm import BaseVLLMEditData
from typing import Dict, List, Tuple
from dataclasses import dataclass
from types import SimpleNamespace
from ...base import BaseConfig
import os, torch, json, yaml
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm


@dataclass
class SERACvlConfig(BaseConfig):
    @dataclass
    class TrainConfig():
        lr: float
        relia_lambda: float
        gen_lambda: float
        loc_lambda: float
    edit_model_name: str
    counterfact_model_path: str
    counterfact_model_rep_dim: int
    classifier_path: str
    classifier_rep_dim: int
    llm_hidden_size: int
    train_config: TrainConfig
    llm_norm_path: str
    llm_voc_path: str
    
    @classmethod
    def from_yaml(self, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        data['train_config'] = self.TrainConfig(**data['train_config'])
        return self(**data)

class SERACvl(VLLMBaseEditorWithTraining):
    def __init__(self, vllm: BaseVLLMForEdit, config: SERACvlConfig, device:str = 'cuda:0'):
        super().__init__(vllm, config, device)
        self.cfg = config
        # Initialize train modules
        from .modules import CounterfactModel, Classifier
        llm_norm = find_module(vllm.model, config.llm_norm_path)
        voc_path = find_module(vllm.model, config.llm_voc_path)
        def reps_to_word_predict(reps:torch.Tensor):
            return voc_path(llm_norm(reps))
        self.counterfact_model = CounterfactModel(config, reps_to_word_predict).to(device)
        self.classifier = Classifier(config).to(device)
        # hook llm `get_llm_outpt` function
        self.wrap_get_llm_outpt()
        self.restore_to_original_model()
        self.set_train(False)
    
    ############################################################################
    #                            Initialize                                    #
    ############################################################################
    def wrap_get_llm_outpt(self):
        def wrap(get_llm_outpt):
            def wrapped_get_llm_outpt(input_embeds, vt_range):
                if self.is_train or len(self.request_pool) == 0:
                    return get_llm_outpt(input_embeds, vt_range)
                assert len(input_embeds['attention_mask']) == 1
                qr = input_embeds['query_range']
                rep1 = self.classifier.get_sim_reps(input_embeds['inputs_embeds'][:, qr[0]:qr[1]], 
                    input_embeds['attention_mask'][:, qr[0]:qr[1]])
                sim = self.classifier.get_sim(rep1, self.sim_reps_pool)
                v, i = torch.max(sim, dim=1) # [1], [1]
                if v[0] >= 10:
                    logits = self.counterfact_model.forward_with_request_embd(
                        [self.request_embed_pool[i]], [input_embeds['inputs_embeds']])[0]
                else:
                    logits = get_llm_outpt(input_embeds, vt_range).logits
                otpt = SimpleNamespace()
                otpt.logits = logits
                return otpt
            return wrapped_get_llm_outpt
        if not hasattr(self.vllm, 'original_get_llm_outpt'):
            self.vllm.original_get_llm_outpt = self.vllm.get_llm_outpt
        self.vllm.get_llm_outpt = wrap(self.vllm.original_get_llm_outpt)

    ############################################################################
    #         Implementation Virtual Functions of Base Class                   #
    ############################################################################
    def name_of_editor_and_model(self):
        return 'serac_vl', self.cfg.edit_model_name

    def if_can_batch_edit(self)->bool:
        return False

    def restore_to_original_model(self):
        self.request_pool = []
        self.sim_reps_pool = torch.zeros([0, self.cfg.classifier_rep_dim], device=self.device)
        self.request_embed_pool = []

    def edit_one_piece(self, request: Dict):
        """request = {'image': PILImage, 'prompt': str, 'target_new': str, ...} """
        (input_embeds, _), _, _ = self.vllm.prompts_imgs_target_to_xym(
            [request['prompt']], [request['image']], [request['target_new']])
        reps = self.classifier.get_sim_reps(input_embeds['inputs_embeds'], input_embeds['attention_mask'])
        self.sim_reps_pool = torch.cat([self.sim_reps_pool, reps], 0)
        self.request_embed_pool.append(input_embeds['inputs_embeds'])
        self.request_pool.append(request)

    def edit_batch(self, requests: List[Dict]):
        raise

    ############################################################################
    #                       Training functions                                 #
    ############################################################################
    def get_modules_for_training(self)->Dict[str, nn.Module]:
        return {'counterfact_model': self.counterfact_model, 'classifier': self.classifier}
    
    def reinit_train_parameters(self):
        pass

    def preprocess_train_data(self, vllm_edit_data:BaseVLLMEditData)->List:
        return vllm_edit_data.data
        
    def organize_batch_data(self, a_batch_of_training_data:List):
        vllm = self.vllm
        edit_xyms = []
        gen_xyms = {k: [] for k in a_batch_of_training_data[0]['generality'].keys()}
        loc_xyms = {k: [] for k in a_batch_of_training_data[0]['locality'].keys()}
        for d in a_batch_of_training_data:
            # edit/rel
            prompts = [d['requests'][0]['prompt']]
            imgs = [d['requests'][0]['image']]
            targets = [d['requests'][0]['target_new']]
            q_embeds = vllm.get_llm_input_embeds(prompts, imgs)[0]
            edit_xyms.append((q_embeds, vllm.prompts_imgs_target_to_xym(prompts, imgs, targets)))
            # gen
            for k in gen_xyms.keys():
                prompts = [d['generality'][k][0]['prompt']]
                imgs = [d['generality'][k][0]['image']]
                targets = [d['generality'][k][0]['target']]
                q_embeds = vllm.get_llm_input_embeds(prompts, imgs)[0]
                gen_xyms[k].append((q_embeds, vllm.prompts_imgs_target_to_xym(prompts, imgs, targets)))
            # loc
            for k in loc_xyms.keys():
                prompts = [d['locality'][k][0]['prompt']]
                imgs = [d['locality'][k][0]['image']]
                targets = [d['locality'][k][0]['target']]
                q_embeds = vllm.get_llm_input_embeds(prompts, imgs)[0]
                (input_embeds, vt_range), label_ids, label_masks = vllm.prompts_imgs_target_to_xym(prompts, imgs, targets)
                logits = self.vllm.get_llm_outpt(input_embeds, vt_range).logits
                loc_xyms[k].append((q_embeds, ((input_embeds, vt_range), logits, label_masks)))
        a_batch_of_training_data = move_to_device((edit_xyms, gen_xyms, loc_xyms), self.device)
        return a_batch_of_training_data
        
    def train_a_batch(self, a_batch_of_training_data:Tuple):
        edit_xyms, gen_xyms, loc_xyms = a_batch_of_training_data
        loss = 0  
        log_dict = {}
        batch_size = len(edit_xyms)
        # collect edit representations
        edit_reps, edit_input_embeds = [], []
        for _, ((input_embeds, _), _, _) in edit_xyms:
            edit_reps.append(self.classifier.get_sim_reps(
                input_embeds['inputs_embeds'], input_embeds['attention_mask']))
            edit_input_embeds.append(input_embeds['inputs_embeds'])
        edit_reps = torch.cat(edit_reps, 0) # [b, d]
        # reliability edit loss
        loss_rel_edit = 0
        rel_sim_reps = []
        for (rel_embeds, ((input_embeds, vt_range), label_ids, label_masks)), eie in zip(edit_xyms, edit_input_embeds):
            logits = self.counterfact_model.forward_with_request_embd([eie], [input_embeds['inputs_embeds']])[0]
            loss_rel_edit += label_loss(logits, label_ids, label_masks)
            rel_sim_reps.append(self.classifier.get_sim_reps(
                rel_embeds['inputs_embeds'], rel_embeds['attention_mask']))
        rel_sim_reps = torch.cat(rel_sim_reps) # [b, d]
        rel_sim = self.classifier.get_sim(rel_sim_reps, edit_reps)
        loss_rel_clas = - torch.log(torch.diag(torch.softmax(rel_sim, 1)) + 1e-8).mean()
        loss_rel_edit = loss_rel_edit / batch_size
        loss += loss_rel_edit + loss_rel_clas
        log_dict['Reliability edit loss'] = float(loss_rel_edit)
        log_dict['Reliability class loss'] = float(loss_rel_clas)
        # generality edit loss
        for k in gen_xyms.keys():
            loss_gen_edit = 0
            gen_sim_reps = []
            for (gen_embeds, ((input_embeds, vt_range), label_ids, label_masks)), eie in zip(gen_xyms[k], edit_input_embeds):
                logits = self.counterfact_model.forward_with_request_embd([eie], [input_embeds['inputs_embeds']])[0]
                loss_gen_edit += label_loss(logits, label_ids, label_masks)
                gen_sim_reps.append(self.classifier.get_sim_reps(
                    gen_embeds['inputs_embeds'], gen_embeds['attention_mask']))
            gen_sim_reps = torch.cat(gen_sim_reps) # [b, d]
            gen_sim = self.classifier.get_sim(gen_sim_reps, edit_reps)
            # loss_gen_clas = bin_cross_entropy(gen_sim, torch.eye(batch_size, device = self.device)).mean()
            loss_gen_clas = - torch.log(torch.diag(torch.softmax(gen_sim, 1)) + 1e-8).mean()
            loss_gen_edit = loss_gen_edit / batch_size
            loss += loss_gen_edit + loss_gen_clas
            log_dict['Generality-%s edit loss'%k] = float(loss_gen_edit)
            log_dict['Generality-%s class loss'%k] = float(loss_gen_clas)
        # locality edit loss
        for k in loc_xyms.keys():
            loss_loc_edit = 0
            loc_sim_reps = []
            for (loc_embeds, ((input_embeds, vt_range), pre_logits, label_masks)), eie in zip(loc_xyms[k], edit_input_embeds):
                logits = self.counterfact_model.forward_with_request_embd([eie], [input_embeds['inputs_embeds']])[0]
                loss_loc_edit += logit_KL_loss(logits, pre_logits, label_masks)
                loc_sim_reps.append(self.classifier.get_sim_reps(loc_embeds['inputs_embeds'], loc_embeds['attention_mask']))
            loc_sim_reps = torch.cat(loc_sim_reps) # [b, d]
            loc_sim = self.classifier.get_sim(loc_sim_reps, edit_reps)
            loc_sim = torch.cat([loc_sim, 10 + torch.zeros([loc_sim.shape[0], 1], device=self.device)], 1)
            loss_loc_clas = - torch.log(torch.softmax(loc_sim, 1) + 1e-8)[:, -1].mean()
            loss_loc_edit = loss_loc_edit / batch_size
            loss += loss_loc_edit + loss_loc_clas
            log_dict['Locality-%s edit loss'%k] = float(loss_loc_edit)
            log_dict['Locality-%s class loss'%k] = float(loss_loc_clas)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return float(loss), log_dict

    def get_a_new_optimizer(self)->torch.optim.Optimizer:
        return Adam([{'params': self.classifier.parameters(), 'lr': self.cfg.train_config.lr},
                {'params': self.counterfact_model.parameters(), 'lr': self.cfg.train_config.lr}])
    
    def set_train(self, is_train = False):
        self.is_train = is_train
        self.vllm.model.train(False)
        self.vllm.model.requires_grad_(False)
        self.classifier.requires_grad_(is_train)
        self.classifier.train(is_train)
        self.counterfact_model.requires_grad_(is_train)
        self.counterfact_model.train(is_train)

    def other_train_init_final(self):
        pass

def bin_cross_entropy(reps, y):
    # reps, y: [b, l]
    eps = 1e-8
    loss = - y * torch.log(reps + eps) - (1 - y) * torch.log(1 - reps + eps)
    return loss
 
def label_loss(logits, label_ids, masks, average = True):
    # logits: [batch_size, total_l, d], label_ids/masks: [batch_size, short_l]
    logits = logits[:, -label_ids.shape[1]:]
    log_pre_p = torch.log_softmax(logits, -1)
    log_pre_p = log_pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, short_l]
    loss = -(log_pre_p * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss

def logit_KL_loss(logits1, logits2, masks, average = True):
    # logits1/logits2: [batch, total_l, voc_size], masks: [batch, short_l]
    logits1 = logits1[:, -masks.shape[1]:]
    logits2 = logits2[:, -masks.shape[1]:]
    log_p1 = torch.log_softmax(logits1, -1)
    log_p2 = torch.log_softmax(logits2, -1)
    p1 = torch.softmax(logits1, 2)
    kl_loss = (p1 * (log_p1 - log_p2)).sum(2) # [batch, max_len]
    loss = (kl_loss * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss
