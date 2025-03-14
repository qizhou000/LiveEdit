from sentence_transformers.SentenceTransformer import SentenceTransformer
from ...vllms_for_edit.base import BaseVLLMForEdit
from ..base import VLLMBaseEditorWithTraining
from dataset.vllm import BaseVLLMEditData
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Union
from utils import move_to_device, find_module
from types import SimpleNamespace
import torch.nn.functional as F
from ...base import BaseConfig
from torch.optim import Adam
import torch, os, yaml
from torch import nn

@dataclass
class LTEvlConfig(BaseConfig):
    @dataclass
    class TrainConfig():
        lr: float
        relia_lambda: float
        gen_lambda: float
        loc_lambda: float
    edit_model_name: str
    train_config: TrainConfig
    fine_tune_modules_path: Union[str, List]
    retriever_path: str
    retrieval_embed_dim: int
    sim_threshold: float

    @classmethod
    def from_yaml(self, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        data['train_config'] = self.TrainConfig(**data['train_config'])
        return self(**data)
    @classmethod
    def from_json(self, fpath):
        raise

class LTEvl(VLLMBaseEditorWithTraining):
    def __init__(self, vllm: BaseVLLMForEdit, config: LTEvlConfig, device = 'cuda',
            vllm_proc_data: BaseVLLMForEdit = None, device_proc_data:str = None):
        super().__init__(vllm, config, device)
        if vllm_proc_data != None:
            self.vllm_proc_data = vllm_proc_data
            self.vllm_proc_data.model.eval()
            self.vllm_proc_data.model.requires_grad_(False)
            self.device_proc_data = device_proc_data
        self.cfg = config
        self.retrieval_model = SentenceTransformer(self.cfg.retriever_path, device = device)
        self.edit_sign = '[Updated Information]'
        self.query_sign = '\n[Query]'
        self.now_input_texts = None # for retrieval
        self.now_input_imgs = None # for retrieval
        # hook for VLLM output function
        self.wrap_get_llm_outpt()
        self.set_train(False) 
        self.restore_to_original_model()

    # def wrap_get_llm_outpt(self): # pure editing and no retrieval
    #     def wrap(get_llm_outpt):
    #         def wrapped_get_llm_outpt(input_embeds, vt_range): 
    #             if self.is_train or len(self.edit_requests_pool) == 0:
    #                 return get_llm_outpt(input_embeds, vt_range)
    #             assert len(input_embeds['inputs_embeds']) == 1 # only for inference
    #             logits = self.__get_edited_output__(get_llm_outpt, self.edit_prefix_pool[0], input_embeds).logits
    #             logits = logits[:, self.edit_prefix_pool[0]['attention_mask'].shape[1]:]
    #             res = SimpleNamespace(logits = logits)
    #             return res 
    #         return wrapped_get_llm_outpt
    #     if not hasattr(self, 'original_get_llm_outpt'):
    #         self.original_get_llm_outpt = self.vllm.get_llm_outpt
    #     self.vllm.get_llm_outpt = wrap(self.original_get_llm_outpt)
        
    def wrap_get_llm_outpt(self):
        def wrap(get_llm_outpt):
            def wrapped_get_llm_outpt(input_embeds, vt_range): 
                if self.is_train or len(self.edit_requests_pool) == 0:
                    return get_llm_outpt(input_embeds, vt_range)
                assert len(input_embeds['inputs_embeds']) == 1 # only for inference
                (prompt, image, target) = input_embeds['query_triple']
                rr, retrieved_prefixs, sim = self.retrieval([prompt])
                if retrieved_prefixs[0] == None:
                    return get_llm_outpt(input_embeds, vt_range)
                logits = self.__get_edited_output__(get_llm_outpt, retrieved_prefixs[0], input_embeds).logits
                logits = logits[:, retrieved_prefixs[0]['attention_mask'].shape[1]:]
                res = SimpleNamespace(logits = logits)
                return res 
            return wrapped_get_llm_outpt
        if not hasattr(self, 'original_get_llm_outpt'):
            self.original_get_llm_outpt = self.vllm.get_llm_outpt
        self.vllm.get_llm_outpt = wrap(self.original_get_llm_outpt)

    def retrieval(self, texts: List[str]): 
        # b = len(texts)
        assert isinstance(texts, list) and len(texts) == 1
        text_embds = torch.tensor(self.retrieval_model.encode(texts), device=self.device) # [b, d]
        text_embds_norm = F.normalize(text_embds, p=2, dim=1)
        text_retr_pool_norm = F.normalize(self.text_retr_pool, p=2, dim=1)
        t_sim = text_embds_norm @ text_retr_pool_norm.T # [b, m]
        values, indices = torch.max(t_sim, dim = 1)
        retrieved_requests, retrieved_prefixs = [], []
        for v, i in zip(values, indices):
            if v > self.cfg.sim_threshold:
                retrieved_requests.append(self.edit_requests_pool[i])
                retrieved_prefixs.append(self.edit_prefix_pool[i])
            else:
                retrieved_requests.append(None)
                retrieved_prefixs.append(None)
        return retrieved_requests, retrieved_prefixs, t_sim

    def __get_edit_prefix__(self, vllm:BaseVLLMForEdit, request: Dict) -> torch.Tensor:
        """request = {'image': PILImage, 'prompt': str, 'target_new': str, ...} """
        if request['prompt'][-1] != ' ' and request['target_new'][0] != ' ':
            t = ' ' + request['target_new']
        else:
            t = request['target_new']
        p = self.edit_sign + request['prompt'] + t + self.query_sign
        prefix = vllm.get_llm_input_embeds([p], [request['image']])[0]
        return prefix

    def __get_edited_output__(self, get_llm_outpt, prefix:Dict, original_inpt:Dict):
        inpt = {}
        inpt['attention_mask'] = torch.cat([prefix['attention_mask'], original_inpt['attention_mask']], 1)
        inpt['inputs_embeds'] = torch.cat([prefix['inputs_embeds'], original_inpt['inputs_embeds']], 1)
        return get_llm_outpt(inpt, None)

    ############################################################################
    ############################# Editor Basic Functions #######################
    ############################################################################
    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'lte_vl', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return False
 
    def restore_to_original_model(self):
        self.edit_requests_pool = []
        self.edit_prefix_pool = []
        self.text_retr_pool = torch.zeros([0, self.cfg.retrieval_embed_dim], device=self.device)

    def edit_batch(self, requests: List[Dict]):
        raise

    def edit_one_piece(self, request: Dict) -> None:
        """request = {'image': PILImage, 'prompt': str, 'target_new': str, ...} """
        self.edit_requests_pool.append(request)
        self.edit_prefix_pool.append(self.__get_edit_prefix__(self.vllm, request))
        t_embd = torch.tensor(self.retrieval_model.encode([request['prompt'] + ' ' + request['target_new']]), device=self.device)
        self.text_retr_pool = torch.cat([self.text_retr_pool, t_embd], 0)

    ############################################################################
    ############################# LTE_vl Training ################################
    ############################################################################
    def set_train(self, is_train = False):
        self.is_train = is_train
        self.vllm.model.train(False)
        self.vllm.model.requires_grad_(False)
        for v in self.get_modules_for_training().values():
            v.train(is_train)
            v.requires_grad_(is_train)
        if hasattr(self, 'vllm_proc_data'):
            self.vllm_proc_data.model.train(False)
            self.vllm_proc_data.model.requires_grad_(False)

    def reinit_train_parameters(self):
        pass

    def preprocess_train_data(self, vllm_edit_data:BaseVLLMEditData)->List:
        return vllm_edit_data.data_with_img
        
    def organize_batch_data(self, a_batch_of_training_data:List):
        assert len(a_batch_of_training_data) == 1
        d = a_batch_of_training_data[0]
        edit_prefix = self.__get_edit_prefix__(self.vllm_proc_data, d['requests'][0])
        # rel
        rel_xym = self.vllm_proc_data.prompts_imgs_target_to_xym([d['requests'][0]['prompt']], 
                                    [d['requests'][0]['image']], [d['requests'][0]['target_new']])
        # gen
        gen_xym = {k:self.vllm_proc_data.prompts_imgs_target_to_xym([v[0]['prompt']], 
            [v[0]['image']], [v[0]['target']]) for k, v in d['generality'].items()}
        # loc
        loc_xym = {}
        for k, v in d['locality'].items():
            (input_embeds, vt_range), label_ids, label_masks = self.vllm_proc_data.prompts_imgs_target_to_xym(
                [v[0]['prompt']], [v[0]['image']], [v[0]['target']])
            pre_logits = self.vllm_proc_data.get_llm_outpt(input_embeds, vt_range).logits
            loc_xym[k] = ((input_embeds, vt_range), pre_logits, label_masks)
        return move_to_device((edit_prefix, rel_xym, gen_xym, loc_xym), self.device)
    
    def get_modules_for_training(self)->Dict[str, nn.Module]:
        if isinstance(self.cfg.fine_tune_modules_path, str):
            return {'llm': find_module(self.vllm.model, self.cfg.fine_tune_modules_path)}
        if isinstance(self.cfg.fine_tune_modules_path, list):
            return {n: find_module(self.vllm.model, n) for n in self.cfg.fine_tune_modules_path}
    
    def get_a_new_optimizer(self)->torch.optim.Optimizer:
        opt = Adam([{'params': v.parameters(), 'lr': self.cfg.train_config.lr} 
                    for k,v in self.get_modules_for_training().items()])
        return opt
        
    def other_train_init_final(self):
        self.restore_to_original_model()

    def train_a_batch(self, a_batch_of_training_data): 
        edit_prefix, rel_xym, gen_xym, loc_xym = a_batch_of_training_data
        loss = 0 
        # reliability loss
        (input_embeds, vt_range), label_ids, label_masks = rel_xym
        logits = self.__get_edited_output__(self.vllm.get_llm_outpt, edit_prefix, input_embeds).logits
        rel_loss = label_loss(logits, label_ids, label_masks)
        loss += rel_loss * self.cfg.train_config.relia_lambda
        # generality loss
        gen_losses = {}
        for loss_name, sp in gen_xym.items():
            (input_embeds, vt_range), label_ids, label_masks = sp
            logits = self.__get_edited_output__(self.vllm.get_llm_outpt, edit_prefix, input_embeds).logits
            gen_loss = label_loss(logits, label_ids, label_masks)
            gen_losses[loss_name] = float(gen_loss)
            loss += gen_loss * self.cfg.train_config.gen_lambda
        # locality loss
        loc_losses = {}
        for loss_name, sp in loc_xym.items():
            (input_embeds, vt_range), pre_logits, label_masks = sp
            logits1 = self.vllm.get_llm_outpt(input_embeds, vt_range).logits
            logits2 = self.__get_edited_output__(self.vllm.get_llm_outpt, edit_prefix, input_embeds).logits
            loc_loss = (logit_KL_loss(pre_logits, logits1, label_masks) + 
                        logit_KL_loss(pre_logits, logits2, label_masks)) / 2
            loc_losses[loss_name] = float(loc_loss)
            loss += loc_loss * self.cfg.train_config.loc_lambda
        # update
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        log_dict = {
            'Reliability loss': float(rel_loss),
            'Generality loss': gen_losses,
            'Locality loss': loc_losses
        }
        return float(loss), log_dict
    

 
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
