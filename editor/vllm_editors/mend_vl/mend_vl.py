from ...vllms_for_edit.base import BaseVLLMForEdit
from .auxiliary_networks import GradientTransform
from utils import find_module, move_to_device
from ..base import VLLMBaseEditorWithTraining
import transformers, os, torch, json, yaml
from dataset.vllm import BaseVLLMEditData
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass
from ...base import BaseConfig
from torch.optim import Adam
import torch.nn as nn


@dataclass
class MENDvlConfig(BaseConfig):
    @dataclass
    class AuxModelConfig():
        n_hidden: int
        hidden_dim: int
        init: str
        norm: bool
        act: str
        rank: int
        shared: bool
        lr: float
    edit_modules: List[str]
    init_edit_lr: float
    edit_lr_lr: float
    aux_model: AuxModelConfig
    edit_model_name: str
    relia_lambda: float
    gen_lambda: float
    loc_lambda: float

    @classmethod
    def from_yaml(self, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        data['aux_model'] = self.AuxModelConfig(**data['aux_model'])
        return self(**data)
    @classmethod
    def from_json(self, fpath):
        raise


class EditLinear():
    def __init__(self, edit_module:nn.Linear, module_name:str, idx:int, aux_model_weight:GradientTransform, 
                 config:MENDvlConfig, aux_model_bias:GradientTransform = None) -> None:
        assert type(edit_module) == transformers.pytorch_utils.Conv1D or\
                type(edit_module) == nn.Linear
        self.edit_module = edit_module
        self.original_weight = self.edit_module.weight.clone()
        self.original_bias = self.edit_module.bias.clone() if hasattr(self.edit_module, 'bias') and self.edit_module.bias != None else None
        self.module_name = module_name
        self.idx = idx
        self.aux_model_weight = aux_model_weight
        self.aux_model_bias = aux_model_bias
        self.lr = nn.Parameter(torch.tensor(config.init_edit_lr))
        self.register_hooks()
        self.clear_delta(True, True, True)

    def register_hooks(self):
        def forward_x_hook(module, args, output):
            # args: tuple(input) [batch_size, max_length, dim_in]
            # output.shape = [batch_size, max_length, dim_out]
            self.__x__ = args[0].detach()
        def backward_delta_hook(module, grad_input, grad_output):
            # grad_input: tuple(input grad) [batch_size, max_length, dim_in]
            # grad_output: tuple(output grad) [batch_size, max_length, dim_out]
            self.__delta__ = grad_output[0].detach()
        def forward_edit_hook(module, args, output):
            # args: tuple(input) [batch_size, max_length, dim_in]
            # output.shape = [batch_size, max_length, dim_out]
            if self.__delta_weight__ != None:
                output = output + args[0] @ self.__delta_weight__ # Wx + b + W'x = (W + W')x + b
            if self.__delta_bias__ != None:
                output = output + self.__delta_bias__ # Wx + b + W'x + b'
            return output
        if not hasattr(self, 'x_handle') or self.x_handle == None:
            self.x_handle = self.edit_module.register_forward_hook(forward_x_hook)
        if not hasattr(self, 'delta_handle') or self.delta_handle == None:
            self.delta_handle = self.edit_module.register_full_backward_hook(backward_delta_hook)
        if not hasattr(self, 'edit_handle') or self.edit_handle == None:
            self.edit_handle = self.edit_module.register_forward_hook(forward_edit_hook)
     
    # def remove_hooks(self, x_handle:bool, delta_handle:bool, edit_handle:bool):
    #     if x_handle:
    #         self.x_handle.remove()
    #         self.x_handle = None
    #     if delta_handle:
    #         self.delta_handle.remove()
    #         self.delta_handle = None
    #     if edit_handle:
    #         self.edit_handle.remove()
    #         self.edit_handle = None

    def update_delta_weight(self):
        '''
        Compute the weights updates using forward/backward and auxiliary model,
        which will influence model inference in hook function.
        '''
        assert self.__x__ != None and self.__delta__ != None
        # x: [request_num, dim_in], delta:[request_num, dim_out]
        x, delta = self.aux_model_weight(self.__x__, self.__delta__, self.idx)
        if self.__delta_weight__ == None:
            self.__delta_weight__ = x.permute(1, 0) @ delta * self.lr # [dim_in, dim_out]
            self.__now_delta_n__ = len(delta)
            self.__delta_weight__ *= 1/self.__now_delta_n__
        else:
            self.__delta_weight__ *= self.__now_delta_n__
            self.__delta_weight__ += x.permute(1, 0) @ delta * self.lr
            self.__now_delta_n__ += len(delta)
            self.__delta_weight__ *= 1/self.__now_delta_n__ 

    def clear_delta(self, x:bool, delta:bool, edit_weight:bool):
        if x:
            self.__x__ = None
        if delta:
            self.__delta__ = None
        if edit_weight:
            self.__delta_weight__ = None
            self.__delta_bias__ = None

    def update_weight_and_clear_delta(self):
        '''
        Add the delta weights into the weights of module. Then clear the delta 
        weights.
        '''
        if self.__delta_weight__.shape == self.edit_module.weight.shape:
            self.edit_module.weight += self.__delta_weight__.detach()
        else:
            self.edit_module.weight += self.__delta_weight__.permute(1, 0).detach()
        self.clear_delta(False, False, True)

    def restore_to_original_weight(self):
        if self.original_bias != None:
            self.edit_module.load_state_dict({'weight': self.original_weight,
                                          'bias': self.original_bias})
        else:
            self.edit_module.load_state_dict({'weight': self.original_weight})


class MENDvl(VLLMBaseEditorWithTraining):
    def __init__(self, vllm: BaseVLLMForEdit, config: MENDvlConfig, device:str = 'cuda:0',
                 vllm_proc_data: BaseVLLMForEdit = None, device_proc_data:str = None):
        super().__init__(vllm, config, device)
        self.cfg = config
        self.vllm_proc_data = vllm_proc_data
        self.device_proc_data = device_proc_data
        self.edit_modules, self.aux_models, self.autograd_params = self.get_edit_modules()
        self.aux_models = self.aux_models.to(device)
        self.edit_lrs = nn.ParameterList([em.lr for em in self.edit_modules]).to(device)
        self.set_train(False)
    
    ############################################################################
    #         Implementation Virtual Functions of Base Class                   #
    ############################################################################
    def name_of_editor_and_model(self):
        return 'mend_vl', self.cfg.edit_model_name

    def if_can_batch_edit(self)->bool:
        return True

    def restore_to_original_model(self):
        self.clear_module_deltas(True, True, True)
        self.restore_to_original_weights()

    def edit_one_piece(self, request: Dict):
        """request = {'image': PILImage, 'prompt': str, 'target_new': str, ...} """
        self.edit_batch([request])

    def edit_batch(self, requests: List[Dict]):
        '''requests = [
            {'image': PILImage, 'prompt': str, 'target_new': str, ...},
            {'image': PILImage, 'prompt': str, 'target_new': str, ...}, ...] '''
        prompts, imgs, new_targets = [], [], []
        for r in requests: 
            prompts.append(r['prompt'])
            imgs.append(r['image'])
            new_targets.append(r['target_new'])
        (input_embeds, vt_range), label_ids, label_masks = self.vllm.prompts_imgs_target_to_xym(
            prompts, imgs, new_targets)
        self.__edit_batch__(input_embeds, vt_range, label_ids, label_masks)
    
    def __edit_batch__(self, input_embeds, vt_range, label_ids, label_masks):
        # input_ids/label_ids/masks: [batch, max_len]
        self.autograd_params.requires_grad_(True)
        logits = self.vllm.get_llm_outpt(input_embeds, vt_range).logits
        edit_loss = label_loss(logits, label_ids, label_masks)
        torch.autograd.grad(edit_loss, self.autograd_params)
        self.autograd_params.zero_grad()
        for em in self.edit_modules:
            em.update_delta_weight() 
        self.autograd_params.requires_grad_(False)

    ############################################################################
    #                   MEND Special Function                                  #
    ############################################################################
    def get_edit_modules(self) -> Tuple[List[EditLinear], nn.ModuleDict, nn.ParameterList]:
        same_shape_modules = defaultdict(list)
        for module_name in self.cfg.edit_modules:
            module = find_module(self.vllm.model, module_name)
            if isinstance(module, transformers.pytorch_utils.Conv1D):
                in_dim, out_dim = module.weight.shape
            elif isinstance(module, nn.Linear):
                out_dim, in_dim = module.weight.shape
            else:
                raise 'Modified module should be liear.'
            shape = (in_dim, out_dim)
            same_shape_modules[shape].append([module, module_name])
        edit_modules, aux_models, autograd_params = [], nn.ModuleDict(), nn.ParameterList()
        for shape, modules in same_shape_modules.items():
            aux_model = GradientTransform(shape[0], shape[1], self.cfg.aux_model, len(modules))
            aux_models[str(shape)] = aux_model
            for idx, [module, module_name] in enumerate(modules):
                em = EditLinear(module, module_name, idx, aux_model, self.cfg, None)
                edit_modules.append(em)
                if hasattr(module, 'bias') and module.bias != None:
                    autograd_params.append(module.bias)
                else:
                    autograd_params.append(module.weight)
        return edit_modules, aux_models, autograd_params

    def clear_module_deltas(self, x:bool, delta:bool, edit_weight:bool):
        for em in self.edit_modules:
            em.clear_delta(x, delta, edit_weight)
    
    def update_weights_and_clear_deltas(self):
        for em in self.edit_modules:
            em.update_weight_and_clear_delta()

    def restore_to_original_weights(self):
        for em in self.edit_modules:
            em.restore_to_original_weight()

    
    def get_modules_for_training(self)->Dict[str, nn.Module]:
        training_modules = {'aux_models': self.aux_models, 'edit_lrs': self.edit_lrs}
        return training_modules
    
    def reinit_train_parameters(self):
        print('Not set reinit function.')

    def preprocess_train_data(self, vllm_edit_data:BaseVLLMEditData)->List:
        return vllm_edit_data.data 
        
    def organize_batch_data(self, a_batch_of_training_data:List):
        vllm = self.vllm_proc_data
        # edit/rel
        prompts = [d['requests'][0]['prompt'] for d in a_batch_of_training_data]
        imgs = [d['requests'][0]['image'] for d in a_batch_of_training_data]
        targets = [d['requests'][0]['target_new'] for d in a_batch_of_training_data]
        edit_xym = vllm.prompts_imgs_target_to_xym(prompts, imgs, targets)
        # gen
        gen_xym = {k:None for k in a_batch_of_training_data[0]['generality'].keys()}
        for k in gen_xym.keys():
            prompts = [d['generality'][k][0]['prompt'] for d in a_batch_of_training_data]
            imgs = [d['generality'][k][0]['image'] for d in a_batch_of_training_data]
            targets = [d['generality'][k][0]['target'] for d in a_batch_of_training_data]
            gen_xym[k] = vllm.prompts_imgs_target_to_xym(prompts, imgs, targets)
        # gen
        loc_xym = {k:None for k in a_batch_of_training_data[0]['locality'].keys()}
        for k in loc_xym.keys():
            prompts = [d['locality'][k][0]['prompt'] for d in a_batch_of_training_data]
            imgs = [d['locality'][k][0]['image'] for d in a_batch_of_training_data]
            targets = [d['locality'][k][0]['target'] for d in a_batch_of_training_data]
            loc_xym[k] = vllm.prompts_imgs_target_to_xym(prompts, imgs, targets)
        a_batch_of_training_data = move_to_device((edit_xym, gen_xym, loc_xym), self.device)
        return a_batch_of_training_data
        
    def get_a_new_optimizer(self)->torch.optim.Optimizer:
        opt = Adam([{'params': self.aux_models.parameters(), 'lr': self.cfg.aux_model.lr},
                {'params': self.edit_lrs.parameters(), 'lr': self.cfg.edit_lr_lr}])
        return opt
    
    def set_train(self, if_train = False):
        self.vllm.model.train(False)
        self.vllm.model.requires_grad_(False)
        self.autograd_params.requires_grad_(False)
        self.autograd_params.train(False)
        self.aux_models.requires_grad_(if_train)
        self.aux_models.train(if_train)
        self.edit_lrs.requires_grad_(if_train)
        self.edit_lrs.train(if_train)

    def other_train_init_final(self):
        pass
                    
    def train_a_batch(self, a_batch_of_training_data:Tuple):
        '''Assume:
        edit_xym: ((input_embeds, vt_range), label_ids, label_masks)
        gen_xym: {
            loss_name_1: ((input_embeds, vt_range), label_ids, label_masks),
            loss_name_2: ((input_embeds, vt_range), label_ids, label_masks), ...
        }
        loc_xm: {
            loss_name_1: ((input_embeds, vt_range), label_ids, label_masks)
            loss_name_2: ((input_embeds, vt_range), label_ids, label_masks), ...
        }
        ''' 
        edit_xym, gen_xym, loc_xym = a_batch_of_training_data
        self.clear_module_deltas(True, True, True)
        # prediction before edit for locality loss and edit
        with torch.no_grad():
            for loss_name, sp in loc_xym.items():
                (input_embeds, vt_range), label_ids, label_masks = sp
                logits = self.vllm.get_llm_outpt(input_embeds, vt_range).logits
                loc_xym[loss_name] = (input_embeds, vt_range), logits, label_masks
        (input_embeds, vt_range), label_ids, label_masks = edit_xym
        self.__edit_batch__(input_embeds, vt_range, label_ids, label_masks)
        loss = 0
        log_dict = {}
        # reliability losses
        logits = self.vllm.get_llm_outpt(input_embeds, vt_range).logits
        rel_loss = self.cfg.relia_lambda * label_loss(logits, label_ids, label_masks)
        log_dict['Reliability loss'] = float(rel_loss)
        loss += rel_loss
        # generality loss
        log_dict['Generality loss'] = {}
        for loss_name, sp in gen_xym.items():
            (input_embeds, vt_range), label_ids, label_masks = sp
            logits = self.vllm.get_llm_outpt(input_embeds, vt_range).logits
            gen_loss = self.cfg.gen_lambda * label_loss(logits, label_ids, label_masks)
            log_dict['Generality loss'][loss_name] = float(gen_loss)
            loss += gen_loss 
        # compute locality loss
        log_dict['Locality loss'] = {}
        for loss_name, sp in loc_xym.items():
            (input_embeds, vt_range), pre_logits, label_masks = sp
            logits = self.vllm.get_llm_outpt(input_embeds, vt_range).logits
            loc_loss = self.cfg.loc_lambda * logit_KL_loss(pre_logits, logits, label_masks)
            log_dict['Locality loss'][loss_name] = float(loc_loss)
            loss += loc_loss
        loss.backward()
        # try gradient clip
        log_dict['Grad-Norm'] = float(torch.nn.utils.clip_grad_norm_(
            self.aux_models.parameters(), 100., error_if_nonfinite=True))
        self.opt.step()
        self.opt.zero_grad()
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
