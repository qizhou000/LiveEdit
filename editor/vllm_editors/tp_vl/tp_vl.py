from transformers import AutoModelForCausalLM, AutoTokenizer
from ...vllms_for_edit.base import BaseVLLMForEdit
from transformers.pytorch_utils import Conv1D
from typing import Dict, List, Tuple, Union
from ..base import VLLMBaseEditor
from dataclasses import dataclass
from datasets import load_dataset
from ...base import BaseConfig
import torch, os, json, re
from copy import deepcopy
from torch import nn
import numpy as np

@dataclass
class TPvlConfig(BaseConfig):
    edit_model_name: str
    edit_layer: int
    num_steps: int
    lr: float
    loss_a_lambda: float
    loss_m_lambda: float
    weight_decay: float
    mlp_in_module_tmps: List[str]
    mlp_out_module_tmps: List[str]


class TPvl(VLLMBaseEditor):

    def __init__(self, vllm: BaseVLLMForEdit, config:TPvlConfig, device = 'cuda', verbose = False, 
            locality_data_path = 'data/wikitext/wikitext-103-raw-v1'):
        super().__init__(vllm, device)
        self.cfg = config
        self.verbose = verbose
        self.vllm.model.train(False)
        self.vllm.model.requires_grad_(False)
        self.edit_in_layers, self.edit_out_layers, self.edit_layer_in_dims, self.edit_layer_out_dims = \
            self.get_edit_layer(self.vllm.model, self.cfg.mlp_in_module_tmps, self.cfg.mlp_out_module_tmps, self.cfg.edit_layer)
        self.init_extra_weights()
        self.hooks = self.register_hooks()
        # load locality dataset
        self.locality_data = os.path.join(locality_data_path)
        self.locality_data = load_dataset(self.locality_data, split='train')
        self.locality_data =  np.array([t for t in self.locality_data['text'] if len(t.split(' ')) > 20 and not \
            re.search(r'^[\s\n]*=', t) and not re.search(r'=[\s\n]*$', t)]) 
        self.rng = np.random.default_rng(None)
        

    def get_edit_layer(self, model, mlp_in_module_tmps:List[str], mlp_out_module_tmps:List[str], layer_n:int):
        edit_in_layers = [find_module(model, mmt.format(layer_n).split('.')) for mmt in mlp_in_module_tmps]
        edit_out_layers = [find_module(model, mmt.format(layer_n).split('.')) for mmt in mlp_out_module_tmps]
        if isinstance(edit_in_layers[0], Conv1D):
            edit_layer_in_dims = [eil.weight.shape[0] for eil in edit_in_layers]
            edit_layer_out_dims = [eol.weight.shape[1] for eol in edit_out_layers]
        elif isinstance(edit_in_layers[0], nn.Linear):
            edit_layer_in_dims = [eil.weight.shape[1] for eil in edit_in_layers]
            edit_layer_out_dims = [eol.weight.shape[0] for eol in edit_out_layers]
        else:
            raise
        return edit_in_layers, edit_out_layers, edit_layer_in_dims, edit_layer_out_dims

    def init_extra_weights(self):
        for layer, in_dim in zip(self.edit_in_layers, self.edit_layer_in_dims):
            layer.extra_weights = torch.zeros([in_dim, 0], device=self.device) 
            layer.extra_biases = torch.zeros([0], device=self.device) 
            layer.new_extra_weights = None
            layer.new_extra_biases = None
        for layer, out_dim in zip(self.edit_out_layers, self.edit_layer_out_dims):
            layer.extra_weights = torch.zeros([0, out_dim], device=self.device) 
            layer.new_extra_weights = None

    def register_hooks(self):
        def in_layer_forward_edit_hook(module, args, output):
            # args: tuple(input) [batch_size, max_length, dim_in]
            # output.shape = [batch_size, max_length, dim_out]
            o = []
            if module.new_extra_weights != None:
                module.new_extra_output = args[0]@module.new_extra_weights + module.new_extra_biases
                o.append(module.new_extra_output)
            o.append(args[0]@module.extra_weights + module.extra_biases)
            o.append(output)
            output = torch.cat(o, -1)
            return output
        def out_layer_forward_pre_edit_hook(module, args):
            # args: tuple(input) [batch_size, max_length, dim_in]
            if module.new_extra_weights != None:
                len_new_extra_weights = len(module.new_extra_weights)
                module.tmp_new_extra_inp = args[0][..., :len_new_extra_weights]
            else:
                len_new_extra_weights = 0
            module.tmp_extra_inp = args[0][..., len_new_extra_weights:\
                        len_new_extra_weights+len(module.extra_weights)]
            return args[0][..., len_new_extra_weights+len(module.extra_weights):]
        def out_layer_forward_edit_hook(module, args, output):
            # args: tuple(input) [batch_size, max_length, dim_in]
            # output.shape = [batch_size, max_length, dim_out]
            extra_out = module.tmp_extra_inp@module.extra_weights 
            output = extra_out + output
            if module.new_extra_weights != None:
                output = output + module.tmp_new_extra_inp@module.new_extra_weights 
            return output
        hooks = []
        for l in self.edit_in_layers:
            edit_handle = l.register_forward_hook(in_layer_forward_edit_hook)
            hooks.append(edit_handle)
        for l in self.edit_out_layers:
            edit_handle = l.register_forward_pre_hook(out_layer_forward_pre_edit_hook)
            edit_handle = l.register_forward_hook(out_layer_forward_edit_hook)
            hooks.append(edit_handle)
        return hooks

    ######################################################################################
    ######################################################################################
    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'tp_vl', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return False

    def restore_to_original_model(self):
        self.init_extra_weights()

    def edit_batch(self, requests: List[Dict]):
        raise

    def edit_one_piece(self, request: Dict) -> None:
        """r = {'image': PILImage, 'prompt': str, 'target_new': str, ...} """ 
        # append new weights
        opt_params = []
        for l, d in zip(self.edit_in_layers, self.edit_layer_in_dims):
            l.new_extra_weights = torch.zeros([d, 1], device=self.device).requires_grad_(True)
            l.new_extra_biases = torch.zeros([1], device=self.device).requires_grad_(True)
            opt_params.append(l.new_extra_weights)
            opt_params.append(l.new_extra_biases)
        for l, d in zip(self.edit_out_layers, self.edit_layer_out_dims):
            l.new_extra_weights = torch.zeros([1, d], device=self.device).requires_grad_(True)
            opt_params.append(l.new_extra_weights)
        # train
        self.train_new_extra_weights(opt_params, request)
        # concate new extra weights
        for l, d in zip(self.edit_in_layers, self.edit_layer_in_dims):
            l.extra_weights = torch.cat([l.new_extra_weights, l.extra_weights], 1).detach().requires_grad_(False)
            l.extra_biases = torch.cat([l.new_extra_biases, l.extra_biases], 0).detach().requires_grad_(False)
            l.new_extra_weights = None
            l.new_extra_biases = None
        for l, d in zip(self.edit_out_layers, self.edit_layer_out_dims):
            l.extra_weights = torch.cat([l.new_extra_weights, l.extra_weights], 0).detach().requires_grad_(False)
            l.new_extra_weights = None


    ######################################################################################
    ############################# T-Patcher Utilization ##################################
    ######################################################################################

    def train_new_extra_weights(self, opt_params:List[torch.Tensor], request: List[Dict]):
        # prepare reliability data
        (input_embeds, vt_range), label_ids, label_masks = self.vllm.prompts_imgs_target_to_xym(
            [request['prompt']], [request['image']], [request['target_new']])
        # train
        opt = torch.optim.Adam(opt_params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        for i in range(self.cfg.num_steps):
            # loss_e
            logits = self.vllm.get_llm_outpt(input_embeds, vt_range).logits
            loss_e = self.vllm.label_loss(logits, label_ids, label_masks)
            # loss_a
            rep1s = []
            loss_a = 0
            for l in self.edit_in_layers:
                loss_a = loss_a + torch.exp(-l.new_extra_output).mean()
                rep1s.append(l.new_extra_output)
            # loss_m
            rand_i = self.rng.choice(len(self.locality_data), 1)[0]
            loc_str = self.locality_data[rand_i]
            input_embeds, vt_range = self.vllm.get_llm_input_embeds([loc_str], None)
            self.vllm.get_llm_outpt(input_embeds, vt_range)
            loss_m1 = 0
            for l in self.edit_in_layers:
                rep = l.new_extra_output * (l.new_extra_output > 0)
                loss_m1 = loss_m1 + torch.exp(rep).mean()
            loss_m2 = 0
            # for l, rep1 in zip(self.edit_in_layers, rep1s):
            #     rep = l.new_extra_output * (l.new_extra_output > 0)
            #     loss_m1 = loss_m1 + torch.exp(rep).mean()
            loss_m = loss_m1 + loss_m2
            # loss            
            loss = loss_e + loss_a * self.cfg.loss_a_lambda + loss_m * self.cfg.loss_m_lambda
            loss.backward() 
            if self.verbose:
                print(i, loss.item())
                print('  loss_e: ', loss_e.item())
                print('  loss_a: ', loss_a.item())
                print('  loss_m: ', loss_m.item())
                print()
            opt.step()
            opt.zero_grad()


def find_module(module:nn.Module, module_path:List[str]):
    for comp in module_path:
        if hasattr(module, comp):
            module = getattr(module, comp)
        elif comp.isdigit():
            module = module[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    return module
 
def label_loss(model, input_ids:torch.Tensor, label_ids:torch.Tensor, masks:torch.Tensor, average = True):
    # input_ids/label_ids/masks: [batch, max_len]
    logits = model(input_ids).logits
    log_pre_p = torch.log_softmax(logits, 2) # [batch, max_len, voc_size]
    log_pre_p = log_pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, max_len]
    loss = -(log_pre_p * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss

def logit_KL_loss(logits1:torch.Tensor, logits2:torch.Tensor, masks:torch.Tensor, average = True):
    # logits1/logits2: [batch, max_len, voc_size], masks: [batch, max_len]
    log_p1 = torch.log_softmax(logits1, 2)
    log_p2 = torch.log_softmax(logits2, 2)
    p1 = torch.softmax(logits1, 2)
    kl_loss = (p1 * (log_p1 - log_p2)).sum(2) # [batch, max_len]
    loss = (kl_loss * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss

