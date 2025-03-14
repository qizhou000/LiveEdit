from ...vllms_for_edit.base import BaseVLLMForEdit
from typing import Any, Dict, List, Tuple
from torch.optim.adam import Adam
from utils import find_module
from dataclasses import dataclass
from ..base import VLLMBaseEditor
from ...base import BaseConfig
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
import torch

@dataclass
class LEMoEvlConfig(BaseConfig):
    edit_model_name:str
    edit_layer_inpt_path: str
    edit_layer_outpt_path: str
    edit_layer_i_of_inpt: int
    llm_layer_tmp: str
    llm_hidden_dim1: int
    llm_hidden_dim2: int
    lora_rank: int
    lora_edit_batch_size: int
    max_steps: int
    min_loss: float
    lr: float
    topk: int


class LEMoEvl(VLLMBaseEditor):

    def __init__(self, vllm: BaseVLLMForEdit, config:LEMoEvlConfig, device='cuda:0',
                 verbose = False):
        super().__init__(vllm, device)
        self.cfg = config
        self.verbose = verbose
        self.is_editing = False
        self.vllm.model.train(False)
        self.vllm.model.requires_grad_(False)
        self.hook_for_edit_layer()
        self.wrap_get_llm_outpt()
        self.restore_to_original_model()

    def wrap_get_llm_outpt(self):
        def wrap(get_llm_outpt):
            def wrapped_get_llm_outpt(input_embeds, vt_range):
                if len(self.now_requests_to_be_edit) != 0 and not self.is_editing:
                    raise
                return get_llm_outpt(input_embeds, vt_range)
            return wrapped_get_llm_outpt
        if not hasattr(self.vllm, 'original_get_llm_outpt'):
            self.vllm.original_get_llm_outpt = self.vllm.get_llm_outpt
        self.vllm.get_llm_outpt = wrap(self.vllm.original_get_llm_outpt)
    
    def hook_for_edit_layer(self):
        def inpt_forward_hook(module, args, output):
            if len(self.lora_ks) == 0:
                return output
            reps = args[0] # [b, l, d]
            reps_shape_len = len(reps.shape)
            if reps_shape_len == 2:
                reps = reps.unsqueeze(0)
            # if self.is_editing:
            v = F.silu(torch.einsum('bD,nDd->bnd', reps.mean(1), self.kws_down))
            v = torch.einsum('bnd,ndD->bnD', v, self.kws_up)
            sim = torch.softmax(torch.einsum('bnd,nd->bn', v, self.lora_ks), 1) # [b,n]
            reps = torch.einsum('bld,ndr,nDr->blnD', reps, self.lora_cs1, self.lora_rs1)
            reps = torch.einsum('blnD,bn,nDr,ndr->bld', F.relu(reps), sim, self.lora_cs2, self.lora_rs2)
            self.new_reps = reps[0] if reps_shape_len == 2 else reps
        def outpt_forward_hook(module, args, output):
            if len(self.lora_ks) == 0:
                return output
            if hasattr(self, 'new_reps') and self.new_reps != None:
                output = output + self.new_reps
                self.new_reps = None
            return output
        edit_layer_inpt = find_module(self.vllm.model, self.cfg.edit_layer_inpt_path)
        edit_layer_inpt._forward_hooks.clear()
        edit_layer_inpt.register_forward_hook(inpt_forward_hook)
        edit_layer_outpt = find_module(self.vllm.model, self.cfg.edit_layer_outpt_path)
        if not edit_layer_inpt is edit_layer_inpt:
            edit_layer_outpt._forward_hooks.clear()
        edit_layer_outpt.register_forward_hook(outpt_forward_hook)

    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'lemoe_vl', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return True

    def restore_to_original_model(self):
        self.edited_requests = []
        self.now_requests_to_be_edit = []
        self.lora_cs1 = torch.zeros([0, self.cfg.llm_hidden_dim1, self.cfg.lora_rank], device = self.device)
        self.lora_rs1 = torch.zeros([0, self.cfg.llm_hidden_dim2, self.cfg.lora_rank], device = self.device)
        self.lora_cs2 = torch.zeros([0, self.cfg.llm_hidden_dim2, self.cfg.lora_rank], device = self.device)
        self.lora_rs2 = torch.zeros([0, self.cfg.llm_hidden_dim1, self.cfg.lora_rank], device = self.device)
        self.lora_ks = torch.zeros([0, self.cfg.llm_hidden_dim1], device = self.device)
        self.kws_down = torch.zeros([0, self.cfg.llm_hidden_dim1, self.cfg.llm_hidden_dim1//4], device = self.device)
        self.kws_up = torch.zeros([0, self.cfg.llm_hidden_dim1//4, self.cfg.llm_hidden_dim1], device = self.device)

    def edit_one_piece(self, request: Dict) -> None:
        """request = {'image': PILImage, 'prompt': str, 'target_new': str, ...} """
        self.now_requests_to_be_edit.append(request)
        if len(self.now_requests_to_be_edit) >= self.cfg.lora_edit_batch_size:
            self.edited_requests.extend(self.now_requests_to_be_edit)
            self.add_new_lora(self.now_requests_to_be_edit)
            self.now_requests_to_be_edit = []

    def edit_batch(self, requests: List[Dict]):
        raise

    def add_new_lora(self, requests:List[Dict]):
        self.is_editing = True
        new_c1 = nn.Parameter(torch.randn([1, self.cfg.llm_hidden_dim1, self.cfg.lora_rank], device = self.device)*0.01)
        new_r1 = nn.Parameter(torch.randn([1, self.cfg.llm_hidden_dim2, self.cfg.lora_rank], device = self.device)*0.01)
        new_c2 = nn.Parameter(torch.randn([1, self.cfg.llm_hidden_dim2, self.cfg.lora_rank], device = self.device)*0.01)
        new_r2 = nn.Parameter(torch.randn([1, self.cfg.llm_hidden_dim1, self.cfg.lora_rank], device = self.device)*0.01)
        new_k = nn.Parameter(torch.randn([1, self.cfg.llm_hidden_dim1], device = self.device)*0.01)
        new_kws_down = nn.Parameter(torch.randn([1, self.cfg.llm_hidden_dim1, self.cfg.llm_hidden_dim1//4], device = self.device)*0.01)
        new_kws_up = nn.Parameter(torch.randn([1, self.cfg.llm_hidden_dim1//4, self.cfg.llm_hidden_dim1], device = self.device)*0.01)
        opt = Adam([new_c1, new_r1, new_c2, new_r2, new_kws_down, new_kws_up], lr = self.cfg.lr)
        o_lora_cs1, o_lora_rs1, o_lora_cs2, o_lora_rs2 = self.lora_cs1, self.lora_rs1, self.lora_cs2, self.lora_rs2
        o_lora_ks, o_kws_down, o_kws_up = self.lora_ks, self.kws_down, self.kws_up
        prompts = [r['prompt'] for r in requests]
        imgs = [r['image'] for r in requests]
        targets = [r['target_new'] for r in requests]
        (input_embeds, vt_range), label_ids, label_masks = self.vllm.prompts_imgs_target_to_xym(prompts, imgs, targets)
        mid_layer_inpt = self.vllm.get_mid_module_inpt(input_embeds, vt_range, self.cfg.edit_layer_inpt_path)
        for i in range(self.cfg.max_steps):
            self.lora_cs1 = torch.cat([o_lora_cs1, new_c1])
            self.lora_rs1 = torch.cat([o_lora_rs1, new_r1])
            self.lora_cs2 = torch.cat([o_lora_cs2, new_c2])
            self.lora_rs2 = torch.cat([o_lora_rs2, new_r2])
            self.lora_ks = torch.cat([o_lora_ks, new_k])
            self.kws_down = torch.cat([o_kws_down, new_kws_down])
            self.kws_up = torch.cat([o_kws_up, new_kws_up])
            # logits = self.vllm.get_llm_outpt(input_embeds, vt_range).logits
            logits = self.vllm.forward_from_mid_layer(input_embeds, vt_range, 
                mid_layer_inpt, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i_of_inpt).logits
            loss = self.vllm.label_loss(logits, label_ids, label_masks)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if self.verbose:
                print('%s Loss: %s'%(i, float(loss)))
            if loss < self.cfg.min_loss:
                break
        self.lora_cs1.detach_().requires_grad_(False)
        self.lora_rs1.detach_().requires_grad_(False)
        self.lora_cs2.detach_().requires_grad_(False)
        self.lora_rs2.detach_().requires_grad_(False)
        self.lora_ks.detach_().requires_grad_(False)
        self.kws_down.detach_().requires_grad_(False)
        self.kws_up.detach_().requires_grad_(False)
        self.is_editing = False
