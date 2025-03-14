from ...vllms_for_edit.base import BaseVLLMForEdit
from ..base import VLLMBaseEditorWithTraining
from utils import find_module, move_to_device
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Tuple
from dataset.vllm import BaseVLLMEditData
from .modules import QVExtractor, LowRankGenerator
from dataclasses import dataclass
from ...base import BaseConfig
from torch.optim import Adam
import torch, os, yaml
from torch import nn
import numpy as np

@dataclass
class LiveEditConfig(BaseConfig):
    @dataclass
    class TrainConfig:
        lr:float
        lr_cut_it:List[int]
        lr_cut_rate:float
        rel_lambda: float
        gen_lambda: float
        loc_lambda: float
        soft_routing_lambda: float
        hard_routing_lambda: float
    @dataclass
    class RetrievalEditor:
        module_dim: int
        cross_att_head_n: int
        lora_rank: int
        lora_scale: float
        eqe_n: int # editing query representation
    edit_model_name: str
    retrieval_editor: RetrievalEditor
    train_cfg: TrainConfig
    llm_mid_dim: int
    llm_layer_tmp: str
    edit_layer_i: int

    @classmethod
    def from_yaml(self, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        data['train_cfg'] = self.TrainConfig(**data['train_cfg'])
        data['retrieval_editor'] = self.RetrievalEditor(**data['retrieval_editor'])
        return self(**data)
    @classmethod
    def from_json(self, fpath):
        raise

class LiveEdit(VLLMBaseEditorWithTraining):
    '''Lifelong Vision Language Model Editor'''

    def __init__(self, vllm: BaseVLLMForEdit, config:LiveEditConfig, device='cuda:0', 
            vllm_data_proc: BaseVLLMForEdit = None, data_proc_device = None, verbose = False):
        super().__init__(vllm, config, device)
        self.cfg = config
        self.verbose = verbose
        if vllm_data_proc != None:
            self.vllm_data_proc = vllm_data_proc
            self.vllm_data_proc.model.eval()
            self.vllm_data_proc.model.requires_grad_(False)
            self.data_proc_device = data_proc_device
            self.vllm_data_proc.set_device(data_proc_device)
        # initialize training parameters
        self.sim_scale = 1 / self.cfg.retrieval_editor.module_dim**0.5
        (eqe_n, llm_mid_dim, module_dim, cross_att_head_n, lora_rank, lora_scale) = (self.cfg.retrieval_editor.eqe_n, 
            self.cfg.llm_mid_dim, self.cfg.retrieval_editor.module_dim, self.cfg.retrieval_editor.cross_att_head_n, 
            self.cfg.retrieval_editor.lora_rank, self.cfg.retrieval_editor.lora_scale)
        self.edit_extractor = QVExtractor(eqe_n, llm_mid_dim, module_dim, cross_att_head_n, self.vllm.get_img_token_n(), False).to(self.device)
        self.inpt_extractor = QVExtractor(eqe_n, llm_mid_dim, module_dim, cross_att_head_n, self.vllm.get_img_token_n(), True).to(self.device)
        self.moegen_c = LowRankGenerator(llm_mid_dim, lora_rank, lora_scale, llm_mid_dim, module_dim, cross_att_head_n).to(self.device)
        self.moegen_r = LowRankGenerator(llm_mid_dim, lora_rank, lora_scale, llm_mid_dim, module_dim, cross_att_head_n).to(self.device)
        self.instant_reps_norm = nn.LayerNorm(self.cfg.llm_mid_dim).to(self.device)
        # for inference & other hyper-parameters
        self.edit_layer_path = self.cfg.llm_layer_tmp.format(self.cfg.edit_layer_i)
        self.wrap_and_hook()
        self.restore_to_original_model()
        self.set_train(False)
        self.is_editing = False

    def wrap_and_hook(self):
        '''Only for inference stage.'''
        self.now_infer_vt_range = None # [vt_begin, vt_end], The vision token range of current VLLM inference.
        self.now_infer_inpt_embd_shape = None # The shape of current input embedding 
        self.now_infer_query_pos_end = None # The range of query of current input embedding 
        # modify VLLM function `get_llm_outpt` for retrieval editor
        def get_llm_outpt_wrap(get_llm_outpt):
            def wrapped_get_llm_outpt(llm_inpt, vt_range):
                if not self.is_train and not self.is_editing and len(self.eqr_pool) > 0:
                    self.now_infer_vt_range = vt_range
                    self.now_infer_inpt_embd_shape = llm_inpt['inputs_embeds'].shape
                    if 'query_range' in llm_inpt:
                        self.now_infer_query_pos_end = llm_inpt['query_range'][1]
                    elif self.verbose:
                        print("Without query range, will use all reps except for visual reps as query reps.")
                return get_llm_outpt(llm_inpt, vt_range)
            return wrapped_get_llm_outpt 
        if not hasattr(self, 'original_get_llm_outpt'):
            self.original_get_llm_outpt = self.vllm.get_llm_outpt
        self.vllm.get_llm_outpt = get_llm_outpt_wrap(self.original_get_llm_outpt)
        # initialize for retrieval editor hooked into the VLLM
        def apply_edit_residual(outpt, edit_residual):
            if isinstance(outpt, (tuple, list)):
                outpt = list(outpt) 
                outpt[0] = outpt[0] + edit_residual
            else:
                outpt = outpt + edit_residual
            return outpt
        def edit_with_moes(module, args, output):
            if self.is_train:
                output = apply_edit_residual(output, self.train_edit_residual)
                self.train_edit_residual = None
            elif (not self.is_editing and len(self.eqr_pool) > 0 and self.now_infer_vt_range != None):
                # inpt_reps = output[0] if isinstance(output, tuple) else output
                reps = output[0] if isinstance(output, (tuple, list)) else output
                assert reps.shape == self.now_infer_inpt_embd_shape 
                vision_reps = reps[:, self.now_infer_vt_range[0]:self.now_infer_vt_range[1]]
                query_reps = reps[:, self.now_infer_vt_range[1]:self.now_infer_query_pos_end] 
                self.now_infer_inpt_embd_shape = self.now_infer_vt_range = self.now_infer_query_pos_end = None
                moe_cs, moe_rs, fuse_coe = self.retrieve_moes(vision_reps, query_reps)
                edit_residual = self.get_edit_residual(reps, moe_cs, moe_rs, fuse_coe)
                output = apply_edit_residual(output, edit_residual)
            return output
        edit_layer = find_module(self.vllm.model, self.edit_layer_path)
        if hasattr(edit_layer, 'edit_layer_hooked'): 
            edit_layer._forward_hooks.clear()
        edit_layer.edit_layer_hooked = True
        edit_layer.register_forward_hook(edit_with_moes)

    def retrieve_moes(self, vision_reps:torch.Tensor, query_reps:torch.Tensor, 
                      return_retr_details = False):
        '''Only for inference stage.'''
        # inpt_vision_reps: [b, l, hidden_size]
        assert len(vision_reps) == len(query_reps) == 1
        # Get lora weights
        ivr = self.inpt_extractor.extract_vision(query_reps, vision_reps) # [1, eqe_n, module_dim]
        vis_sim = torch.einsum('bed,med->bme', ivr, self.evr_pool).mean(2) * self.sim_scale # [1, m]
        ivr_prot = self.inpt_extractor.extract_from_visprot(query_reps) # [1, eqe_n, module_dim]
        vis_sim_prot = torch.einsum('bed,bed->be', ivr, ivr_prot).mean(1, True) * self.sim_scale # [1, 1]
        retrieval_map = (vis_sim > vis_sim_prot).to(torch.bool)[0] # [1, m] -> [m]
        moe_cs = self.moe_cs_pool[retrieval_map]
        moe_rs = self.moe_rs_pool[retrieval_map]
        # Get fuse coefficient
        iqr = self.inpt_extractor.extract_query(query_reps) # [1, eqe_n, module_dim]
        eqr = self.eqr_pool[retrieval_map] # [m, eqe_n, module_dim]
        fuse_coe = self.get_moe_fuse_coe(iqr, eqr)
        if not return_retr_details:
            return moe_cs, moe_rs, fuse_coe
        else:
            retrieved_requests = [r for r, m in zip(self.requests_pool, retrieval_map) if m]
            return vis_sim, vis_sim_prot, retrieved_requests, moe_cs, moe_rs, fuse_coe
    
    ############################################################################
    ######################### LiveEdit Basic Functions ############################
    ############################################################################
    def get_reps_for_edit(self, vllm:BaseVLLMForEdit, request: Dict):
        '''request: {'prompt': str, 'image': PILImage, 'target': str}'''
        (input_embeds, vt_range), label_ids, label_masks = vllm.prompts_imgs_target_to_xym(
            [request['prompt']], [request['image']], [request['target']])
        edit_signal = vllm.get_mid_module_outpt(input_embeds, vt_range, self.edit_layer_path) 
        pre_vision_reps = edit_signal[:, :vt_range[0]] # [1, l1, d]
        vision_reps = edit_signal[:, vt_range[0]:vt_range[1]] # [1, l1, d]
        query_reps = edit_signal[:, vt_range[1]:1 - label_masks.shape[1]] # [1, l2, d]
        ans_reps = edit_signal[:, 1 - label_masks.shape[1]:] # [1, l3, d]
        return pre_vision_reps, vision_reps, query_reps, ans_reps # Edit signals

    def get_new_edit(self, vision_reps:torch.Tensor, query_reps:torch.Tensor, ans_reps:torch.Tensor)->Tuple[torch.Tensor]:
        # vision_reps: [1, l1, d]; query_reps: [1, l2, d]; ans_reps: [1, l3, d]
        assert len(vision_reps) == len(query_reps) == len(ans_reps) == 1
        evr = self.edit_extractor.extract_vision(query_reps, vision_reps)
        eqr = self.edit_extractor.extract_query(query_reps)
        edit_reps = torch.cat([vision_reps, query_reps, ans_reps], 1)
        moe_c = self.moegen_c(edit_reps)
        #########################
        moe_r = self.moegen_r(edit_reps)
        return eqr, evr, moe_c, moe_r
    
    def get_edit_residual(self, inpt_reps:torch.Tensor, moe_cs:torch.Tensor, 
                          moe_rs:torch.Tensor, fuse_coe:torch.Tensor)->torch.Tensor:
        '''inpt_reps: [1, l, hidden_size]; moe_in_cs: [m, lora_rank, hidden_size (ffn_in)]; 
            moe_in_rs: [m, lora_rank, ffn_out]; moe_out_cs: [m, lora_rank, ffn_out]; 
            moe_out_rs: [m, lora_rank, ffn_in]; fuse_coe: [1, m]'''
        assert len(inpt_reps) == len(fuse_coe) == 1
        x = self.instant_reps_norm(inpt_reps)[0] # [1, l, hidden_size] -> [l, hidden_size]
        # x = torch.einsum('ld,mrd,mrD,m->lD', x, moe_cs, moe_rs, fuse_coe[0])
        x = torch.relu(torch.einsum('ld,mrd->lmr', x, moe_cs))
        x = torch.einsum('lmr,mrd,m->ld', x, moe_rs, fuse_coe[0])
        return x.unsqueeze(0) # [1, l, hidden_size]
    
    def get_moe_fuse_coe(self, iqrs:torch.Tensor, eqrs:torch.Tensor, split = False):
        sim = torch.einsum('ned,med->nme', iqrs, eqrs).mean(2) * self.sim_scale # [n, m]
        rela_sim = torch.softmax(sim, 1) # [n, m]
        abs_sim = torch.sigmoid(sim) # [n, m] rescale ???
        if split: 
            return rela_sim, abs_sim
        return rela_sim * abs_sim # [n, m]

    ############################################################################
    ########################### Editor Functions ###############################
    ############################################################################
    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'liveedit', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return False

    def restore_to_original_model(self):
        self.requests_pool = []
        self.eqr_pool = torch.zeros([0, self.cfg.retrieval_editor.eqe_n, self.cfg.retrieval_editor.module_dim], device=self.device) # editing query representation pool
        self.evr_pool = torch.zeros([0, self.cfg.retrieval_editor.eqe_n, self.cfg.retrieval_editor.module_dim], device=self.device) # editing-relevant vision representation pool
        self.moe_cs_pool = torch.zeros([0, self.cfg.retrieval_editor.lora_rank, self.cfg.llm_mid_dim], device=self.device) # lora 1 column weights pool
        self.moe_rs_pool = torch.zeros([0, self.cfg.retrieval_editor.lora_rank, self.cfg.llm_mid_dim], device=self.device) # lora 1 row weights pool

    def edit_one_piece(self, request: Dict) -> None:
        """request = {'image': PILImage, 'prompt': str, 'target_new': str, ...} """
        self.is_editing = True
        self.requests_pool.append(request)
        r = {'prompt': request['prompt'], 'image': request['image'], 'target': request['target_new']}
        pre_vision_reps, vision_reps, query_reps, ans_reps = self.get_reps_for_edit(self.vllm, r)
        eqr, evr, moe_c, moe_r =  self.get_new_edit(vision_reps, query_reps, ans_reps)
        self.eqr_pool = torch.cat([self.eqr_pool, eqr], 0)
        self.evr_pool = torch.cat([self.evr_pool, evr], 0)
        self.moe_cs_pool = torch.cat([self.moe_cs_pool, moe_c], 0)
        self.moe_rs_pool = torch.cat([self.moe_rs_pool, moe_r], 0)
        self.is_editing = False 

    def edit_batch(self, requests: List[Dict]):
        raise
    
    ############################################################################
    ####################### Training Functions #################################
    ############################################################################
    def get_modules_for_training(self)->Dict[str, nn.Module]:
        return {'edit_extractor': self.edit_extractor, 'inpt_extractor': self.inpt_extractor, 
            'moegen_c': self.moegen_c, 'moegen_r': self.moegen_r, 
            'instant_reps_norm': self.instant_reps_norm}
    
    def reinit_train_parameters(self):
        self.edit_extractor.reset_parameters()
        self.inpt_extractor.reset_parameters()
        self.moegen_c.reset_parameters()
        self.moegen_r.reset_parameters()
        self.instant_reps_norm.reset_parameters()

    def preprocess_train_data(self, vllm_edit_data:BaseVLLMEditData)->List:
        return vllm_edit_data.data_with_img
        
        
    def organize_batch_data(self, a_batch_raw_data:List):
        vllm = self.vllm_data_proc # use the VLLM used to preprocess data
        batch_size = len(a_batch_raw_data)
        batch_edit_signal = []
        rel_edit_i = [] 
        batch_rel_data = [] # for each: {'prompts': [str], 'imgs': [PILImage], 'targets': [str]}
        batch_gen_data = {gen_name: [] for gen_name in a_batch_raw_data[0]['generality']}
        batch_loc_data = {loc_name: [] for loc_name in a_batch_raw_data[0]['locality']
                if a_batch_raw_data[0]['locality'][loc_name][0]['image'] != None}
        for d in a_batch_raw_data:
            # Edit signal
            batch_edit_signal.append([self.get_reps_for_edit(vllm, {'prompt': r['prompt'],
                'image': r['image'], 'target': r['target_new']}) for r in d['requests']])
            # Reliability 
            i = self.rng_data_proc.integers(0, len(d['requests']))
            rel_edit_i.append(i)
            r = d['requests'][i]
            batch_rel_data.append({'prompts': [r['prompt']], 'imgs': [r['image']], 'targets': [r['target_new']]})
            # Generality
            for gen_name in batch_gen_data.keys():
                i = self.rng_data_proc.integers(0, len(d['generality'][gen_name]))
                g = d['generality'][gen_name][i]
                batch_gen_data[gen_name].append({'prompts': [g['prompt']], 'imgs': [g['image']], 'targets': [g['target']]})
            # Locality
            for loc_name in batch_loc_data.keys():
                i = self.rng_data_proc.integers(0, len(d['locality'][loc_name]))
                l = d['locality'][loc_name][i]
                batch_loc_data[loc_name].append({'prompts': [l['prompt']], 'imgs': [l['image']], 'targets': [l['target']]})
        # Fuse moe masks
        edit_ns = torch.tensor([len(bes) for bes in batch_edit_signal], device=self.device)
        cols = torch.sum(edit_ns)
        start_indices = torch.cumsum(torch.cat([torch.tensor([0], device=self.device), edit_ns[:-1]]), dim=0).unsqueeze(1)
        rel_edit_i = start_indices[:, 0] + torch.tensor(rel_edit_i, device=self.device)
        rel_moe_mask = torch.zeros((batch_size, cols), dtype=torch.int, device=self.device)
        rel_moe_mask[torch.arange(batch_size), rel_edit_i] = 1
        idx = torch.arange(cols, device = self.device).expand(batch_size, cols)
        gen_moe_mask = ((idx >= start_indices) & (idx < (start_indices + edit_ns.unsqueeze(1)))).int()
        loc_moe_mask = torch.zeros_like(gen_moe_mask) 
        for i in range(batch_size):
            ns = self.rng_train.integers(0, int(cols) + 1, 3)
            rel_moe_mask[i, :ns[0]], gen_moe_mask[i, :ns[1]], loc_moe_mask[i, :ns[2]] = 1, 1, 1
        rel_moe_mask, gen_moe_mask, loc_moe_mask = rel_moe_mask.bool(), gen_moe_mask.bool(), loc_moe_mask.bool()
        # Reliability edit batch data
        rel_xyms = [vllm.prompts_imgs_target_to_xym(**r) for r in batch_rel_data] # for each: (input_embeds, vt_range), label_ids, label_masks
        rel_mid_reps = [vllm.get_mid_module_outpt(input_embeds, vt_range, self.edit_layer_path) 
                        for (input_embeds, vt_range), label_ids, label_masks in rel_xyms] 
        rel_edit_reps = [self.get_reps_for_edit(vllm, {'prompt': r['prompts'][0], 
            'image': r['imgs'][0], 'target': r['targets'][0]}) for r in batch_rel_data]
        packed_rel_data = (rel_xyms, rel_mid_reps, rel_edit_reps)
        # Generality edit batch data
        packed_gen_data = {}
        for k, v in batch_gen_data.items():
            gen_xyms = [vllm.prompts_imgs_target_to_xym(**g) for g in v]
            gen_mid_reps = [vllm.get_mid_module_outpt(input_embeds, vt_range, self.edit_layer_path) 
                for (input_embeds, vt_range), label_ids, label_masks in gen_xyms] 
            gen_edit_reps = [self.get_reps_for_edit(vllm, {'prompt': g['prompts'][0], 
                'image': g['imgs'][0], 'target': g['targets'][0]}) for g in v]
            packed_gen_data[k] = (gen_xyms, gen_mid_reps, gen_edit_reps)
        # Locality edit batch data
        packed_loc_data = {}
        for k, v in batch_loc_data.items():
            loc_xyms = [vllm.prompts_imgs_target_to_xym(**l) for l in v]
            loc_mid_reps = [vllm.get_mid_module_outpt(input_embeds, vt_range, self.edit_layer_path) 
                for (input_embeds, vt_range), label_ids, label_masks in loc_xyms] 
            pre_logits = [vllm.forward_from_mid_layer(input_embeds, vt_range, lmr, 
                self.cfg.llm_layer_tmp, self.cfg.edit_layer_i).logits for 
                lmr, ((input_embeds, vt_range), label_ids, label_masks) in zip(loc_mid_reps, loc_xyms)]
            loc_edit_reps = [self.get_reps_for_edit(vllm, {'prompt': l['prompts'][0], 
                'image': l['imgs'][0], 'target': l['targets'][0]}) for l in v] 
            packed_loc_data[k] = (loc_xyms, loc_mid_reps, pre_logits, loc_edit_reps)
        # Retrieval contrastive learning data for neighbor loss
        def get_rand_gn_ln():
            gn = list(batch_gen_data.keys())[self.rng_data_proc.integers(0, len(batch_gen_data.keys()))]
            ln = list(batch_loc_data.keys())[self.rng_data_proc.integers(0, len(batch_loc_data.keys()))]
            return gn, ln
        batch_retr_neib_data = [[], []]
        for j in range(batch_size):
            # neighbor 1
            n = self.rng_data_proc.integers(0, 3) # 0: rel, 1: gen, 2: loc. Where rel and gen are in the same neighbor.
            gn, ln = get_rand_gn_ln()
            d = [packed_rel_data[-1], packed_gen_data[gn][-1], packed_loc_data[ln][-1]][n][j][1:3]
            batch_retr_neib_data[0].append(d)
            # neighbor 2 
            n = self.rng_data_proc.integers(0, 2) if n != 2 else n # reselect in the same neighbor
            gn, ln = get_rand_gn_ln()
            d = [packed_rel_data[-1], packed_gen_data[gn][-1], packed_loc_data[ln][-1]][n][j][1:3]
            batch_retr_neib_data[1].append(d)
        # Retrieval contrastive learning data for prototype loss 
        batch_retr_prot_data = [[], []]
        for j in range(batch_size):
            n = self.rng_data_proc.integers(0, 2) # 0: r/g, 1: l
            # data_1
            gn, ln = get_rand_gn_ln()
            d = [[packed_rel_data[-1], packed_gen_data[gn][-1]][self.rng_data_proc.integers(0, 2)], packed_loc_data[ln][-1]][n][j][1:3]
            batch_retr_prot_data[0].append(d)
            # data_2
            gn, ln = get_rand_gn_ln()
            d = [[packed_rel_data[-1], packed_gen_data[gn][-1]][self.rng_data_proc.integers(0, 2)], packed_loc_data[ln][-1]][1-n][j][1:3]
            batch_retr_prot_data[1].append(d)
        a_batch_organized_data = (batch_size, batch_edit_signal, rel_moe_mask, gen_moe_mask, loc_moe_mask, 
            packed_rel_data, packed_gen_data, packed_loc_data, batch_retr_neib_data, batch_retr_prot_data)
        return move_to_device(a_batch_organized_data, self.device)
        
    def train_a_batch(self, a_batch_organized_data):
        eps = 1e-8
        vllm = self.vllm
        (batch_size, batch_edit_signal, rel_moe_mask, gen_moe_mask, loc_moe_mask, 
            packed_rel_data, packed_gen_data, packed_loc_data, batch_retr_neib_data, 
            batch_retr_prot_data) = a_batch_organized_data
        # initialize losses
        loss = 0
        log_dict = {}
        # Prepare edit data eqr, evr, moe_c, moe_r
        new_edit = [self.get_new_edit(vision_reps, query_reps, ans_reps) for bes in batch_edit_signal 
                    for pre_vision_reps, vision_reps, query_reps, ans_reps in bes]
        eqrs = torch.cat([ne[0] for ne in new_edit], 0) # [batch_request_n, eqe_n, d]
        # evrs = torch.cat([ne[1] for ne in new_edit], 0) # [batch_request_n, eqe_n, d]
        moe_cs = torch.cat([ne[2] for ne in new_edit], 0) # [batch_request_n, rank_n, d1]
        moe_rs = torch.cat([ne[3] for ne in new_edit], 0) # [batch_request_n, rank_n, d2]
        # Reliability edit loss
        rel_loss = 0
        rel_xyms, rel_mid_reps, rel_edit_reps = packed_rel_data
        for xyms, mr, er, mm in zip(rel_xyms, rel_mid_reps, rel_edit_reps, rel_moe_mask):
            (input_embeds, vt_range), label_ids, label_masks = xyms
            pre_vision_reps, vision_reps, query_reps, ans_reps = er
            iqr = self.inpt_extractor.extract_query(query_reps) # [1, eqe_n, module_dim]
            fuse_coe = self.get_moe_fuse_coe(iqr, eqrs[mm]) # [1, selected_request_n]
            self.train_edit_residual = self.get_edit_residual(torch.cat(er, 1), 
                moe_cs[mm], moe_rs[mm], fuse_coe)
            logits = vllm.forward_from_mid_layer(input_embeds, vt_range, 
                mr, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i).logits
            rel_loss += vllm.label_loss(logits, label_ids, label_masks, True)
        rel_loss /= batch_size
        log_dict['Reliability loss'] = float(rel_loss)
        loss += rel_loss * self.cfg.train_cfg.rel_lambda
        # Generality edit losses
        gen_loss = 0
        for gen_name, (gen_xyms, gen_mid_reps, gen_edit_reps) in packed_gen_data.items():
            gen_name_loss = 0
            for xyms, mr, er, mm in zip(gen_xyms, gen_mid_reps, gen_edit_reps, gen_moe_mask):
                (input_embeds, vt_range), label_ids, label_masks = xyms
                pre_vision_reps, vision_reps, query_reps, ans_reps = er
                iqr = self.inpt_extractor.extract_query(query_reps) # [1, eqe_n, module_dim]
                fuse_coe = self.get_moe_fuse_coe(iqr, eqrs[mm]) # [1, selected_request_n]
                self.train_edit_residual = self.get_edit_residual(torch.cat(er, 1), 
                    moe_cs[mm], moe_rs[mm], fuse_coe)
                logits = vllm.forward_from_mid_layer(input_embeds, vt_range, 
                    mr, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i).logits
                gen_name_loss += vllm.label_loss(logits, label_ids, label_masks, True)
            gen_name_loss /= batch_size
            log_dict['Generality loss %s'%gen_name] = float(gen_name_loss)
            gen_loss += gen_name_loss
        log_dict['Generality loss'] = float(gen_loss)
        loss += gen_loss * self.cfg.train_cfg.gen_lambda
        # Locality edit losses
        loc_loss = 0
        for loc_name, (loc_xyms, loc_mid_reps, pre_logits, loc_edit_reps) in packed_loc_data.items():
            loc_name_loss = 0
            for xyms, mr, pl, er, mm in zip(loc_xyms, loc_mid_reps, pre_logits, loc_edit_reps, loc_moe_mask):
                (input_embeds, vt_range), label_ids, label_masks = xyms
                pre_vision_reps, vision_reps, query_reps, ans_reps = er
                iqr = self.inpt_extractor.extract_query(query_reps) # [1, eqe_n, module_dim]
                fuse_coe = self.get_moe_fuse_coe(iqr, eqrs[mm]) # [1, selected_request_n]
                self.train_edit_residual = self.get_edit_residual(torch.cat(er, 1), 
                    moe_cs[mm], moe_rs[mm], fuse_coe)
                logits = vllm.forward_from_mid_layer(input_embeds, vt_range, 
                    mr, self.cfg.llm_layer_tmp, self.cfg.edit_layer_i).logits
                loc_name_loss += vllm.logit_KL_loss(logits, pl, label_masks, True)
            loc_name_loss /= batch_size
            log_dict['Locality loss %s'%loc_name] = float(loc_name_loss)
            loc_loss += loc_name_loss
        log_dict['Locality loss'] = float(loc_loss)
        loss += loc_loss * self.cfg.train_cfg.loc_lambda
        # Moe soft routing loss
        iqrs = torch.cat([self.inpt_extractor.extract_query(d[1]) for d in batch_retr_neib_data[0]], 0) # [b, eqe_n, module_dim]
        eqrs = torch.cat([self.edit_extractor.extract_query(d[1]) for d in batch_retr_neib_data[1]], 0) # [b, eqe_n, module_dim]
        rela_sim, abs_sim = self.get_moe_fuse_coe(iqrs, eqrs, True) # [b, b]
        moe_soft_routing_loss_rela = - torch.log(torch.diag(rela_sim) + eps).mean(0)
        abs_pos_sim = torch.diag(abs_sim)
        abs_neg_sim = torch.diag(abs_sim.roll(1, 1))
        moe_soft_routing_loss_abs = - (torch.log(abs_pos_sim + eps) + torch.log(1 - abs_neg_sim + eps)).mean()
        moe_soft_routing_loss = moe_soft_routing_loss_rela + moe_soft_routing_loss_abs
        log_dict['MoE soft routing relative loss'] = float(moe_soft_routing_loss_rela)
        log_dict['MoE soft routing abstract loss'] = float(moe_soft_routing_loss_abs)
        log_dict['MoE soft routing mean abstract positive similarity'] = float(abs_pos_sim.mean(0)) 
        log_dict['MoE soft routing mean abstract negative similarity'] = float(abs_neg_sim.mean(0))
        log_dict['MoE soft routing loss'] = float(moe_soft_routing_loss)
        loss += moe_soft_routing_loss * self.cfg.train_cfg.soft_routing_lambda
        # Hard Routing losses
        def hard_routing(inpt_reps, edit_reps):
            ivrs = torch.cat([self.inpt_extractor.extract_vision(d[1], d[0]) for d in inpt_reps], 0) # [b, eqe_n, module_dim]
            evrs = torch.cat([self.edit_extractor.extract_vision(d[1], d[0]) for d in edit_reps], 0) # [b, eqe_n, module_dim]
            sim = torch.einsum('bed,med->bme', ivrs, evrs).mean(2) * self.sim_scale # [b, b]
            ivrs_prot = torch.cat([self.inpt_extractor.extract_from_visprot(d[1]) for d in inpt_reps]) # [b, eqe_n, module_dim]
            sim_prot = torch.einsum('bed,bed->be', ivrs, ivrs_prot).mean(1, True) * self.sim_scale # [b,1]
            sim = torch.softmax(torch.cat([sim, sim_prot], 1), 1) # [b, b+1]
            return sim # [b, b+1]
        # Hard routing neighbor loss
        sim = hard_routing(*batch_retr_neib_data) # [b, b+1]
        loss_retr_neb = - torch.log(torch.diag(sim) + eps).mean(0)
        log_dict['MoE hard routing neighbor loss'] = float(loss_retr_neb)
        # Hard routing prototype loss 
        sim = hard_routing(*batch_retr_prot_data) # [b, b+1]
        loss_retr_prot = - torch.log(sim[:, -1] + eps).mean(0)
        log_dict['MoE hard routing prototype loss'] = float(loss_retr_prot)
        hard_routing_loss = loss_retr_neb + loss_retr_prot 
        log_dict['MoE hard routing loss'] = float(hard_routing_loss)
        loss += hard_routing_loss * self.cfg.train_cfg.hard_routing_lambda
        # update
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.lr_scheduler.step()
        log_dict['Learning rate'] = float(self.lr_scheduler.get_last_lr()[0])
        return float(loss), log_dict
    
    def get_a_new_optimizer(self)->torch.optim.Optimizer:
        from torch.optim.lr_scheduler import LambdaLR
        opt = Adam([*self.edit_extractor.parameters(), *self.inpt_extractor.parameters(),
                    *self.instant_reps_norm.parameters(), *self.moegen_c.parameters(), 
                    *self.moegen_r.parameters()], self.cfg.train_cfg.lr)
        lr_cut_it = torch.tensor(self.cfg.train_cfg.lr_cut_it, device=self.device)
        def lr_lambda(step):
            return self.cfg.train_cfg.lr_cut_rate**(step > lr_cut_it).sum()
        lr_scheduler = LambdaLR(opt, lr_lambda)
        return opt, lr_scheduler

    def set_train(self, is_train:bool):
        self.is_train = is_train
        self.vllm.model.requires_grad_(False)
        self.vllm.model.train(False)
        if hasattr(self, 'vllm_data_proc') and self.vllm_data_proc != None:
            self.vllm_data_proc.model.requires_grad_(False)
            self.vllm_data_proc.model.train(False)
        self.edit_extractor.train(is_train)
        self.edit_extractor.requires_grad_(is_train)
        self.inpt_extractor.train(is_train)
        self.inpt_extractor.requires_grad_(is_train)
        self.instant_reps_norm.train(is_train)
        self.instant_reps_norm.requires_grad_(is_train)
        self.moegen_c.train(is_train)
        self.moegen_c.requires_grad_(is_train)
        self.moegen_r.train(is_train)
        self.moegen_r.requires_grad_(is_train)

    def other_train_init_begin(self):
        self.rng_data_proc = np.random.default_rng(self.random_seed)
        self.rng_train = np.random.default_rng(self.random_seed + 1)
