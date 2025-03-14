from ...vllms_for_edit.base import BaseVLLMForEdit
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from ..base import VLLMBaseEditor
from ...base import BaseConfig
from ... import nethook
from copy import deepcopy
import numpy as np
import torch

@dataclass
class FTvlConfig(BaseConfig):
    edit_model_name:str
    # Method
    rewrite_module_tmp: str
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    norm_constraint: float
    batch_size: int = 128


class FTvl(VLLMBaseEditor):

    def __init__(self, vllm: BaseVLLMForEdit, config:FTvlConfig, device='cuda:0',
                 verbose = False):
        super().__init__(vllm, device)
        self.cfg = config
        self.verbose = verbose
        self.original_w = {
            n: p.clone()
            for n, p in self.vllm.model.named_parameters()
            for layer in self.cfg.layers
            if self.cfg.rewrite_module_tmp.format(layer) in n
        } 
 
    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'ft_vl', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return True

    def restore_to_original_model(self):
        self.vllm.model.load_state_dict(self.original_w, strict = False)

    def edit_one_piece(self, request: Dict) -> None:
        """request = {'image': PILImage, 'prompt': str, 'target_new': str, ...} """
        self.edit_batch([request])

    def edit_batch(self, requests: List[Dict]):
        '''requests = [
            {'image': PILImage, 'prompt': str, 'target_new': str, ...},
            {'image': PILImage, 'prompt': str, 'target_new': str, ...}, ...] '''
        deltas = self.execute_ft(requests)
        with torch.no_grad():
            for w_name, upd_matrix in deltas.items():
                if self.verbose:
                    print('edit:',w_name)
                w = nethook.get_parameter(self.vllm.model, w_name) 
                w[...] += upd_matrix

    ######################################################################################
    ############################# Fine-Tune Utilization ##################################
    ######################################################################################
    def execute_ft(self, requests: List[Dict]) -> Dict[str, Tuple[torch.Tensor]]:
        """
        Executes the FT update algorithm for the specified update at the specified layer
        Invariant: model at beginning of function == model at end of function
        """
        # Update target and print info
        requests = deepcopy(requests)
        for request in requests:
            if request["target_new"][0] != " ":
                request["target_new"] = " " + request["target_new"]
            if self.verbose:
                print(
                    f"Executing FT algo for: "
                    f"[{request['prompt']}] -> [{request['target_new']}]"
                )
        
        # Retrieve weights that user desires to change
        weights = {
            n: p
            for n, p in self.vllm.model.named_parameters()
            for layer in self.cfg.layers
            if self.cfg.rewrite_module_tmp.format(layer) in n
        }
        
        # Save old weights for future restoration
        weights_copy = {k: v.detach().clone() for k, v in weights.items()}
        if self.verbose:
            print(f"Weights to be updated: {list(weights.keys())}")

        # Define inputs
        images = [r["image"] for r in requests]
        texts = [r["prompt"] for r in requests]
        targets = [r["target_new"] for r in requests]
        
        # Configure optimizer / gradients
        opt = torch.optim.AdamW( # ASGD
            [v for _, v in weights.items()],
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        for name, w in self.vllm.model.named_parameters():
            w.requires_grad = name in weights

        # Update loop: intervene at layers simultaneously
        loss_meter = AverageMeter()
        for it in range(self.cfg.num_steps):
            if self.verbose:
                print(20 * "=")
                print(f"Epoch: {it}")
                print(20 * "=")
            loss_meter.reset()

            for imgs, txt, tgt in zip(chunks(images, self.cfg.batch_size),
                chunks(texts, self.cfg.batch_size), chunks(targets, self.cfg.batch_size)
            ):
                (input_embeds, vt_range), label_ids, label_masks = self.vllm.prompts_imgs_target_to_xym(txt, imgs, tgt) 
                opt.zero_grad()  
                # compute loss
                logits = self.vllm.get_llm_outpt(input_embeds, vt_range).logits
                loss = label_loss(logits, label_ids, label_masks)

                if self.verbose:
                    print(f"Batch loss {loss.item()}")
                loss_meter.update(loss.item(), n=label_ids.shape[0])

                if loss.item() >= 1e-2:
                    loss.backward()
                    opt.step()

                if type(self.cfg.norm_constraint) is float:
                    eps = self.cfg.norm_constraint
                    with torch.no_grad():
                        for k, v in weights.items():
                            v[...] = torch.clamp(
                                v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                            )
            if self.verbose:
                print(f"Total loss {loss_meter.avg}")

            if loss_meter.avg < 1e-2:
                break

        deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

        # Restore state of original model
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = weights_copy[k]

        if self.verbose:
            print(f"Deltas successfully computed for {list(weights.keys())}")

        return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self): 
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def label_loss(logits, label_ids, masks, average = True):
    # logits: [batch_size, total_l, d], label_ids/masks: [batch_size, short_l]
    logits = logits[:, -label_ids.shape[1]:]
    log_pre_p = torch.log_softmax(logits, -1)
    log_pre_p = log_pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, short_l]
    loss = -(log_pre_p * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss