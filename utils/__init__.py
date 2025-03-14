from editor.vllm_editors.base import VLLMBaseEditorWithTraining
from editor.vllms_for_edit.base import BaseVLLMForEdit
from utils.GLOBAL import model_path_map, ROOT_PATH
from typing import Union, List, Dict, Optional
from transformers import AutoModelForCausalLM
from torch import nn
import numpy as np
import torch, os


def print_list_structure(data, indent=0):
    indentation = '  ' * indent  
    if isinstance(data, list):
        print(f"List:")
        for index, item in enumerate(data):
            print(f"{indentation}  [{index}]:", end=' ')
            print_list_structure(item, indent + 1)  
    elif isinstance(data, tuple):
        print(f"Tuple:")
        for index, item in enumerate(data):
            print(f"{indentation}  ({index}):", end=' ')
            print_list_structure(item, indent + 1)  
    elif isinstance(data, torch.Tensor):
        print(f"Tensor: {data.shape}")
    else:
        print(f"{data}")


def find_module(module, module_path:str)->Union[torch.Tensor, nn.Module]:
    for comp in module_path.split('.'):
        if hasattr(module, comp):
            module = getattr(module, comp)
        elif comp.isdigit():
            module = module[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    return module
 
def move_to_device(data, device):
    '''Move list and dictionary nested PyTorch tensors to a specific device.'''
    if isinstance(data, (torch.Tensor, nn.Module)):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple([move_to_device(item, device) for item in data])
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (int, float, str, bool, type(None), np.integer, np.floating)):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def get_full_model_name(model_name_part:str)->str:
    model_name_part = model_name_part.lower()
    if 'blip2' in model_name_part:
        return 'blip2-opt-2.7b'
    elif 'llava' in model_name_part:
        return 'llava-v1.5-7b'
    elif 'mini' in model_name_part:
        if '4' in model_name_part and 'gpt' in model_name_part:
            return 'minigpt-4-vicuna-7b'
        else:
            raise
    elif 'bert' in model_name_part:
        if 'base' in model_name_part:
            if 'uncased' in model_name_part:
                return 'bert-base-uncased'
            elif 'cased' in model_name_part:
                return 'bert-base-cased'
            else:
                raise
        else:
            raise
    elif 'roberta' in model_name_part:
        return 'roberta-base'
    elif 'opt' in model_name_part:
        if '125m' in model_name_part:
            return 'opt-125m'
        else:
            raise
    elif 'gpt' in model_name_part:
        if 'j' in model_name_part:
            return 'gpt-j-6b'
        elif '2' in model_name_part:
            return 'gpt2-xl'
        else:
            raise
    elif 'llama' in model_name_part:
        if '7b' in model_name_part:
            if 'chat' in model_name_part:
                return 'llama-2-7b-chat'
            else:
                return 'llama-2-7b'
        elif '160m' in model_name_part:
            return 'llama-160m'
        else:
            raise
    raise

def get_editor_config_path(editor_name:str, edit_model_name:str):
    path = os.path.join(ROOT_PATH, 'configs', editor_name.lower(), '%s.yaml'%get_full_model_name(edit_model_name))
    return path

def get_model_path(model_name:str)->str:
    return model_path_map[get_full_model_name(model_name)]

################################################################################
################################ For VLLM ######################################
################################################################################
def load_vllm_for_edit(model_name:str, device:str)->BaseVLLMForEdit:
    model_name = get_full_model_name(model_name)
    model_path = get_model_path(model_name)
    print('Loading %s from "%s".'%(model_name, model_path))
    if 'llava' in model_name:
        from editor.vllms_for_edit.llava.llava import LlavaForEdit
        return LlavaForEdit(model_path, device, True)
    elif 'blip2' in model_name:
        from editor.vllms_for_edit.blip2.blip2 import BLIP2OPTForEdit
        return BLIP2OPTForEdit(model_path, device)
    elif 'mini' in model_name and 'gpt' in model_name and '4' in model_name:
        from editor.vllms_for_edit.minigpt4.minigpt4 import MiniGPT4ForEdit
        return MiniGPT4ForEdit(model_path, device, True)
    raise BaseException('Have not write `BaseVLLMForEdit` for `%s`.'%model_name)

def load_vllm_editor(editor_name:str, edit_model_name:str, device:int, 
        extra_devices:List[int] = [1], editor_ckpt_path = None, for_train = False):
    '''`for_train`: set features of some editors for training during initializing.'''
    editor_name = editor_name.lower()
    config_path = get_editor_config_path(editor_name, edit_model_name)
    vllm = load_vllm_for_edit(edit_model_name, device)
    # load editor
    if editor_name == 'liveedit':
        from editor.vllm_editors.liveedit.liveedit import LiveEdit, LiveEditConfig
        data_proc_device = 'cuda:%s'%extra_devices[0] if for_train else None
        vllm_data_proc = load_vllm_for_edit(edit_model_name, data_proc_device) if for_train else None
        config = LiveEditConfig.from_yaml(config_path)
        editor = LiveEdit(vllm, config, device, vllm_data_proc, data_proc_device) 
    elif editor_name == 'ft_vl':
        from editor.vllm_editors.ft_vl.ft_vl import FTvl, FTvlConfig
        config = FTvlConfig.from_yaml(config_path)
        editor = FTvl(vllm, config, device)
    elif editor_name == 'mend_vl':
        from editor.vllm_editors.mend_vl.mend_vl import MENDvl, MENDvlConfig
        data_proc_device = 'cuda:%s'%extra_devices[0] if for_train else None
        vllm_data_proc = load_vllm_for_edit(edit_model_name, data_proc_device) if for_train else None
        config = MENDvlConfig.from_yaml(config_path)
        editor = MENDvl(vllm, config, device, vllm_data_proc, data_proc_device)
    elif editor_name == 'serac_vl':
        from editor.vllm_editors.serac_vl.serac_vl import SERACvl, SERACvlConfig
        config = SERACvlConfig.from_yaml(config_path)
        editor = SERACvl(vllm, config, device)
    elif editor_name == 'tp_vl': 
        from editor.vllm_editors.tp_vl.tp_vl import TPvl, TPvlConfig
        config = TPvlConfig.from_yaml(config_path)
        editor = TPvl(vllm, config, device)
    elif editor_name == 'lte_vl':
        from editor.vllm_editors.lte_vl.lte_vl import LTEvl, LTEvlConfig
        data_proc_device = 'cuda:%s'%extra_devices[0] if for_train else None
        vllm_data_proc = load_vllm_for_edit(edit_model_name, data_proc_device) if for_train else None
        config = LTEvlConfig.from_yaml(config_path)
        editor = LTEvl(vllm, config, device, vllm_data_proc, data_proc_device)
    elif editor_name == 'recipe_vl':
        from editor.vllm_editors.recipe_vl.recipe_vl import RECIPEvl, RECIPEvlConfig
        config = RECIPEvlConfig.from_yaml(config_path)
        editor = RECIPEvl(vllm, config, device)
    elif editor_name == 'lemoe_vl':
        from editor.vllm_editors.lemoe_vl.lemoe_vl import LEMoEvl, LEMoEvlConfig
        config = LEMoEvlConfig.from_yaml(config_path)
        editor = LEMoEvl(vllm, config, device)
    else:
        raise 'No such editor %s'%editor_name
    if editor_ckpt_path != None and isinstance(editor, VLLMBaseEditorWithTraining):
        editor.load_ckpt(editor_ckpt_path, True, False)
    return editor

