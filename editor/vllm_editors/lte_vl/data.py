#%%
import numpy as np
import os, json, re, torch
from typing import Dict, List, Union
from utils.GLOBAL import ROOT_PATH
from transformers import  AutoTokenizer 
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def set_tokenizer_pad_id(tokenizer:AutoTokenizer):
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print('Set [pad_token] as [eos_token].')


def pt2xym(tokenizer, prompt:str, target:str):
    target = target if target[0] == ' ' else ' ' + target
    prompt_ids = tokenizer(prompt, return_tensors = 'pt').input_ids[0]
    all_ids = tokenizer(prompt + target, return_tensors = 'pt').input_ids[0]
    label_ids = all_ids[1:]
    input_ids = all_ids[:-1]
    mask = torch.zeros(len(label_ids))
    mask[len(prompt_ids)-1:] += 1
    return input_ids, label_ids, mask

def stack_xym(tokenizer, x_list, y_list, m_list, device):
    x = pad_sequence(x_list, True, tokenizer.pad_token_id).to(device)
    y = pad_sequence(y_list, True, tokenizer.pad_token_id).to(device)
    m = pad_sequence(m_list, True, 0).to(device)
    return x, y, m

class LTETrainData():
    def __init__(self, tokenizer:AutoTokenizer, data_n = None, data_name = 'cf', 
            data_path = None, device='cuda', seed = 0) -> None:
        set_tokenizer_pad_id(tokenizer)
        if data_name.lower() in ['cf', 'counterfact']:
            if data_path == None:
                data_path = os.path.join(ROOT_PATH, 'data/meta-train/cf/cf-no-repeat-subject-train.json')
            data = self.__cf__(tokenizer, data_n, data_path)
        elif data_name.lower() in ['zsre']:
            if data_path == None:
                data_path = os.path.join(ROOT_PATH, 'data/meta-train/zsre/zsre_mend_train.json')
            data = self.__zsre__(tokenizer, data_n, data_path)
        elif data_name.lower() in ['ripe', 'ripple effect']:
            if data_path == None:
                data_path = os.path.join(ROOT_PATH, 'data/meta-train/ripple_effect/ripe_train.json')
            data = self.__ripe__(tokenizer, data_n, data_path)
        else:
            raise KeyError
        self.sample_count, (self.knowledges, self.relia_xym, self.gen_xym, self.loc_xym) = data
        self.tokenizer = tokenizer
        self.device = device
        self.rng = np.random.default_rng(seed)

    def get_knowledge(self, prompt, target):
        k = prompt + ' ' if prompt[-1] != ' ' and target[0] != ' ' else prompt
        k += target
        return k

    # Counterfact
    def __cf__(self, tokenizer:AutoTokenizer, data_n, data_path):
        with open(data_path, 'r') as f: 
            data = json.load(f)
        sample_count = min(len(data), data_n) if data_n != None else len(data)
        knowledges, relia_xym, gen_xym, loc_xym = [], [], [], []
        for d in tqdm(data[:sample_count], 'Counterfact data preparing...'):
            knowledges.append(self.get_knowledge(d['prompt'], d['target_new']))
            relia_xym.append(pt2xym(tokenizer, d['prompt'], d['target_new']))
            gen_xym.append({'rephrase': [pt2xym(tokenizer, d['rephrase_prompt'], d['target_new'])]})
            loc_xym.append({'original': [pt2xym(tokenizer, d['locality_prompt'], d['locality_ground_truth'])]})
        return sample_count, (knowledges, relia_xym, gen_xym, loc_xym)

    # zsre
    def __zsre__(self, tokenizer:AutoTokenizer, data_n, data_path):
        with open(data_path, 'r') as f: 
            data = json.load(f)
        sample_count = min(len(data), data_n) if data_n != None else len(data)
        knowledges, relia_xym, gen_xym, loc_xym = [], [], [], []
        for d in tqdm(data[:sample_count], 'ZSRE data preparing...'):
            knowledges.append(self.get_knowledge(d['src'], d['alt']))
            relia_xym.append(pt2xym(tokenizer, d['src'], d['alt']))
            gen_xym.append({'rephrase': [pt2xym(tokenizer, d['rephrase'], d['alt'])]})
            loc_xym.append({'original': [pt2xym(tokenizer, d['loc'], d['loc_ans'])]})
        return sample_count, (knowledges, relia_xym, gen_xym, loc_xym)

    # ripe
    def __ripe__(self, tokenizer:AutoTokenizer, data_n, data_path):
        def get_pt_xym_from_a_type(type_data_list:List[Dict[str, Union[str, List]]]):
            pts = []
            for pt in type_data_list:
                for t in pt['targets']:
                    if t != "":
                        pts.append((pt['prompt'], t))
                        break
            xym = [pt2xym(tokenizer, pt[0], pt[1]) for pt in pts]
            return pts, xym
        with open(data_path, 'r') as f: 
            data = json.load(f)
        gen_types = ['Logical_Generalization', 'Compositionality_I', 
                            'Compositionality_II', 'Subject_Aliasing']
        loc_types = ['Relation_Specificity', 'Forgetfulness']
        knowledges, relia_xym, gen_xym, loc_xym = [], [], [], []
        data_n = len(data) if data_n == None else data_n
        bar = tqdm(total = data_n, desc='Ripple Effect data preparing...')
        now_data_n = 0
        for d in data:
            new_gen_xym, new_loc_xym = {}, {}
            for gen_type in gen_types:
                pts, xym = get_pt_xym_from_a_type(d[gen_type])
                if pts != []:
                    new_gen_xym[gen_type] = xym
            for loc_type in loc_types:
                pts, xym = get_pt_xym_from_a_type(d[loc_type])
                if pts != []:
                    new_loc_xym[loc_type] = xym
            if len(new_gen_xym) != 0 and len(new_loc_xym) != 0:
                knowledges.append(self.get_knowledge(d['prompt'], d['target_new']))
                relia_xym.append(pt2xym(tokenizer, d['prompt'], d['target_new']))
                gen_xym.append(new_gen_xym)
                loc_xym.append(new_loc_xym)
                now_data_n += 1
                bar.update(1)
                if now_data_n >= data_n:
                    break 
        return now_data_n, (knowledges, relia_xym, gen_xym, loc_xym)
            
    def get_data_by_ids(self, ids:List[int]):
        def random_select(d:Dict[str, List]):
            ks = list(d.keys())
            d = d[ks[self.rng.integers(0, len(ks))]]
            return d[self.rng.integers(0, len(d))]
        # knowledges
        knowledges = [self.knowledges[i] for i in ids]
        # reliability data
        x_list, y_list, m_list = [[self.relia_xym[i][j] for i in ids] for j in range(3)]
        batch_relia_xym = stack_xym(self.tokenizer, x_list, y_list, m_list, self.device)
        # generality data
        x_list, y_list, m_list = [], [], []
        for i in ids:
            xym = random_select(self.gen_xym[i])
            x_list.append(xym[0])
            y_list.append(xym[1])
            m_list.append(xym[2])
        batch_gen_xym = {'original': stack_xym(self.tokenizer, x_list, y_list, m_list, self.device)}
        # locality data
        x_list, y_list, m_list = [], [], []
        for i in ids:
            xym = random_select(self.loc_xym[i])
            x_list.append(xym[0])
            y_list.append(xym[1])
            m_list.append(xym[2])
        batch_loc_xym = {'original_loc': stack_xym(self.tokenizer, x_list, y_list, m_list, self.device)}
        return knowledges, batch_relia_xym, batch_gen_xym, batch_loc_xym

