#%%
from typing import Dict, List, Tuple, Union
from . import BaseEditData
from copy import deepcopy
import torch, os, json, re
from PIL import Image
from tqdm import tqdm


class BaseVLLMEditData(BaseEditData):
    '''
    Functions used to read and preprocess VLLM editing datasets, which should be
        structured as a list like [
            { # test1
                'requests': [
                    {'image': PILImage, 'prompt': str, 'target_new': str, ...},
                    {'image': PILImage, 'prompt': str, 'target_new': str, ...}, ...
                ],
                'generality': {
                    'gen_1_name':[
                        {'image': PILImage, 'prompt': str, 'target': str, ...},
                        {'image': PILImage, 'prompt': str, 'target': str, ...}, ...
                    ],
                    'gen_2_name':[...], ...
                },
                'locality': {
                    'loc_1_name':[
                        {'image': PILImage, 'prompt': str, 'target': str, ...}, ...
                    ],
                    'loc_2_name':[...], ...
                }
            }, 
            { # test2
                'requests': ...
            }, ...
        ]
    '''
    def __init__(self, data_with_img, data_with_img_path) -> None:
        super().__init__(data_with_img) 
        self.data = data_with_img
        self.data_with_img = data_with_img
        self.data_with_img_path = data_with_img_path

    def __load_imgs_for_data_with_img_path__(self, d:Union[List, Dict, str]):
        if isinstance(d, dict):
            for k in d.keys():
                if k == 'image':
                    if d[k] != None:
                        d[k] = Image.open(d[k]) 
                else:
                    self.__load_imgs_for_data_with_img_path__(d[k])
        elif isinstance(d, list):
            for i in d:
                self.__load_imgs_for_data_with_img_path__(i)
        elif isinstance(d, str): return
        else: raise
    
    def get_data_with_img_path(self):
        return self.data_with_img_path

    def __init_eic_evqa__(self, data_path:str, img_root_dir:str, data_n = None):
        if data_n == None: data_n = 99999999
        with open(data_path, 'r') as f:
            data = json.load(f)
        data_n = min(len(data), data_n)
        return_data = []
        for i in tqdm(range(data_n), 'Loading data'):
            d = data[i]
            new_d = {'requests': [{}], 
                     'generality': {'text_rephrase': [], 'image_rephrase': []}, 
                     'locality': {'text_loc': [], 'image_loc': []}}
            # requests
            new_d['requests'][0]['image'] = os.path.join(img_root_dir, d['image'])
            new_d['requests'][0]['prompt'] = d['src']
            new_d['requests'][0]['target_new'] = d['alt']
            # generality
            a_gen_data = {'image': new_d['requests'][0]['image'], 'prompt': d['rephrase'], 'target': d['alt']}
            new_d['generality']['text_rephrase'].append(a_gen_data)
            a_gen_data = {'image': os.path.join(img_root_dir, d['image_rephrase']), 
                          'prompt': d['src'], 'target': d['alt']}
            new_d['generality']['image_rephrase'].append(a_gen_data)
            # locality
            a_loc_data = {'image': None, 'prompt': d['loc'], 'target': d['loc_ans']}
            new_d['locality']['text_loc'].append(a_loc_data)
            a_loc_data = {'image': os.path.join(img_root_dir, d['m_loc']), 
                          'prompt': d['m_loc_q'], 'target': d['m_loc_a']}
            new_d['locality']['image_loc'].append(a_loc_data)
            return_data.append(new_d)
        return return_data


class EVQA(BaseVLLMEditData):
    def __init__(self, data_path:str = 'data/easy-edit-mm/vqa/vqa_train.json', 
                  img_root_dir:str = 'data/easy-edit-mm/images', data_n = None) -> None:
        if 'vqa' not in os.path.basename(data_path): raise
        print('Load E-VQA from: %s '% data_path)
        data_with_img_path = self.__init_eic_evqa__(data_path, img_root_dir, data_n)
        for d in data_with_img_path:
            d['requests'][0]['prompt'] = '%s The answer is:'%d['requests'][0]['prompt']
            d['generality']['text_rephrase'][0]['prompt'] = '%s The answer is:'%d['generality']['text_rephrase'][0]['prompt']
            d['generality']['image_rephrase'][0]['prompt'] = '%s The answer is:'%d['generality']['image_rephrase'][0]['prompt']
            d['locality']['text_loc'][0]['prompt'] = '%s?'%d['locality']['text_loc'][0]['prompt']
            d['locality']['image_loc'][0]['prompt'] = '%s The answer is:'%d['locality']['image_loc'][0]['prompt']
        data_with_img = deepcopy(data_with_img_path)
        for d in tqdm(data_with_img, 'Loading images'):
            self.__load_imgs_for_data_with_img_path__(d)
        super().__init__(data_with_img, data_with_img_path)

    def dataset_name(self):
        return 'EVQA'


class EIC(BaseVLLMEditData):
    def __init__(self, data_path:str = 'data/easy-edit-mm/caption/caption_train_edit.json', 
                  img_root_dir:str = 'data/easy-edit-mm/images', data_n = None):
        if 'caption' not in os.path.basename(data_path): raise
        print('Load E-IC from: %s '% data_path)
        data_with_img_path = self.__init_eic_evqa__(data_path, img_root_dir, data_n)
        for d in data_with_img_path:
            d['locality']['text_loc'][0]['prompt'] = '%s?'%d['locality']['text_loc'][0]['prompt']
            d['locality']['image_loc'][0]['prompt'] = '%s The answer is:'%d['locality']['image_loc'][0]['prompt']
        data_with_img = deepcopy(data_with_img_path)
        for d in tqdm(data_with_img, 'Loading images'):
            self.__load_imgs_for_data_with_img_path__(d)
        super().__init__(data_with_img, data_with_img_path)

    def dataset_name(self):
        return 'EIC'


class VLKEB(BaseVLLMEditData):
    def __init__(self, data_path:str = 'data/VLKEB/train.json', 
                  img_root_dir:str = 'data/VLKEB/mmkb_images', data_n = None):
        print('Load VLKEB from: %s '% data_path)
        data_with_img_path = self.__init_eic_evqa__(data_path, img_root_dir, data_n)
        for d in data_with_img_path:
            d['locality']['text_loc'][0]['prompt'] = '%s?'%d['locality']['text_loc'][0]['prompt']
            d['locality']['image_loc'][0]['prompt'] = '%s The answer is:'%d['locality']['image_loc'][0]['prompt']
        data_with_img = deepcopy(data_with_img_path)
        for d in tqdm(data_with_img, 'Loading images'):
            self.__load_imgs_for_data_with_img_path__(d)
        super().__init__(data_with_img, data_with_img_path)

    def dataset_name(self):
        return 'VLKEB'




# d = VLKEB(data_n=10)
# #%%
# d.data[0]['requests'][0]['image'].show()
# d.data[0]['requests']
# d.data[0]['generality']
# d.data[0]['locality']