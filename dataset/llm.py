#%%
from typing import Dict, List, Tuple, Union
import torch, os, json, re
import threading, time
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from . import BaseEditData

class BaseLLMEditData(BaseEditData):
    '''
    Functions used to read and preprocess various LLM editing datasets, which 
    structures a dataset as a list like [
        { # test1
            'request': {'prompt': str, 'target_new': str, ...},
            'generality': {
                'gen_1_name':[
                    {'prompt': str, 'target': str, ...},
                    {'prompt': str, 'target': str, ...}, ...
                ],
                'gen_2_name':[...], ...
            },
            'locality': {
                'loc_1_name':[
                    {'prompt': str, 'target': str, ...}, ...
                ],
                'loc_2_name':[...], ...
            }
        }, 
        { # test2
            'request': ...
        }, ...
    ]. 
    '''
    def __init__(self) -> None:
        pass

