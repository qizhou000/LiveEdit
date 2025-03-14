from typing import Dict, List, Tuple, Optional, Union
from torch.utils.tensorboard import SummaryWriter 
from ..vllms_for_edit.base import BaseVLLMForEdit
from dataset.vllm import BaseVLLMEditData
from abc import ABC, abstractmethod
from dataset import ParallelDataset
from dataclasses import asdict
from datetime import datetime
from ..base import BaseConfig
import json, yaml, os, torch
from copy import deepcopy
from tqdm import tqdm
from torch import nn
import numpy as np

        
############################################################################
####################### VLLM Base Editor Classes ########################### 
############################################################################
class VLLMBaseEditor(ABC):
    def __init__(self, vllm:BaseVLLMForEdit, device='cuda'):
        if not isinstance(vllm, BaseVLLMForEdit): raise
        self.vllm = vllm
        self.vllm.set_device(device)
        self.device = device if device != 'auto' else 'cuda:0'
        assert self.if_model_decoder_only() # temporary only support decoder-only llm

    def if_model_decoder_only(self)->bool:
        if self.vllm.model.config.is_encoder_decoder:
            return False
        return True

    @abstractmethod
    def name_of_editor_and_model(self)->Tuple[str, str]:
        '''
        return editor_name:str, model_name:str
        '''

    @abstractmethod
    def restore_to_original_model(self):
        '''
        A method for restoring the original model weights after editing with as 
        low GPU memory usage as possible.
        '''

    @abstractmethod
    def edit_one_piece(self, request:Dict):
        '''
        request = {'image': PILImage, 'prompt': str, 'target_new': str, ...}
        '''

    @abstractmethod
    def edit_batch(self, requests:List[Dict]):
        '''Assume: 
        requests = [
          {'image': PILImage, 'prompt': str, 'target_new': str, ...},
          {'image': PILImage, 'prompt': str, 'target_new': str, ...}, ...
        ]
        '''

    @abstractmethod
    def if_can_batch_edit(self)->bool:
        pass



class VLLMBaseEditorWithTraining(VLLMBaseEditor):
    def __init__(self, vllm:BaseVLLMForEdit, config:BaseConfig, device='cuda'):
        super().__init__(vllm, device)
        self.cfg = config 

    @abstractmethod
    def get_modules_for_training(self)->Dict[str, nn.Module]:
        '''
        Get modules for training, used for `self.save_ckpt` and `self.load_ckpt`.
        Assume return `train_modules`: Dict[str, nn.Module]
        '''
    
    @abstractmethod
    def reinit_train_parameters(self):
        '''Reinitialize parameters of modules to be trained.'''

    @abstractmethod
    def preprocess_train_data(self, vllm_edit_data:BaseVLLMEditData)->List:
        '''Process raw training data, used in `self.train_init`. 
        Return the data list used to sample batch data every iteration.'''
        
    @abstractmethod
    def organize_batch_data(self, a_batch_of_training_data:List):
        '''
        This function is used to dynamically organized data during training in 
            `self.data_generator`.
        `a_batch_of_training_data`: a batch/list of training data.
        return `a_batch_of_organized_training_data`
        '''
        
    @abstractmethod
    def train_a_batch(self, a_batch_of_organized_training_data):
        '''
        This function would be inputted a batch of organized training data and 
        should train the editor once. 
        `a_batch_of_organized_training_data`: a batch of organized training data 
            coming from `self.organize_batch_data`.
        return loss:float, log_dict: Dict[str, int]
        '''
    
    @abstractmethod
    def get_a_new_optimizer(self)->Union[torch.optim.Optimizer, 
        Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]]:
        ''' Initialize optimizer for training.
            Assume return an `opt`: 
                torch.optim.Optimizer
            or an `opt` and its learning rate scheduler: 
                [torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler].
        '''

    @abstractmethod
    def set_train(self, is_train:bool):
        '''Set training state for editor. '''

    def set_random_seeds(self, seed:int):
        import random, time
        from torch.backends import cudnn
        if seed == None:
            seed = int(time.time()*10000)%99999999
        print('Random seed is', seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        np.random.seed(seed)  
        random.seed(seed)   
        cudnn.benchmark = False # Do not test and select a optim algorithm for CNN
        cudnn.deterministic = True # Deterministic mode for Convolution/Pooling
        self.random_seed = seed

    def other_train_init_final(self):
        '''Called at the end of `self.train_init`.'''

    def other_train_init_begin(self):
        '''Called at the begin of `self.train_init`.'''

    def train_init(self, vllm_edit_data:BaseVLLMEditData, batch_size:int, 
            records_dir:str = 'records', train_name_prefix:str = None, 
            train_name:str = None, load_ckpt_path:str = None, 
            save_ckpt_per_i:int = 3000, log_per_i:int = 10, 
            ema_alpha:float = 0.1, random_seed:int = None, 
            data_buffer_size = 8, seed_init_train_params_if_no_ckpt_path = True):  
        '''Used to initialize data generator `self.data_generator`, checkpoint/log 
            directory, writer, and optimizer. '''
        self.set_random_seeds(random_seed)
        self.other_train_init_begin()
        # initialize data generator
        def get_data_by_ids_func(ids):
            a_batch_of_training_data = [training_data[i] for i in ids]
            a_batch_of_organized_training_data = self.organize_batch_data(a_batch_of_training_data)
            return a_batch_of_organized_training_data
        assert isinstance(vllm_edit_data, BaseVLLMEditData)
        training_data = self.preprocess_train_data(vllm_edit_data)
        self.data_generator = ParallelDataset(len(training_data), get_data_by_ids_func, 
            batch_size, True, data_buffer_size, False, self.random_seed, True)
        # initialize checkpoint/log directory and writer
        t = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        train_name = (train_name_prefix + '-' if train_name_prefix else "") + \
            (train_name if train_name else t)
        records_dir = os.path.join(records_dir, *self.name_of_editor_and_model(), train_name)
        self.save_ckpt_dir = os.path.join(records_dir, 'checkpoints')
        os.makedirs(self.save_ckpt_dir, exist_ok = True)
        logs_path = os.path.join(records_dir, 'logs')
        os.makedirs(logs_path, exist_ok = True)
        with open(os.path.join(records_dir, 'config.yaml'), 'w') as f:
            cfg = deepcopy(self.cfg)
            cfg.train_batch_size = batch_size
            cfg.random_seed = self.random_seed
            yaml.dump(asdict(cfg), f)
        self.log_writer = SummaryWriter(logs_path)
        self.save_ckpt_per_i = save_ckpt_per_i
        self.log_per_i = log_per_i
        self.ema_alpha = ema_alpha
        # initialize optimizer and load checkpoints
        opt = self.get_a_new_optimizer()
        self.opt, self.lr_scheduler = [opt, None] if isinstance(opt, torch.optim.Optimizer) else opt
        if load_ckpt_path:
            assert os.path.isfile(load_ckpt_path)
            self.train_i, self.train_epoch, _, self.ema_loss = self.load_ckpt(load_ckpt_path, True)
        else:
            if seed_init_train_params_if_no_ckpt_path:
                print('Train parameters are reinitialized with seed %s.'%self.random_seed)
                self.reinit_train_parameters()
            self.train_i = self.train_epoch = self.ema_loss = 1
        # initialize other settings
        self.other_train_init_final()


    def train(self, total_epochs):
        if self.log_writer == None:
            raise "Call `self.train_init()` to initialize training first!"
        print('Checkpoints dir: ', self.save_ckpt_dir)
        start_epoch = self.train_epoch
        self.set_train(True) 
        for self.train_epoch in range(start_epoch, total_epochs + 1): 
            progress_bar = tqdm(total = self.data_generator.sample_count, 
                position = 0, leave = True, desc = "Epoch %d"%self.train_epoch, dynamic_ncols = True)
            for a_batch_samples, samp_n in self.data_generator:
                # train after edit
                loss, log_dict = self.train_a_batch(a_batch_samples)
                self.ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self.ema_loss
                # log
                log_dict['Loss'] = loss
                log_dict['EMA Loss'] = self.ema_loss
                log_dict['Epoch'] = self.train_epoch
                if self.train_i % self.log_per_i == 0:
                    self.write_logs(self.train_i, log_dict)
                if self.train_i % self.save_ckpt_per_i == 0:
                    self.save_ckpt(self.train_i, self.train_epoch, loss, self.ema_loss)
                self.train_i += 1 
                progress_bar.update(samp_n)
            progress_bar.close() 
        self.set_train(False)

    def write_logs(self, i, logs:dict):
        for log_name, log in logs.items():
            if type(log) == dict:
                logs1 = {}
                for n, l in log.items():
                    logs1[log_name + '-' + n] = l
                self.write_logs(i, logs1)
            else:   
                self.log_writer.add_scalar(log_name, log, i)

    def save_ckpt(self, i:int, epoch:int, loss:float, ema_loss:float = None):
        '''Save checkpoint.'''
        train_modules = self.get_modules_for_training()
        ckpt = {
            'i': i,
            'epoch': epoch,
            'loss': loss,
            'ema_loss': ema_loss,
            'train_modules': {k:v.state_dict() for k, v in train_modules.items()},
            'opt': self.opt.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler != None else None
        }
        if ema_loss != None:
            ckpt_name = 'epoch-%d-i-%d-ema_loss-%.4f'%(int(epoch), int(i), float(ema_loss))
        else:
            ckpt_name = 'epoch-%d-i-%d-loss-%.4f'%(int(epoch), int(i), float(loss))
        ckpt_path = os.path.join(self.save_ckpt_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)

    def load_ckpt(self, ckpt_path, restrict = True, load_opt = True):
        '''Load checkpoint.'''
        ckpt = torch.load(ckpt_path, 'cpu')
        train_modules = self.get_modules_for_training()
        for k in train_modules.keys():
            train_modules[k].load_state_dict(ckpt['train_modules'][k], restrict)
        if load_opt:
            self.opt.load_state_dict(ckpt['opt'])
            if self.lr_scheduler != None and ckpt['lr_scheduler'] != None:
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        print('Load %s checkpoint from %s.'%(self.name_of_editor_and_model()[0], ckpt_path))
        return ckpt['i'], ckpt['epoch'], ckpt['loss'], ckpt['ema_loss']
