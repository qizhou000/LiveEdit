from typing import List, Union, Optional
from ..base import BaseVLLMForEdit
from torch.nn.utils.rnn import pad_sequence
from PIL.Image import Image as ImageClass
from omegaconf import OmegaConf
import torch, inspect, os
from PIL import Image 

class MiniGPT4ForEdit(BaseVLLMForEdit):
    '''For MiniGPT4'''
    def __init__(self, model_path:str, device = 'cuda', auto_add_img_special_token = True) -> None:
        import types
        from editor.vllms_for_edit.minigpt4.modules.minigpt4 import MiniGPT4
        from editor.vllms_for_edit.minigpt4.modules.blip_processors import Blip2ImageEvalProcessor
        model_config = OmegaConf.load(os.path.join(model_path, 'model_config.yaml'))
        model_config.llama_model = os.path.join(model_path, model_config.llama_model)
        model_config.vit_path = os.path.join(model_path, model_config.vit_path)
        model_config.bert_base_uncased_config_path = os.path.join(model_path, model_config.bert_base_uncased_config_path)
        model_config.q_former_model = os.path.join(model_path, model_config.q_former_model)
        model_config.ckpt = os.path.join(model_path, model_config.ckpt)
        model_config.device = device
        self.model = MiniGPT4.from_config(model_config).to(device)
        self.model.config = types.SimpleNamespace()
        self.model.config.is_encoder_decoder = False
        self.model = self.model.eval().requires_grad_(False)
        processor_config = OmegaConf.load(os.path.join(model_path, 'processor_config.yaml'))
        self.img_processor = Blip2ImageEvalProcessor.from_config(processor_config)
        super().__init__(self.model, device, auto_add_img_special_token)

    def get_llm_tokenizer(self):
        return self.model.llama_tokenizer

    def get_llm_input_embeds(self, texts:List[str], imgs:Optional[List[ImageClass]] = None):
        '''Only support one image in one text.'''
        def get_minigpt4_llm_inpt(texts, imgs): 
            processed_imgs = torch.stack([self.img_processor(img).to(self.device) for img in imgs], 0)
            img_embeds, atts_img = self.model.encode_img(processed_imgs)
            image_lists = [[image_emb[None]] for image_emb in img_embeds]

            batch_embs = [self.model.get_context_emb(text, img_list)[0] for text, img_list in zip(texts, image_lists)]

            embs = pad_sequence(batch_embs, batch_first=True)
            attn_mask = pad_sequence([torch.ones(len(e), dtype=torch.int, device=self.device) 
                                      for e in batch_embs], batch_first=True)
            return {'inputs_embeds': embs, 'attention_mask': attn_mask}
        if imgs != None:
            llm_inpt = get_minigpt4_llm_inpt(texts, imgs)
        else:
            inpt = self.get_llm_tokenizer()(texts, return_tensors = 'pt', padding = True).to(self.device)
            inputs_embeds = self.model.llama_model.get_input_embeddings()(inpt.input_ids)
            llm_inpt = {'attention_mask': inpt.attention_mask, 'inputs_embeds': inputs_embeds}
        if self.auto_add_img_special_token:
            vt_range = None if imgs == None else [1, self.get_img_token_n()+1]
        else:
            raise
        return llm_inpt, vt_range

    def get_llm_outpt(self, llm_inpt, vt_range = None):
        assert 'inputs_embeds' in llm_inpt.keys()
        sig = inspect.signature(self.model.llama_model.forward)
        llm_inpt = {k: v for k, v in llm_inpt.items() if k in sig.parameters}
        outpt = self.model.llama_model(**llm_inpt, use_cache = False)
        return outpt

    def get_img_special_token_str(self):
        return '<ImageHere>'

    def get_img_special_token_id(self):
        raise
         
    def get_img_token_n(self):
        return 32

    def is_q_former_based(self):
        return True
