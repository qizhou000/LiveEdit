from typing import List, Union, Optional
from ..base import BaseVLLMForEdit
from PIL.Image import Image as ImageClass
from transformers import  AutoTokenizer
import torch

class BLIP2OPTForEdit(BaseVLLMForEdit):
    '''For blip2-opt 2.7b'''
    def __init__(self, model_path:str, device = 'cuda') -> None:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_path, device_map = device)
        self.processor = Blip2Processor.from_pretrained(model_path)
        self.model = self.model.eval().requires_grad_(False)
        super().__init__(self.model, device, False)

    def get_llm_tokenizer(self):
        return self.processor.tokenizer

    def get_llm_input_embeds(self, texts:List[str], imgs:Optional[List[ImageClass]] = None):
        '''Only support one image in one text.'''
        def get_blip2_llm_inpt(pixel_values, input_ids, attention_mask):
            # step 1: forward the images through the vision encoder,
            # to get image embeddings of shape (batch_size, seq_len, hidden_size)
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,
            )
            image_embeds = vision_outputs[0]
            # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,
            )
            query_output = query_outputs[0]
            # step 3: use the language model, conditioned on the query outputs and the prompt
            language_model_inputs = self.model.language_projection(query_output)
            language_model_attention_mask = torch.ones(
                language_model_inputs.size()[:-1], dtype=torch.long, device=self.device)
            inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(self.device)], dim=1)
            attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(self.device)], dim=1)
            inpt = {'attention_mask': attention_mask, 'inputs_embeds': inputs_embeds}
            return inpt
        if imgs != None:
            inpt = self.processor(imgs, texts, return_tensors = 'pt', padding = True)
            inpt = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inpt.items()}
            llm_inpt = get_blip2_llm_inpt(inpt['pixel_values'], inpt['input_ids'], inpt['attention_mask'])
        else:
            inpt = self.get_llm_tokenizer()(texts, return_tensors = 'pt', padding = True).to(self.device)
            inputs_embeds = self.model.language_model.get_input_embeddings()(inpt.input_ids)
            llm_inpt = {'attention_mask': inpt.attention_mask, 'inputs_embeds': inputs_embeds}
        vt_range = None if imgs == None else [0, self.get_img_token_n()]
        return llm_inpt, vt_range

    def get_llm_outpt(self, llm_inpt, vt_range = None):
        outpt = self.model.language_model(
            inputs_embeds=llm_inpt['inputs_embeds'],
            attention_mask=llm_inpt['attention_mask'],
            output_attentions=None, output_hidden_states=None,
            return_dict=True, use_cache = False
        )
        return outpt

    def get_img_special_token_str(self):
        return None

    def get_img_special_token_id(self):
        return None
        
    def get_img_token_n(self):
        return self.model.config.num_query_tokens

    def is_q_former_based(self):
        return True

