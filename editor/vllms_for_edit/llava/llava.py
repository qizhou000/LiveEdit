from typing import List, Union, Optional
from ..base import BaseVLLMForEdit
from PIL.Image import Image as ImageClass
from transformers import  AutoTokenizer
import torch, inspect


class LlavaForEdit(BaseVLLMForEdit):
    '''For llava-v1.5-7b-hf'''
    def __init__(self, model_path:str, device = 'cuda', 
                 auto_add_img_special_token = True) -> None:
        from transformers import LlavaForConditionalGeneration, LlavaProcessor
        self.model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map = device)
        self.processor = LlavaProcessor.from_pretrained(model_path)
        # self.processor.patch_size = self.model.config.vision_config.patch_size
        # self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        self.model = self.model.eval().requires_grad_(False)
        super().__init__(self.model, device, auto_add_img_special_token)

    def get_llm_tokenizer(self):
        return self.processor.tokenizer

    def get_llm_input_embeds(self, texts:List[str], imgs:Optional[List[ImageClass]] = None):
        '''Only support one image in one text.'''
        def get_llava_llm_inpt(mllm, input_ids, attention_mask, pixel_values):
            position_ids = None
            vision_feature_layer = mllm.config.vision_feature_layer
            vision_feature_select_strategy = mllm.config.vision_feature_select_strategy
            # 1. Extra the input embeddings
            inputs_embeds = mllm.get_input_embeddings()(input_ids)
            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = mllm.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {mllm.config.vision_feature_select_strategy}"
                    )
                image_features = mllm.multi_modal_projector(selected_image_feature)
                inputs_embeds, attention_mask, labels, position_ids = mllm._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, None
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, mllm.config.ignore_index).to(torch.long)
            inpt = {'attention_mask': attention_mask, 'inputs_embeds': inputs_embeds, 'position_ids': position_ids}
            return inpt
        inpt = self.processor(texts, imgs, return_tensors = 'pt', padding = True)
        for k, v in inpt.items(): inpt[k] = v.to(self.device) if hasattr(v, 'to') else v
        llm_inpt = get_llava_llm_inpt(self.model, inpt.input_ids, inpt.attention_mask, inpt.pixel_values)
        if imgs != None:
            img_begin = torch.where(inpt['input_ids'][0] == self.get_img_special_token_id())[0][0]
            img_end = img_begin + self.get_img_token_n()
            vt_range = [int(img_begin), int(img_end)]
        else: 
            vt_range = None
        return llm_inpt, vt_range

    def get_llm_outpt(self, llm_inpt, vt_range = None):
        assert 'inputs_embeds' in llm_inpt.keys()
        sig = inspect.signature(self.model.language_model.forward)
        llm_inpt = {k: v for k, v in llm_inpt.items() if k in sig.parameters}
        outpt = self.model.language_model(**llm_inpt, use_cache = False)
        return outpt

    def get_img_special_token_str(self):
        return '<image>'

    def get_img_special_token_id(self):
        return self.model.config.image_token_index
        
    def get_img_token_n(self):
        return (self.model.config.vision_config.image_size//self.model.config.vision_config.patch_size)**2 

    def is_q_former_based(self):
        return False

