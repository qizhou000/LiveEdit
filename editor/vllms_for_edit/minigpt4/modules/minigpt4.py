from .Qformer import BertConfig, BertLMHeadModel
from torch.cuda.amp import autocast as autocast
from transformers import StoppingCriteriaList
from .conversation import StoppingCriteriaSub
from .base_model import BaseModel
import torch.nn as nn
import random
import torch

class MiniGPT4(BaseModel):
    """MiniGPT-4 model   """
    def __init__(
            self,
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            has_qformer=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            vit_path = None,
            bert_base_uncased_config_path = None
    ):
        super().__init__()
        # For MiniGPTBase
        self.llama_model, self.llama_tokenizer = self.init_llm(
            llama_model_path=llama_model,
            low_resource=low_resource,
            device=device,
            lora_r=0,
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
        )
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, 
            freeze_vit, vit_path)

        self.max_txt_len = max_txt_len
        self.max_context_len = 3800
        self.end_sym = end_sym
        self.prompt_template = prompt_template
        self.prompt_list = []

        # For MiniGPT4
        self.has_qformer = has_qformer
        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features, bert_base_uncased_config_path
            )
            self.load_from_pretrained(filename=q_former_model)  # load q-former weights here

            img_f_dim = self.Qformer.config.hidden_size
            print('Loading Q-Former Done')
        else:
            img_f_dim = self.visual_encoder.num_features * 4
            print('Do not use Q-Former here.')

        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []


    def get_context_emb(self, prompt, img_list):
        # img_list: image vision tokens list
        device = img_list[0].device
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i==0).to(device).input_ids # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    @torch.no_grad()
    def generate(
        self,
        images,
        texts,
        num_beams=1,
        max_new_tokens=20,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=[2],
    ):
        '''
            function for generate test use
        '''

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])

        img_embeds, atts_img = self.encode_img(images.to(self.device))
        image_lists = [[image_emb[None]] for image_emb in img_embeds]

        batch_embs = [self.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1

        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                # stopping_criteria=stopping_criteria,
            )

        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)

        return answers














    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, bert_base_uncased_config_path):
        encoder_config = BertConfig.from_json_file(bert_base_uncased_config_path)
        # encoder_config = OmegaConf.load(bert_base_uncased_config_path)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        for name, param in Qformer.named_parameters():
            param.requires_grad = False
        Qformer = Qformer.eval()
        query_tokens.requires_grad = False

        return Qformer, query_tokens

    def encode_img(self, image):
        device = image.device

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            if self.has_qformer:
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                inputs_llama = self.llama_proj(query_output.last_hidden_state)
            else:
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

                inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        vit_path = cfg.get("vit_path")
        bert_base_uncased_config_path = cfg.get("bert_base_uncased_config_path")
        
        device = cfg.get("device", 0)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            has_qformer=has_qformer,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device=device,
            vit_path = vit_path,
            bert_base_uncased_config_path = bert_base_uncased_config_path
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        print("Load MiniGPT-4 Checkpoint: {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
