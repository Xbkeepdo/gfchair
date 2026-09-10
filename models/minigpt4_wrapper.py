"""MiniGPT-4 Vicuna-v0, using the released EVA/Q-Former architecture."""
from functools import partial
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import LlamaForCausalLM, LlamaTokenizer

from models.visual_prefix_wrapper import PrefixProcessor, VisualPrefixModel, VisualPrefixWrapper


def query_patch_mapping(attention, grid):
    """Last cross-attention layer, mean heads, remove CLS, normalize queries."""
    mapping = attention.float().mean(dim=1)[0, :, 1:]
    if mapping.shape[-1] != grid[0]*grid[1] or not torch.isfinite(mapping).all() or (mapping < 0).any():
        raise ValueError('Invalid Q-Former spatial attention')
    sums = mapping.sum(-1, keepdim=True)
    if (sums <= 0).any():
        raise ValueError('Q-Former query has no patch mass')
    return mapping / sums


class MiniGPT4Wrapper(VisualPrefixWrapper):
    stop_separator = '###'

    def _load_model(self):
        from models.vendor_minigpt4.eva_vit import VisionTransformer
        from models.vendor_minigpt4.Qformer import BertConfig, BertLMHeadModel
        root = Path(self.cfg['hf_name'])
        self.tokenizer = LlamaTokenizer.from_pretrained(root/'vicuna-7b-v0', legacy=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        lm, info = LlamaForCausalLM.from_pretrained(root/'vicuna-7b-v0', dtype=torch.float16,
            device_map=self.device, attn_implementation='eager', output_loading_info=True)
        if info['missing_keys'] or info['mismatched_keys']:
            raise ValueError(f'Incomplete Vicuna weights: {info}')
        self.model = VisualPrefixModel(lm, -200).eval()
        self.grid = (16, 16)
        self.cfg['num_visual_tokens'] = 32
        # Vision weights live on CPU between images to reserve GPU space for JVP.
        with torch.device('meta'):
            self.vision = VisionTransformer(img_size=224, patch_size=14, use_mean_pooling=False,
                embed_dim=1408, depth=39, num_heads=16, mlp_ratio=4.3637, qkv_bias=True,
                drop_path_rate=0, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), use_checkpoint=False)
        state = torch.load(root/'eva_vit_g.pth', map_location='cpu', weights_only=False, mmap=True)
        # Official EVA uses 39 blocks from the 40-block classifier checkpoint.
        allowed = ('blocks.39.', 'norm.', 'head.')
        extras = set(state) - set(self.vision.state_dict())
        if any(not key.startswith(allowed) for key in extras):
            raise ValueError(f'Unexpected EVA weights: {extras}')
        self.vision.load_state_dict({k:v for k,v in state.items() if k not in extras}, strict=True, assign=True)
        self.vision = self.vision.half().eval()
        del state
        config = BertConfig(encoder_width=1408, add_cross_attention=True,
                            cross_attention_freq=2, query_length=32)
        self.qformer = BertLMHeadModel(config)
        self.qformer.cls = None
        self.qformer.bert.embeddings.word_embeddings = None
        self.qformer.bert.embeddings.position_embeddings = None
        for layer in self.qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.ln_vision = torch.nn.LayerNorm(1408)
        state = torch.load(root/'blip2_pretrained_flant5xxl.pth', map_location='cpu', weights_only=False, mmap=True)['model']
        qstate = {k[len('Qformer.'):]: v for k,v in state.items() if k.startswith('Qformer.')}
        missing, unexpected = self.qformer.load_state_dict(qstate, strict=False)
        if missing or unexpected:
            raise ValueError(f'Q-Former weight mismatch: {missing}, {unexpected}')
        self.ln_vision.load_state_dict({k[len('ln_vision.'):]:v for k,v in state.items() if k.startswith('ln_vision.')}, strict=True)
        self.query_tokens = state['query_tokens'].to(self.device, torch.float16)
        del state, qstate
        self.projector = torch.nn.Linear(768, lm.config.hidden_size)
        state = torch.load(root/'pretrained_minigpt4_7b.pth', map_location='cpu', weights_only=False)['model']
        self.projector.load_state_dict({k[len('llama_proj.'):]:v for k,v in state.items() if k.startswith('llama_proj.')}, strict=True)
        self.qformer.to(self.device, torch.float16).eval()
        self.ln_vision.to(self.device, torch.float16).eval()
        self.projector.to(self.device, torch.float16).eval()
        self.preprocess = transforms.Compose([transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
        self.processor = PrefixProcessor(self)

    def _format_prompt(self, raw_prompt):
        return ('Give the following image: <Img>ImageContent</Img>. You will be able to see the image '
                'once I provide it to you. Please answer my questions.###Human: <Img><ImageHere></Img> '
                + raw_prompt + '###Assistant:')

    @torch.no_grad()
    def encode_image(self, image):
        self.vision.to(self.device)
        try:
            features = self.vision(self.preprocess(image).unsqueeze(0).to(self.device, torch.float16))
        finally:
            self.vision.cpu()
        features = self.ln_vision(features)
        output = self.qformer.bert(query_embeds=self.query_tokens,
            encoder_hidden_states=features, encoder_attention_mask=torch.ones(features.shape[:2], device=self.device, dtype=torch.long),
            output_attentions=True, return_dict=True)
        # Released BertEncoder also appends KV tuples for non-cross layers.
        cross = [a for a in output.cross_attentions if torch.is_tensor(a)]
        self.query_to_patch = query_patch_mapping(cross[-1], self.grid).cpu()
        return self.projector(output.last_hidden_state)

    def _visual_grid_for_output(self, inputs, visual_start, visual_end):
        # Learned queries have no spatial ordering.
        return None

    def attach_spatial_attention(self, output):
        if output.text_to_patch_attn.numel():
            output.spatial_attention = output.text_to_patch_attn.float() @ self.query_to_patch
            output.spatial_grid = self.grid
