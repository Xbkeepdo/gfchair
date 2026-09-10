"""Shikra's native CLIP/-2 + linear projection and Vicuna image prefix.

Reference: https://github.com/shikras/shikra/blob/main/mllm/models/shikra/shikra.py
The local checkpoint is already the merged model, not a delta.
"""
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModel, LlamaConfig, LlamaForCausalLM, LlamaTokenizer

from models.visual_prefix_wrapper import PrefixProcessor, VisualPrefixModel, VisualPrefixWrapper


class ShikraWrapper(VisualPrefixWrapper):
    def _load_model(self):
        root = Path(self.cfg['hf_name'])
        config = LlamaConfig.from_pretrained(root)
        self.tokenizer = LlamaTokenizer.from_pretrained(root, legacy=True)
        lm, info = LlamaForCausalLM.from_pretrained(root, config=config, dtype=torch.float16,
            device_map=self.device, attn_implementation='eager', output_loading_info=True)
        if info['missing_keys'] or info['mismatched_keys']:
            raise ValueError(f'Incomplete Shikra LM: {info}')
        unexpected = [k for k in info['unexpected_keys'] if not k.startswith('model.mm_projector.')]
        if unexpected:
            raise ValueError(f'Unexpected Shikra weights: {unexpected}')
        vision = root / 'clip-vit-large-patch14'
        self.vision = CLIPVisionModel.from_pretrained(vision, dtype=torch.float16).to(self.device).eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(vision)
        self.projector = torch.nn.Linear(config.mm_hidden_size, config.hidden_size).to(self.device, torch.float16)
        index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        state = {}
        for name in ('weight', 'bias'):
            key = 'model.mm_projector.' + name
            with safe_open(root/index[key], framework='pt', device='cpu') as source:
                state[name] = source.get_tensor(key)
        self.projector.load_state_dict(state, strict=True)
        self.model = VisualPrefixModel(lm, self.tokenizer.convert_tokens_to_ids('<im_patch>')).eval()
        self.processor = PrefixProcessor(self)
        self.grid = (self.vision.config.image_size // self.vision.config.patch_size,) * 2
        self.cfg['num_visual_tokens'] = self.grid[0]*self.grid[1]

    def _format_prompt(self, raw_prompt):
        return ('A chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                'USER: <im_start><ImageHere><im_end>\n' + raw_prompt + ' ASSISTANT:')

    @torch.no_grad()
    def encode_image(self, image):
        # Released Shikra eval applies Expand2square with a white background.
        width, height = image.size
        square = Image.new('RGB', (max(width, height),)*2, (255,255,255))
        square.paste(image, ((square.width-width)//2, (square.height-height)//2))
        image = square
        pixels = self.image_processor(images=image, return_tensors='pt').pixel_values.to(self.device, torch.float16)
        hidden = self.vision(pixels, output_hidden_states=True).hidden_states[-2][:, 1:]
        return self.projector(hidden)

    def _visual_grid_for_output(self, inputs, visual_start, visual_end):
        return self.grid
