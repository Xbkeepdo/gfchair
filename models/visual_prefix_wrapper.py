"""Shared Llama capture interface for fixed visual-prefix models."""
from types import SimpleNamespace

import torch
from transformers import BatchFeature, StoppingCriteria, StoppingCriteriaList

from models.base_wrapper import GenerationOutput
from models.llava_wrapper import LLaVAWrapper


class VisualPrefixModel(torch.nn.Module):
    def __init__(self, language_model, image_token_id):
        super().__init__()
        self.language_model = language_model
        self.config = SimpleNamespace(image_token_index=image_token_id,
                                      vocab_size=language_model.config.vocab_size)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def embeddings(self, input_ids, visual_embeds):
        mask = input_ids == self.config.image_token_index
        if input_ids.shape[0] != 1 or int(mask.sum()) != visual_embeds.shape[1]:
            raise ValueError('Expected one image and one placeholder per visual embedding')
        embeds = self.get_input_embeddings()(input_ids.masked_fill(mask, 0))
        embeds[mask] = visual_embeds.reshape(-1, embeds.shape[-1]).to(embeds)
        return embeds

    def forward(self, input_ids, visual_embeds, attention_mask=None, **kwargs):
        return self.language_model(inputs_embeds=self.embeddings(input_ids, visual_embeds),
                                   attention_mask=attention_mask, **kwargs)


class PrefixProcessor:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.tokenizer = wrapper.tokenizer

    def __call__(self, text, images, return_tensors='pt'):
        if text.count('<ImageHere>') != 1:
            raise ValueError('Expected exactly one image placeholder')
        before, after = text.split('<ImageHere>')
        first = self.tokenizer(before, add_special_tokens=True).input_ids
        last = self.tokenizer(after, add_special_tokens=False).input_ids
        visual = self.wrapper.encode_image(images)
        ids = first + [self.wrapper._image_token_id()] * visual.shape[1] + last
        return BatchFeature(dict(input_ids=torch.tensor([ids]),
                                 attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                 visual_embeds=visual))


class StopSeparator(StoppingCriteria):
    def __init__(self, tokenizer, separator):
        self.tokenizer, self.separator = tokenizer, separator

    def __call__(self, input_ids, scores, **kwargs):
        return self.separator in self.tokenizer.decode(input_ids[0, -8:], skip_special_tokens=False)


class VisualPrefixWrapper(LLaVAWrapper):
    """Reuse tested expanded-placeholder causal extraction, including MetaToken."""
    @property
    def model_label(self):
        return type(self).__name__

    @staticmethod
    def _extract_attention_features(*args, **kwargs):
        return tuple(t.float() for t in LLaVAWrapper._extract_attention_features(*args, **kwargs))

    @staticmethod
    def _extract_attention_features_at_position(*args, **kwargs):
        return tuple(t.float() for t in LLaVAWrapper._extract_attention_features_at_position(*args, **kwargs))

    @torch.no_grad()
    def generate(self, image, prompt=None):
        inputs = self.processor(text=self._format_prompt(self.resolve_prompt(prompt)), images=image)
        inputs = inputs.to(self.device, torch.float16)
        lm = self.model.language_model
        kwargs = {}
        if getattr(self, 'stop_separator', None):
            kwargs['stopping_criteria'] = StoppingCriteriaList([StopSeparator(self.tokenizer, self.stop_separator)])
        # inputs_embeds-only generation returns just response IDs in current HF.
        ids = lm.generate(inputs_embeds=self.model.embeddings(inputs['input_ids'], inputs['visual_embeds']),
                          attention_mask=inputs['attention_mask'], do_sample=False,
                          max_new_tokens=self.generation_max_new_tokens,
                          bos_token_id=self.tokenizer.bos_token_id,
                          eos_token_id=self.tokenizer.eos_token_id,
                          pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                          **kwargs)[0].tolist()
        if getattr(self, 'stop_separator', None):
            # Remove only the terminal separator's actual generated IDs.
            for start in range(max(0, len(ids)-8), len(ids)):
                if self.stop_separator in self.tokenizer.decode(ids[start:], skip_special_tokens=False):
                    if self.tokenizer.decode(ids[start:], skip_special_tokens=False).strip() == self.stop_separator:
                        ids = ids[:start]
                        break
        return GenerationOutput(-1, self.tokenizer.decode(ids, skip_special_tokens=True), ids,
                                [self.tokenizer.decode([i], skip_special_tokens=False) for i in ids])

    def extract_token_features_batch(self, *args, **kwargs):
        outputs = super().extract_token_features_batch(*args, **kwargs)
        for output in outputs:
            self.attach_spatial_attention(output)
        return outputs

    def extract_token_features(self, *args, **kwargs):
        output = super().extract_token_features(*args, **kwargs)
        self.attach_spatial_attention(output)
        return output

    def attach_spatial_attention(self, output):
        pass
