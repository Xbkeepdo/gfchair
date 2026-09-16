"""First-token causal positions for the two visual-prefix ENDAC models."""

import unittest

import torch
from transformers import BatchFeature, LlamaConfig, LlamaForCausalLM

from models.visual_prefix_wrapper import VisualPrefixModel
from scripts.extract_endac_811 import _capture


class TinyPrefix:
    device = "cpu"
    cfg = {"dgst_attention_query_chunk_size": 8}
    grid = (4, 8)

    def __init__(self):
        config = LlamaConfig(
            vocab_size=40, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        )
        config._attn_implementation = "eager"
        self.model = VisualPrefixModel(LlamaForCausalLM(config).half().eval(), -200)

    def resolve_prompt(self, prompt):
        return prompt

    def _format_prompt(self, prompt):
        return prompt

    def _image_token_id(self):
        return -200

    def _find_visual_token_range(self, inputs, image_token_id):
        positions = (inputs["input_ids"][0] == image_token_id).nonzero().flatten()
        return int(positions[0]), int(positions[-1]) + 1

    def processor(self, **_kwargs):
        ids = torch.tensor([[1] + [-200] * 32 + [3, 4]])
        return BatchFeature({
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "visual_embeds": torch.randn(1, 32, 16).half(),
        })


class EndacPrefixCaptureTest(unittest.TestCase):
    def test_first_response_token_uses_last_prompt_position(self):
        wrapper = TinyPrefix()
        response = [5, 6]
        targets = [{"response_index": 0}, {"response_index": 1}]
        captures, positions, start, end, grid, stats = _capture(
            wrapper, "minigpt4_7b", None, response, targets, "Describe this image."
        )
        self.assertEqual(positions, [34, 35])
        self.assertEqual((start, end, grid), (1, 33, []))
        self.assertEqual(len(captures), 2)
        self.assertEqual(captures[0]["attn_weights"].shape[-2], 2)
        self.assertEqual(stats["response_top1_probs"].shape[0], len(response))


if __name__ == "__main__":
    unittest.main()
