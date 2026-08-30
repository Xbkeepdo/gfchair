import argparse
import unittest

import torch
from PIL import Image

from scripts.run_jffn_second_round_logit_causal import prepare_inputs
from scripts.run_tc_fvpa_comprehensive import _base_args
from scripts.tc_fvpa_common import validate_common_args


class _ImageProcessor:
    merge_size = 1


class _Qwen2Processor:
    image_processor = _ImageProcessor()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert messages[0]["content"][0]["type"] == "image"
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered prompt"

    def __call__(self, *, text, images, return_tensors):
        assert text == ["rendered prompt"]
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[10, 99, 99, 11]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
            "position_ids": torch.arange(4).unsqueeze(0),
        }


class _QwenWrapper:
    device = "cpu"
    _image_token_id = 99
    processor = _Qwen2Processor()

    def resolve_prompt(self, prompt):
        return prompt

    def _prompt_inputs(self, image, prompt):
        return {
            "input_ids": torch.tensor([[20, 99, 99, 21]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
            "rope_deltas": torch.tensor([0]),
        }

    def _find_vision_token_range(self, input_ids):
        positions = torch.where(input_ids == self._image_token_id)[0].tolist()
        return positions[0], positions[-1] + 1

    def _resolve_visual_grid(self, inputs, visual_token_count):
        assert visual_token_count == 2
        return 1, 2


def _formal_args(model):
    return argparse.Namespace(
        model=model,
        device="cuda:0",
        devices="cuda:0,cuda:1",
        resume=True,
        shard_id=0,
        num_shards=2,
        output_dir="/tmp/tc-fvpa-test",
        seed=20260829,
        layers="8,16,24,32",
        target_scalars="log_probability,margin,logit",
        integration_points="1,4,8,16,32",
        dry_run=False,
        smoke=False,
        formal=True,
        config="configs/model_configs_inslen_official_target.yaml",
    )


class TCFVPAQwenAdapterTest(unittest.TestCase):
    def test_qwen2_exact_prefix_and_dynamic_visual_grid(self):
        inputs, prediction, visual_start, visual_end, grid = prepare_inputs(
            _QwenWrapper(),
            "qwen2_5_vl_7b",
            Image.new("RGB", (8, 8)),
            [31, 32],
            "describe",
        )
        self.assertEqual(inputs["input_ids"].tolist(), [[10, 99, 99, 11, 31, 32]])
        self.assertEqual(inputs["attention_mask"].tolist(), [[1, 1, 1, 1, 1, 1]])
        self.assertNotIn("position_ids", inputs)
        self.assertEqual((prediction, visual_start, visual_end, grid), (5, 1, 3, [1, 2]))

    def test_qwen3_exact_prefix_and_dynamic_visual_grid(self):
        inputs, prediction, visual_start, visual_end, grid = prepare_inputs(
            _QwenWrapper(),
            "qwen3_vl_8b",
            Image.new("RGB", (8, 8)),
            [41],
            "describe",
        )
        self.assertEqual(inputs["input_ids"].tolist(), [[20, 99, 99, 21, 41]])
        self.assertNotIn("rope_deltas", inputs)
        self.assertEqual((prediction, visual_start, visual_end, grid), (4, 1, 3, [1, 2]))

    def test_formal_qwen_layers_are_frozen_depth_quartiles(self):
        qwen2 = validate_common_args(_formal_args("qwen2_5_vl_7b"))
        qwen3 = validate_common_args(_formal_args("qwen3_vl_8b"))
        self.assertEqual(qwen2["layers"], [7, 14, 21, 28])
        self.assertEqual(qwen3["layers"], [9, 18, 27, 36])

        forwarded = _base_args(_formal_args("qwen2_5_vl_7b"), qwen2)
        layer_index = forwarded.index("--layers") + 1
        self.assertEqual(forwarded[layer_index], "7,14,21,28")


if __name__ == "__main__":
    unittest.main()
