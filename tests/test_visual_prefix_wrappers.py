"""Real tiny-Llama causal checks plus query-space mapping invariants."""
import unittest
from types import SimpleNamespace

import torch
from transformers import BatchFeature, LlamaConfig, LlamaForCausalLM

from models.base_wrapper import ExtractionRequirements
from models.visual_prefix_wrapper import VisualPrefixModel, VisualPrefixWrapper
from models.minigpt4_wrapper import query_patch_mapping
from utils.inslen_targeting import inslen_model_branch, inslen_resolver_compatibility_note


class TinyWrapper(VisualPrefixWrapper):
    def _load_model(self):
        torch.manual_seed(7)
        cfg = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
        cfg._attn_implementation = 'eager'
        self.model = VisualPrefixModel(LlamaForCausalLM(cfg).half().eval(), -200)
        self.tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids))
        visual = torch.randn(1,4,16).half()
        self.processor = lambda **kw: BatchFeature(dict(input_ids=torch.tensor([[1,-200,-200,-200,-200,3,4]]),
            attention_mask=torch.ones(1,7,dtype=torch.long),visual_embeds=visual))

    def _format_prompt(self, raw_prompt):
        return raw_prompt

    def _visual_grid_for_output(self, *args):
        return (2,2)


class PrefixTests(unittest.TestCase):
    def test_causal_rows_and_actual_target_ids(self):
        wrapper = TinyWrapper(dict(num_visual_tokens=4),device='cpu')
        req = ExtractionRequirements(dgst_capture=False)
        ids = [5,6,7]
        batch = wrapper.extract_token_features_batch(None,ids,[0,2],requirements=req)
        changed = wrapper.extract_token_features_batch(None,[5,6,9],[0,2],requirements=req)
        for offset,index in enumerate((0,2)):
            single = wrapper.extract_token_features(None,ids[:index],index,target_token_id=ids[index],requirements=req)
            self.assertEqual(batch[offset].baseline_capture['prediction_position'],6+index)
            torch.testing.assert_close(batch[offset].token_logits,single.token_logits,rtol=0,atol=2e-3)
            torch.testing.assert_close(batch[offset].token_logits,changed[offset].token_logits,rtol=0,atol=0)
            torch.testing.assert_close(batch[offset].text_to_patch_attn,single.text_to_patch_attn,rtol=0,atol=2e-4)
        with self.assertRaises(ValueError):
            wrapper.extract_token_features_batch(None,ids,[0],target_token_ids=[9])

    def test_head_mean_is_unrenormalized_fp32(self):
        wrapper = TinyWrapper(dict(num_visual_tokens=4),device='cpu')
        per_head = wrapper.extract_token_features_batch(None,[5],[0],requirements=ExtractionRequirements(dgst_capture=False))[0]
        mean = wrapper.extract_token_features_batch(None,[5],[0],requirements=ExtractionRequirements(attention='head_mean',dgst_capture=False))[0]
        self.assertEqual(mean.text_to_patch_attn.dtype,torch.float32)
        torch.testing.assert_close(mean.text_to_patch_attn[:,0],per_head.text_to_patch_attn.float().mean(1),rtol=0,atol=0)
        self.assertTrue((mean.text_to_patch_attn.sum(-1)<1).all())

    def test_mapping_mass_and_query_permutation(self):
        cross = torch.rand(1,2,3,5)
        mapping = query_patch_mapping(cross,(2,2))
        attention = torch.rand(2,3)
        mapped = attention@mapping
        torch.testing.assert_close(mapped.sum(-1),attention.sum(-1))
        perm=torch.tensor([2,0,1])
        torch.testing.assert_close(mapped,attention[:,perm]@mapping[perm])
        with self.assertRaises(ValueError):query_patch_mapping(torch.zeros_like(cross),(2,2))
        with self.assertRaises(ValueError):query_patch_mapping(cross,(3,3))

    def test_explicit_inslen_adaptation(self):
        for model in ('minigpt4_7b','shikra_7b'):
            self.assertEqual(inslen_model_branch(model),'llava_default_surface')
            self.assertIn('adaptation',inslen_resolver_compatibility_note(model))


if __name__=='__main__':
    unittest.main()
