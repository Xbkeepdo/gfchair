import unittest
from types import SimpleNamespace

import torch

from features.ffn_target_consequence import path_consequences,fp32_cached_suffix
from features.ffn_target_consequence_fast import joint_score_adapter


class FastConsequenceTest(unittest.TestCase):
    def test_mixed_serial_joint_lineage_is_explicit(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts import run_cqb_optimized as optimized
        from features.tc_fvpa_artifacts import sha256_file
        model='qwen2_5_vl_7b'
        old=dict(target_key='1:0',C_m=torch.ones(3,2,4))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'gate.json').write_text('{}')
            snapshot=dict(gate_sha256=sha256_file(root/'gate.json'),models={model:dict(completed={},partial_targets={'1:0':optimized.row_digest(old)})})
            (root/'reference_snapshot.json').write_text(json.dumps(snapshot))
            with patch.object(optimized,'OUT',root),patch.object(optimized,'gate',return_value={'reference_gate_sha256':'reference'}),\
                 patch.object(optimized.reference,'validate_shard',side_effect=lambda p,*a,**k:p):
                check,metadata=optimized.lineage_checker(model)
                new=dict(target_key='1:1',C_m=torch.ones(3,2,4),execution_backend=metadata)
                check(dict(image_id=1,positions=[old,new]))
                with self.assertRaises(ValueError):check(dict(image_id=1,positions=[dict(old,C_m=torch.zeros(3,2,4))]))
                with self.assertRaises(ValueError):check(dict(image_id=1,positions=[dict(target_key='2:0',C_m=torch.ones(3,2,4))]))

    def test_values_and_first_order_path_parity(self):
        torch.manual_seed(4)
        for dtype in (torch.float32,torch.float64):
            z=torch.randn(5,dtype=dtype)
            writes=torch.randn(9,5,dtype=dtype)*.2
            def callback(x):
                logits=torch.sin(x)+x*x*.3
                return torch.stack((logits[...,1],logits[...,1]-logits[...,2],
                    torch.log_softmax(logits,-1)[...,1]),dim=-1)
            for batch in (1,4):
                reference=path_consequences(ffn_map=torch.tanh,score_from_ffn=callback,z=z,writes=writes,node_batch=batch)
                fast=path_consequences(ffn_map=torch.tanh,score_from_ffn=joint_score_adapter(callback),z=z,writes=writes,node_batch=batch)
                for key in ('C_m','signed_sum','node_scores','clean_scores','baseline_scores'):
                    torch.testing.assert_close(fast[key],reference[key],atol=2e-6,rtol=2e-6)
                zero=path_consequences(ffn_map=torch.tanh,score_from_ffn=joint_score_adapter(callback),z=z,writes=writes*0,node_batch=batch)
                self.assertEqual(zero['C_m'].abs().sum().item(),0.)

    def test_real_llama_checkpoint_batched_suffix_vjp(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from models.dgst_capture import run_forward_with_dgst_captures
        from scripts.run_ffn_target_consequence import decoder_calls
        cfg=LlamaConfig(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=3,
                        num_attention_heads=4,num_key_value_heads=2)
        cfg._attn_implementation='eager'
        model=LlamaForCausalLM(cfg).eval().requires_grad_(False)
        with decoder_calls(model.model.layers) as calls:
            _,captures=run_forward_with_dgst_captures(model,input_ids=torch.tensor([[2,3,4,5,6]]),
                attention_query_positions=[4],attention_query_chunk_size=2,output_hidden_states=False)
        callback=fp32_cached_suffix(model=model,layers=model.model.layers,captures=captures,calls=calls,
            layer_index=0,prediction_position=4,target_id=2,competitor_id=3)
        z=captures[0]['h_mid'][0,4];writes=torch.randn(7,16)*.01
        function=lambda x:model.model.layers[0].mlp(model.model.layers[0].post_attention_layernorm(x))
        a=path_consequences(ffn_map=function,score_from_ffn=callback,z=z,writes=writes,node_batch=4)
        b=path_consequences(ffn_map=function,score_from_ffn=joint_score_adapter(callback),z=z,writes=writes,node_batch=4)
        torch.testing.assert_close(a['C_m'],b['C_m'],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(a['node_scores'],b['node_scores'],atol=1e-7,rtol=1e-7)


if __name__=='__main__':unittest.main()
