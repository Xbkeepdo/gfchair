import unittest
from types import SimpleNamespace

import torch

from features.ffn_target_consequence import path_consequences, signed_totals, q_features, detach_tree, native_suffix, fp32_suffix
from features.ffn_visual_path_attribution import scalar_path_attribution


class TargetConsequenceTests(unittest.TestCase):
    def test_full_caption_capture_keeps_future_tokens_out_of_query(self):
        from unittest.mock import patch
        from transformers import LlamaConfig,LlamaForCausalLM
        from scripts.run_ffn_target_consequence import capture_image
        cfg=LlamaConfig(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                        num_attention_heads=4,num_key_value_heads=2)
        cfg._attn_implementation='eager'
        model=LlamaForCausalLM(cfg).eval().requires_grad_(False)
        wrapper=SimpleNamespace(model=model,cfg={})
        targets=[dict(target_key='1:0',response_index=0),dict(target_key='1:2',response_index=2)]
        inputs={'input_ids':torch.tensor([[2,3,4,5,6,7,8]])}
        with patch('scripts.run_ffn_target_consequence.prepare_inputs',return_value=(inputs,6,1,3,[1,2])):
            first=capture_image(wrapper,'qwen2_5_vl_7b',None,[6,7,8],targets,'prompt')
            inputs['input_ids'][0,4:]=torch.tensor([15,16,17])
            second=capture_image(wrapper,'qwen2_5_vl_7b',None,[6,7,8],targets,'prompt')
        self.assertEqual(first['positions'],[3,5])
        for a,b in zip(first['captures'],second['captures']):
            torch.testing.assert_close(a['h_mid'][:,:4],b['h_mid'][:,:4])
        for a,b in zip(first['directions'],second['directions']):
            self.assertEqual(a['a_tokens'].shape,(2,2,16))
            self.assertEqual(a['a_tokens'].device.type,'cpu')
            torch.testing.assert_close(a['a_tokens'][:,0],b['a_tokens'][:,0])

    def test_canonical_json_resume_is_read_only_and_rejects_real_change(self):
        from pathlib import Path
        import tempfile
        from scripts.run_cqb_workflow import canonical_json
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json'
            canonical_json({'scores':('logit','margin'),'hidden_sizes':(128,64,32)},path)
            before=(path.stat().st_mtime_ns,path.read_bytes())
            canonical_json({'scores':['logit','margin'],'hidden_sizes':(128,64,32)},path)
            self.assertEqual(before,(path.stat().st_mtime_ns,path.read_bytes()))
            with self.assertRaises(ValueError):canonical_json({'scores':['different']},path)

    def test_causal_mask_and_real_llama_cache_parity(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from features.ffn_target_consequence import fp32_cached_suffix,_PrefixKV
        from models.dgst_capture import run_forward_with_dgst_captures
        from scripts.run_ffn_target_consequence import causal_decoder_kwargs,decoder_calls
        mask=causal_decoder_kwargs(torch.zeros(1,5,4),{'attention_mask':None})['attention_mask']
        self.assertTrue((mask[0,0,:4,4]<0).all())
        self.assertTrue((mask[0,0,4]==0).all())
        cfg=LlamaConfig(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=3,
                        num_attention_heads=4,num_key_value_heads=2)
        cfg._attn_implementation='eager'
        model=LlamaForCausalLM(cfg).eval().requires_grad_(False)
        layers=model.model.layers
        with decoder_calls(layers) as calls:
            _,captures=run_forward_with_dgst_captures(model,input_ids=torch.tensor([[2,3,4,5,6]]),
                attention_query_positions=[4],attention_query_chunk_size=2,output_hidden_states=False)
        args=dict(model=model,layers=layers,captures=captures,calls=calls,layer_index=0,
                  prediction_position=4,target_id=2,competitor_id=3)
        full,cached=fp32_suffix(**args),fp32_cached_suffix(**args)
        for offset in (0.,.1,-.2):
            a=(captures[0]['o_ffn'][0,4]+offset).clone().requires_grad_()
            b=a.detach().clone().requires_grad_()
            va,vb=full(a),cached(b)
            torch.testing.assert_close(va,vb,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(torch.autograd.grad(va.sum(),a)[0],torch.autograd.grad(vb.sum(),b)[0],atol=2e-5,rtol=2e-5)
        handles=[]
        def strict_mask(_module,args,kwargs):
            self.assertEqual(args[0].shape[0],kwargs['attention_mask'].shape[0])
        for layer in layers:
            handles.append(layer.register_forward_pre_hook(strict_mask,with_kwargs=True))
        try:
            batch=captures[0]['o_ffn'][0,4].repeat(4,1).requires_grad_()
            values=cached(batch)
            self.assertEqual(values.shape,(4,3))
            self.assertTrue(torch.isfinite(torch.autograd.grad(values.sum(),batch)[0]).all())
        finally:
            for handle in handles:handle.remove()
        slot=_PrefixKV((torch.zeros(1,2,3,4),torch.zeros(1,2,3,4)))
        for _ in range(2):
            out=slot.update(torch.ones(1,2,1,4),torch.ones(1,2,1,4),0)
            self.assertEqual(out[0].shape[-2],4)
            self.assertEqual(slot.states[0].shape[-2],3)

    def test_suffix_checkpoint_gradients_and_dtype_preservation(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.b = torch.nn.Linear(4,4), torch.nn.Linear(4,4)
            def parts(self,x):
                middle=x+torch.tanh(self.a(x.cumsum(1)))
                return middle,torch.tanh(self.b(middle))
            def forward(self,x,scale=1.):
                middle,write=self.parts(x)
                return (middle+write*scale,)
        torch.manual_seed(12)
        layers=torch.nn.ModuleList([Block() for _ in range(3)])
        norm,head=torch.nn.LayerNorm(4),torch.nn.Linear(4,7)
        model=SimpleNamespace(model=SimpleNamespace(layers=layers,norm=norm),get_output_embeddings=lambda:head)
        for module in (layers,norm,head):module.requires_grad_(False)
        hidden=torch.randn(1,5,4);captures=[]
        for layer in layers:
            middle,write=layer.parts(hidden)
            captures.append(dict(h_prev=hidden.detach(),h_mid=middle.detach(),o_ffn=write.detach()))
            hidden=middle+write
        calls=[((),{'scale':1.}) for _ in layers]
        reference=None
        for factory in (native_suffix,fp32_suffix):
            for use_checkpoint in (False,True):
                callback=factory(model=model,layers=layers,captures=captures,calls=calls,layer_index=0,
                                 prediction_position=4,target_id=2,competitor_id=3,use_checkpoint=use_checkpoint)
                leaf=captures[0]['o_ffn'][0,4].clone().requires_grad_()
                values=callback(leaf)
                gradients=torch.stack([torch.autograd.grad(v,leaf,retain_graph=i<2)[0] for i,v in enumerate(values)])
                if reference is None:reference=(values,gradients)
                torch.testing.assert_close(values,reference[0],atol=2e-6,rtol=2e-6)
                torch.testing.assert_close(gradients,reference[1],atol=2e-6,rtol=2e-6)
        for module in (layers,norm,head):
            self.assertTrue(all(p.dtype==torch.float32 and not p.requires_grad for p in module.parameters()))

    def test_feature_group_order_and_no_fitted_transform(self):
        import numpy as np
        from scripts.analyze_ffn_target_consequence import feature_groups
        raw={name:np.full((3,2),v,dtype=np.float32) for name,v in
             (('AE',1),('S',2),('kappa_vec',.5),('Q_positive',3),('Q_negative',4),('B_Q',.2))}
        for score in ('logit','margin','log_probability'):
            raw['C_'+score+'_positive']=np.full((3,2),5.,dtype=np.float32)
            raw['C_'+score+'_negative']=np.full((3,2),6.,dtype=np.float32)
        groups=feature_groups(raw,True)
        self.assertEqual(len(groups),20)
        self.assertEqual(groups['F_C_logit_Q_B_Q'].shape,(3,14))
        np.testing.assert_allclose(groups['Q'][:,:2],np.log1p(3))
        np.testing.assert_allclose(groups['Q'][:,2:],np.log1p(4))
        np.testing.assert_array_equal(groups['F_K_B_Q'][:,-2:],raw['B_Q'])
        changed={k:v.copy() for k,v in raw.items()};changed['S'][2]*=1000
        np.testing.assert_array_equal(feature_groups(changed,True)['F'][:2],groups['F'][:2])

    def test_linear_and_nonlinear_scalar_oracle(self):
        torch.manual_seed(43)
        z, writes = torch.randn(4, dtype=torch.float64), torch.randn(7, 4, dtype=torch.float64) * .1
        for function in (lambda x: x * 2, torch.tanh):
            def scores(x):
                return torch.stack((x.sum(), x[0] - x[1], torch.log_softmax(x, -1)[0]))
            a = path_consequences(ffn_map=function, score_from_ffn=scores, z=z, writes=writes, integration_points=16, token_chunk=2)
            b = path_consequences(ffn_map=function, score_from_ffn=scores, z=z, writes=writes, integration_points=16, token_chunk=7)
            torch.testing.assert_close(a['C_m'], b['C_m'])
            torch.testing.assert_close(a['signed_sum'], a['finite_score_effect'], atol=1e-10, rtol=1e-10)
            for i in range(3):
                oracle = scalar_path_attribution(ffn_map=function, score_from_ffn_output=lambda x: scores(x)[i],
                                                z=z, writes=writes, method='gauss_legendre', integration_points=16)
                torch.testing.assert_close(a['C_m'][i], oracle.token_scores)

    def test_batched_nodes_and_resume_validation(self):
        import copy
        from scripts.run_ffn_target_consequence import validate_shard
        torch.manual_seed(43)
        z,writes=torch.randn(4),torch.randn(7,4)*.1
        def scores(x):
            return torch.stack((x.sum(-1),x[...,0]-x[...,1],torch.log_softmax(x,-1)[...,0]),dim=-1)
        a=path_consequences(ffn_map=torch.tanh,score_from_ffn=scores,z=z,writes=writes,node_batch=1)
        b=path_consequences(ffn_map=torch.tanh,score_from_ffn=scores,z=z,writes=writes,node_batch=4)
        torch.testing.assert_close(a['C_m'],b['C_m'],atol=1e-6,rtol=1e-6)
        parent={'positions':[dict(target_key='1:0',target_token_id=2,response_index=0,path_signed_q=torch.zeros(1,7))]}
        row=dict(target_key='1:0',target_token_id=2,response_index=0,layers=[1],score_names=['logit','margin','log_probability'],
                 C_m=a['C_m'].unsqueeze(1),layer_statistics=[dict(layer=1,signed_sum=a['signed_sum'])])
        payload=dict(fingerprint='fixed',processed_image=True,positions=[row])
        validate_shard(payload,'fixed',parent,[1])
        with self.assertRaises(ValueError):validate_shard(payload,'wrong',parent,[1])
        invalid=copy.deepcopy(payload);invalid['positions'].append(copy.deepcopy(row))
        with self.assertRaises(ValueError):validate_shard(invalid,'fixed',parent,[1])
        invalid=copy.deepcopy(payload);invalid['positions'][0]['C_m'][0,0,0]=float('nan')
        with self.assertRaises(ValueError):validate_shard(invalid,'fixed',parent,[1])

    def test_zero_and_signed_features(self):
        z = torch.ones(4)
        out = path_consequences(ffn_map=torch.tanh, score_from_ffn=lambda x: torch.stack((x.sum(), x[0], x[1])),
                                z=z, writes=torch.zeros(7, 4))
        self.assertTrue(out['score_effect_degenerate'].all())
        self.assertTrue(torch.isnan(out['closure_relative_error']).all())
        self.assertEqual(out['C_m'].abs().sum().item(), 0)
        q = torch.tensor([[.8, .3, -.4, .1], [0, 0, 0, 0.]])
        features = q_features(q, torch.tensor([2., 0.]), torch.tensor([False, True]))
        torch.testing.assert_close(features['Q_positive'], torch.tensor([1.2, 0.]))
        torch.testing.assert_close(features['Q_negative'], torch.tensor([.4, 0.]))
        torch.testing.assert_close(features['B_Q'], torch.tensor([.2, 0.]))
        self.assertTrue(features['Q_degenerate'][1])
        with self.assertRaises(ValueError):
            signed_totals(torch.tensor([float('nan')]))
        with self.assertRaises(TypeError):
            detach_tree(object())


if __name__ == '__main__':
    unittest.main()
