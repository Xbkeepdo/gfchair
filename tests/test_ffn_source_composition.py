import unittest
import math
import numpy as np
import torch
from torch import nn

from features.ffn_source_composition import compose_ffn, norm_sources, composition_q
from scripts.run_ffn_source_composition import build_groups


class RMS(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.7,1.2,1.8],dtype=torch.float64))
        self.variance_epsilon = eps

    def forward(self,x):
        return self.weight*x*torch.rsqrt(x.square().mean(-1,keepdim=True)+self.variance_epsilon)


class Gated(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Linear(3,5,dtype=torch.float64)
        self.up=nn.Linear(3,5,dtype=torch.float64)
        self.down=nn.Linear(5,3,dtype=torch.float64)
        self.gate_proj,self.up_proj,self.down_proj=self.gate,self.up,self.down
        self.act_fn=torch.nn.functional.silu

    def forward(self,x):
        return self.down(torch.nn.functional.silu(self.gate(x))*self.up(x))


class CompositionTest(unittest.TestCase):
    def test_sources_bias_and_full_gated_jvp(self):
        torch.manual_seed(9)
        ffn=Gated().requires_grad_(False)
        sources=torch.randn(6,2,3,dtype=torch.float64)*.2
        z=sources.sum(0)
        for eps in (1e-6,1e-5):
            norm=RMS(eps).requires_grad_(False)
            directions=norm_sources(norm,z,sources)
            torch.testing.assert_close(directions.sum(0),norm(z))
            result=compose_ffn(norm,ffn,z,sources,(1,3),16,2,True)
            fast=compose_ffn(norm,ffn,z,sources,(1,3),16,2,True,factored=True)
            torch.testing.assert_close(result['vectors']['c'],fast['vectors']['c'],atol=1e-12,rtol=1e-12)
            vectors=result['vectors']
            q,length,bad=composition_q(norm,ffn,z,sources,k=16)
            delta=vectors['output']-vectors['baseline']
            expected=torch.einsum('mtd,td->tm',vectors['c'],delta/delta.norm(dim=-1)[:,None])
            torch.testing.assert_close(q,expected,atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(q.sum(-1),length,atol=1e-10,rtol=1e-10)
            self.assertFalse(bool(bad.any()))
            torch.testing.assert_close(vectors['c'].sum(0)+vectors['baseline'],ffn(norm(z)),atol=1e-10,rtol=1e-10)
            self.assertLess(float(result['closure_relative'].max()),1e-9)
            # Independent reverse-mode Jacobian includes BOTH SwiGLU branches.
            point=norm(z)[0]*.43
            direction=directions[1,0]
            jvp=torch.func.jvp(ffn,(point,),(direction,))[1]
            reverse=torch.autograd.functional.jacobian(ffn,point)@direction
            torch.testing.assert_close(jvp,reverse)
            torch.testing.assert_close(result['other_token_mag'].sum(-1),
                                      vectors['c'][[0,3]].norm(dim=-1).sum(0))

    def test_reconstruction_error_is_not_hidden(self):
        norm=RMS(1e-5).requires_grad_(False)
        z=torch.ones(1,3,dtype=torch.float64)
        sources=torch.zeros(4,1,3,dtype=torch.float64)
        result=compose_ffn(norm,nn.Identity(),z,sources,(0,2),4)
        self.assertEqual(float(result['closure_relative'][0]),1.)
        torch.testing.assert_close(result['p_c'],torch.full((1,2),.5,dtype=torch.float64))

    def test_strength_groups(self):
        a=np.ones((2,3),dtype=np.float32)
        groups=build_groups(a,2*a,3*a,4*a,5*a,6*a,7*a,8*a)
        self.assertEqual(len(groups),10)
        np.testing.assert_allclose(groups['F'],np.concatenate((a,np.log1p(3*a)),1))
        np.testing.assert_allclose(groups['F_C'],np.concatenate((a,np.log1p(4*a)),1))
        np.testing.assert_allclose(groups['B_JS_TC'][:,-3:],6*a)

    def test_signed_q_temperature_point_two(self):
        from scripts.run_ffn_qe_qc_tau02 import q_js
        p=np.array([1/(1+math.exp(2)),1/(1+math.exp(-2))])
        target=np.array([.8,.2]);mid=(p+target)/2
        expected=.5*np.sum(p*np.log(p/mid)+target*np.log(target/mid))
        value,stats=q_js(torch.tensor([[-.2,.2]],dtype=torch.float64),target[None])
        self.assertAlmostEqual(float(value[0]),float(expected),places=7)
        self.assertAlmostEqual(float(stats['p_max'][0]),float(p.max()),places=12)


if __name__=='__main__': unittest.main()
