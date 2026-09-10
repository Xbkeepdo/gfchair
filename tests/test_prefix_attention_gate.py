import unittest
import torch
from scripts.extract_prefix_attention_gate import gate,prefix_arrays
from features.dgst_t import _gaussian_mad_gate

class PrefixGateTests(unittest.TestCase):
    def test_bfloat_rounding_is_not_renormalized(self):
        a=torch.full((2,3),1/3,dtype=torch.bfloat16)
        raw,_,_,_=prefix_arrays(a,torch.tensor([0.,1.,2.]),2,(0,2),'full',1e-6)
        torch.testing.assert_close(raw,a.float().mean(0))
        self.assertGreater(abs(float(raw.sum())-1),1e-3)
    def test_gate_scope(self):
        x=torch.tensor([-10.,0.,1.,2.,50.])
        full,_,_=gate(x,(1,3),'full');torch.testing.assert_close(full,_gaussian_mad_gate(x,epsilon=1e-6))
        visual,_,_=gate(x,(1,3),'visual');self.assertFalse(torch.equal(full,visual))
    def test_causal_support_and_raw_product(self):
        a=torch.tensor([[.2,.3,.5,0.],[.1,.4,.5,0.]])
        x=torch.tensor([0.,1.,2.,1000.])
        raw,weighted,_,_=prefix_arrays(a,x,2,(0,2),'full',1e-6)
        torch.testing.assert_close(raw,torch.tensor([.15,.35,.5]))
        torch.testing.assert_close(weighted,raw*_gaussian_mad_gate(x[:3],epsilon=1e-6))
        self.assertLess(float(weighted.sum()),1)
        a[0,-1]=.1
        with self.assertRaises(ValueError):prefix_arrays(a,x,2,(0,2),'full',1e-6)

if __name__=='__main__':unittest.main()
