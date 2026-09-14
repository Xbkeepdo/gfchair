import unittest
import numpy as np
import torch

from scripts.extract_region_ae import region_ae
from scripts.train_all_attention_ae_strength import build_groups


class RegionFeatureTests(unittest.TestCase):
    def test_region_gate_and_attention_ignore_outside(self):
        a=torch.tensor([.1,.3,.6]);v=torch.tensor([-1.,2.,100.]);mask=torch.tensor([True,True,False])
        expected=region_ae(a,v,mask)
        a[-1]=1000;v[-1]=-1000
        self.assertEqual(float(expected),float(region_ae(a,v,mask)))
        torch.testing.assert_close(region_ae(a*2,v,mask),expected)

    def test_empty_generation_is_undefined_before_detector(self):
        self.assertTrue(torch.isnan(region_ae(torch.tensor([1.]),torch.tensor([2.]),torch.tensor([False]))))
        groups=build_groups({'visual':np.array([[.3]]),'visual_prompt_sum':np.array([[.4]]),
            'generation':np.array([[np.nan]])},{'visual':np.array([[1.]]),'prompt':np.array([[2.]]),'generation':np.array([[0.]])})
        np.testing.assert_array_equal(groups['generation'],[[0.,0.]])

    def test_vp_gross_then_log_and_complete_blocks(self):
        ae={'visual':np.array([[.3,.4]]),'visual_prompt_sum':np.array([[.6,.7]]),'generation':np.array([[.2,.1]])}
        s={'visual':np.array([[1.,2.]]),'prompt':np.array([[3.,4.]]),'generation':np.array([[5.,6.]])}
        groups=build_groups(ae,s)
        np.testing.assert_allclose(groups['visual_prompt_sum'],[[.6,.7,np.log(5),np.log(7)]],rtol=1e-7)
        np.testing.assert_array_equal(groups['vp_generation'],np.concatenate([groups['visual_prompt_sum'],groups['generation']],1))
        np.testing.assert_array_equal(groups['ae_only_vp_generation'],np.concatenate([ae['visual_prompt_sum'],ae['generation']],1).astype(np.float32))


if __name__=='__main__':unittest.main()
