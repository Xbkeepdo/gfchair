import unittest
import numpy as np
from scripts.plot_prefix_attention_regions import region_masks,region_sums

class RegionsTests(unittest.TestCase):
    def test_disjoint_bos_and_prefix(self):
        layout=dict(position_types=['prompt_text','prompt_text','visual','visual','prompt_text','response_text','response_text'],token_ids=[1,2,None,None,3,4,5],bos_token_id=1,explicit_bos_at_zero=True)
        masks=region_masks(layout,6);np.testing.assert_array_equal(masks.sum(1),[1,2,2,1])
        a=np.array([[.1,.2,.15,.05,.3,.2]])
        values,error=region_sums(dict(prefix_length=6,raw_attention=a,attention_x_gate=a*.5),layout)
        np.testing.assert_allclose(values[0,0],[.1,.2,.5,.2]);np.testing.assert_allclose(values[1],values[0]*.5);self.assertLess(error,1e-12)
        layout['explicit_bos_at_zero']=False
        np.testing.assert_array_equal(region_masks(layout,5).sum(1),[0,2,3,0])
    def test_reject_unknown_regions(self):
        with self.assertRaises(ValueError):region_masks(dict(position_types=['padding'],explicit_bos_at_zero=False),1)

if __name__=='__main__':unittest.main()
