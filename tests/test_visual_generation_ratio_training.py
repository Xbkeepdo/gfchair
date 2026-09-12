import unittest
import numpy as np
from scripts.train_visual_generation_ratio import build_groups

class RatioTrainingTests(unittest.TestCase):
    def test_both_families_and_no_log(self):
        groups=dict(attention_raw_visual=np.array([[.6,.2]]),attention_raw_generation=np.array([[.2,.4]]),
                    gated_raw_visual=np.array([[.3,.1]]),gated_raw_generation=np.array([[.05,.1]]))
        x=build_groups(groups);np.testing.assert_allclose(x['attention_ratio'],[[3,.5]]);np.testing.assert_allclose(x['gated_ratio'],[[6,1]])
        self.assertEqual(x['attention_ratio'].dtype,np.float32)
        groups['attention_raw_generation'][0,0]=0
        with self.assertRaises(ValueError):build_groups(groups)

if __name__=='__main__':unittest.main()
