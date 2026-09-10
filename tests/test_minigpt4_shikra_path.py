import unittest

import numpy as np
import torch

from scripts.run_minigpt4_shikra_path import build_groups
from utils.config_utils import load_config


class PathExperimentTests(unittest.TestCase):
    def test_matrix_groups_share_sources_and_use_train_only_scales(self):
        positions, mentions = {}, []
        torch.manual_seed(12)
        for image in range(3):
            components = torch.randn(2,4,5)
            endpoint = components.sum(1)
            norms = components.norm(dim=-1)
            net = endpoint.norm(dim=-1)
            q = (components * (endpoint/net[:,None])[:,None,:]).sum(-1)
            row = dict(path_signed_q=q,ffn_path_gross=norms,net_strength=net,
                net_degenerate=torch.zeros(2,dtype=torch.bool),attention_evidence=torch.softmax(torch.randn(2,4),-1),
                S=norms.sum(-1),N_vec=net,N_end=net,kappa_vec=net/norms.sum(-1),kappa_end=net/norms.sum(-1))
            for name in ('AE','I','R_cos','D_EW','D_WF','D_EF'):
                row[name]=torch.ones(2)*(image+1)
            key=f'{image}:0'
            positions[key]=row
            mentions.append(dict(image_id=image,target_key=key,label=image%2))
        groups, scales = build_groups(positions,mentions,[0,1])
        self.assertEqual(len(groups),36)
        self.assertTrue(all(v.shape[0]==3 and np.isfinite(v).all() for v in groups.values()))
        np.testing.assert_array_equal(groups['F'],groups['AE+log1p(S)'])
        np.testing.assert_array_equal(groups['J_cosT'],groups['endpoint_cosine_js'])
        np.testing.assert_allclose(groups['J_T']+groups['J_E'],groups['J_cosT'],atol=1e-7)
        positions['2:0']['S'] *= 1000
        _, changed_scales=build_groups(positions,mentions,[0,1])
        for key in scales:
            np.testing.assert_array_equal(scales[key],changed_scales[key])

    def test_exact_baseline_configuration(self):
        config=load_config('configs/model_configs_minigpt4_shikra_path.yaml')
        wanted=['svar','projectaway','metatoken']
        self.assertEqual(config['feature_extraction']['baseline']['methods'],wanted)
        self.assertEqual(config['training']['baseline']['methods'],wanted)
        self.assertEqual(config['training']['feature_sets']['ads_cgc'],['ads+cgc'])
        self.assertEqual(config['dataset']['num_images'],4000)


if __name__=='__main__':
    unittest.main()
