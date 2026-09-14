from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from scripts import analyze_all_source_paths as study


class AllSourceAnalysisTests(unittest.TestCase):
    def test_fifteen_families_log_only_norms_and_preserve_signed_geometry(self):
        fields = set(study.F1_FIELDS+study.F2_FIELDS)
        fields.update(b+'_'+k for b in ('b1', 'b2') for k in study.F3_FIELDS+study.F4_FIELDS)
        arrays = {k: np.full((3, 2), 3.) for k in fields}
        arrays['prompt_group_gain'][:] = 7
        arrays['prompt_group_rotation'][:] = -.5
        arrays['generation_group_rotation'][0] = np.nan
        arrays['b1_residual_share'][:] = .7
        arrays['b1_cos_residual_visual'][:] = -.8
        arrays['b2_cos_residual_visual'][:] = .9
        mentions = [dict(response_index=i) for i in range(3)]
        groups = study.feature_groups(arrays, np.ones((3, 6)), mentions)
        self.assertEqual(len(groups), 15)
        self.assertEqual(groups['F1_raw'].shape, (3, 18))
        self.assertEqual(groups['F2'].shape, (3, 30))
        self.assertEqual(groups['b1_F3_raw'].shape, (3, 10))
        self.assertEqual(groups['b1_F4'].shape, (3, 14))
        self.assertEqual(groups['b1_F5'].shape, (3, 44))
        self.assertEqual(groups['b1_F6'].shape, (3, 50))
        np.testing.assert_allclose(groups['F1_log1p'][:, :4], np.log(4))
        np.testing.assert_array_equal(groups['F1_log1p'][:, 4:6], 7)
        np.testing.assert_array_equal(groups['F2'][:, :2], -.5)
        np.testing.assert_allclose(groups['b1_F3_log1p'][:, -2:], .7)
        np.testing.assert_allclose(groups['b1_F4'][:, :2], -.8)
        np.testing.assert_allclose(groups['b2_F4'][:, :2], .9)
        np.testing.assert_array_equal(groups['length'], [[0, 0], [1, 1], [2, 2]])
        self.assertTrue(all(np.isfinite(x).all() for x in groups.values()))
        arrays['visual_context_cos'][0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, 'Infinite'):
            study.feature_groups(arrays, np.ones((3, 6)), mentions)

    def test_weighted_metrics_match_repeated_mentions_with_ties(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        p = np.array([.2, .2, .8, .7, .8, .1])
        weights = np.array([[1, 1, 1, 1, 1, 1], [3, 3, 0, 0, 2, 2], [0, 1, 0, 2, 3, 0]])
        result = study.weighted_rank_metrics(y, p, weights)
        for i, w in enumerate(weights):
            indices = np.repeat(np.arange(len(y)), w)
            expected = [roc_auc_score(y[indices], p[indices]),
                        average_precision_score(1-y[indices], 1-p[indices])]
            np.testing.assert_allclose(result[i], expected, atol=1e-14)
        self.assertTrue(np.isnan(study.weighted_rank_metrics(y, p, [[1, 0, 0, 1, 0, 0]])).all())

    def test_baseline_alignment_rejects_reordered_mentions_even_same_labels(self):
        def mention(i):
            return dict(mention_id=str(i), target_key=str(i), image_id=i, response_index=1, target_token_id=3, label=1)
        ref = dict(ntrain=1, y=np.ones(2), mentions=[mention(1), mention(2)])
        study.aligned(ref, ref)
        with self.assertRaisesRegex(ValueError, 'mention order'):
            study.aligned(dict(ref, mentions=ref['mentions'][::-1]), ref)

    def test_formal_train_collection_rejects_incomplete_images_before_loading(self):
        ref = dict(ntrain=1, y=np.ones(2), mentions=[])
        with tempfile.TemporaryDirectory() as folder, patch.object(study, 'reference', return_value=ref), \
             patch.object(study, 'formal_split', return_value=({1}, {2})):
            with self.assertRaisesRegex(ValueError, 'Incomplete formal extraction'):
                study.collect('model', Path(folder))

    def test_subset_collect_preserves_mentions_and_rejects_mixed_protocol(self):
        mentions = [dict(mention_id=str(i), target_key='1:0' if i<2 else '2:0', image_id=1 if i<2 else 2,
                         response_index=0, target_token_id=9, label=i % 2) for i in range(3)]
        ref = dict(ntrain=2, y=np.array([0, 1, 0]), mentions=mentions)
        with tempfile.TemporaryDirectory() as folder, patch.object(study, 'reference', return_value=ref), \
             patch.object(study, 'formal_split', return_value=({1}, {2})):
            root = Path(folder)
            paths = root/'model/shards'
            paths.mkdir(parents=True)
            for image_id in (1, 2):
                local = [m for m in mentions if m['image_id'] == image_id]
                value = dict(image_id=image_id, complete=True, protocol_signature='fixed', sample_table=local,
                             positions=[dict(target_key=f'{image_id}:0', response_index=0, target_token_id=9,
                                             metrics=dict(metric=torch.tensor([image_id, 3.])))])
                torch.save(value, paths/f'image_{image_id:012d}.pt')
            arrays, ordered, _, signature = study.collect('model', root, [1, 2])
            self.assertEqual(ordered, mentions)
            self.assertEqual(signature, 'fixed')
            np.testing.assert_array_equal(arrays['metric'][:, 0], [1, 1, 2])
            value['protocol_signature'] = 'changed'
            torch.save(value, paths/'image_000000000002.pt')
            with self.assertRaisesRegex(ValueError, 'Mixed extraction'):
                study.collect('model', root, [1, 2])

    def test_mechanism_filters_conflicts_before_primary_and_paired_summaries(self):
        mentions = [dict(target_key='conflict', label=0, image_id=1),
                    dict(target_key='conflict', label=1, image_id=1),
                    dict(target_key='clean', label=1, image_id=2)]
        arrays = dict(metric=np.array([[1., 2.], [1., 2.], [3., 4.]]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'outputs/ffn_output_cosine_20260909/cohort500.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(split=dict(train=list(range(400)), test=list(range(400, 500))))))
            with patch.object(study, 'ROOT', root), patch.object(study, 'collect', return_value=(arrays, mentions, {}, 'fixed')), \
                 patch.object(study, 'plot_mechanism'), \
                 patch('scripts.analyze_ffn_input_geometry.summarize', return_value=([], [])) as summarize:
                study.mechanism('model', root/'new')
            args = summarize.call_args.args
            np.testing.assert_array_equal(args[1]['metric'], [[3., 4.]])
            self.assertEqual(len(args[2]), 1)
            self.assertFalse(args[2][0]['label_conflict'])


if __name__ == '__main__':
    unittest.main()
