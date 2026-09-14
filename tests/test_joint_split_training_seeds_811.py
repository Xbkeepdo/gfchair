import unittest

import numpy as np

from scripts.evaluate_joint_split_training_seeds_811 import MODELS, image_split, masks_for


class JointSeedSplitTests(unittest.TestCase):
    def test_full_image_split_is_deterministic_and_disjoint(self):
        for model in MODELS:
            first = image_split(model, 43)
            self.assertEqual(first, image_split(model, 43))
            self.assertNotEqual(first, image_split(model, 44))
            self.assertEqual([len(first[name]) for name in ('train','validation','test')], [3200,400,400])
            sets = {name:set(values) for name,values in first.items()}
            self.assertFalse(sets['train'] & sets['validation'])
            self.assertFalse(sets['train'] & sets['test'])
            self.assertFalse(sets['validation'] & sets['test'])
            self.assertEqual(len(set.union(*sets.values())), 4000)

    def test_mention_masks_preserve_duplicates_and_partition_rows(self):
        split = image_split(MODELS[0], 43)
        mentions = [{'image_id': image_id} for name in ('train','validation','test')
                    for image_id in split[name][:2] for _ in range(3)]
        masks = masks_for(mentions, split)
        np.testing.assert_array_equal(sum(value.astype(np.int8) for value in masks.values()),
                                      np.ones(len(mentions), dtype=np.int8))
        self.assertEqual([int(masks[name].sum()) for name in ('train','validation','test')], [6,6,6])


if __name__ == '__main__':
    unittest.main()
