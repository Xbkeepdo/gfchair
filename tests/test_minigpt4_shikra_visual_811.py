import unittest

from scripts import search_minigpt4_shikra_visual_811 as study


class Visual811ExtensionTests(unittest.TestCase):
    def test_split_preserves_train_and_splits_old_test(self):
        old = {'train':list(range(3200)), 'val':[], 'test':list(range(3200,4000))}
        first = study.split_images(old)
        self.assertEqual(first, study.split_images(old))
        self.assertEqual(first['train'], old['train'])
        self.assertEqual([len(first[k]) for k in first], [3200,400,400])
        self.assertEqual(set(first['validation'])|set(first['test']), set(old['test']))

    def test_duplicate_baseline_rows_align_as_a_multiset(self):
        mention = dict(image_id=1,response_index=2,target_token_id=3,label=1,word='cat')
        record = dict(image_id=1,response_token_idx=2,target_token_id=3,label=1,token_str='cat',
                      metadata={'svar_protocol':'controlled'})
        ordered = study.align_baselines([dict(record, marker=2),dict(record, marker=1)],
                                        [dict(mention),dict(mention)])
        self.assertEqual([r['marker'] for r in ordered], [2,1])
        with self.assertRaises(ValueError):
            study.align_baselines([record], [mention,mention])


if __name__ == '__main__':
    unittest.main()
