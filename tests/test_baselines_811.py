import unittest
from scripts.train_baselines_811 import align_records


class AlignmentTests(unittest.TestCase):
    def record(self,label):
        return dict(image_id=1,response_token_idx=2,target_token_id=3,label=label,token_str='dog',metadata=dict(svar_protocol='controlled'))

    def mention(self,label,index):
        return dict(image_id=1,response_index=2,target_token_id=3,label=label,word='dog',mention_id=f'1:{index}')

    def test_conflicting_labels_and_duplicate_mentions_are_preserved(self):
        records=[self.record(1),self.record(0),self.record(1)]
        mentions=[self.mention(0,0),self.mention(1,1),self.mention(1,2)]
        self.assertEqual([r['label'] for r in align_records(records,mentions)],[0,1,1])

    def test_missing_and_alternate_samples_fail(self):
        with self.assertRaisesRegex(ValueError,'Missing'):
            align_records([self.record(1)],[self.mention(1,0),self.mention(1,1)])
        record=self.record(1);record['metadata']['svar_protocol']='official'
        with self.assertRaisesRegex(ValueError,'controlled'):
            align_records([record],[self.mention(1,0)])


if __name__=='__main__':unittest.main()
