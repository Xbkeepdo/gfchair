import tempfile
import unittest
from pathlib import Path

from scripts.train_cqb_subset import select_images, canonical_json


class SubsetTests(unittest.TestCase):
    def test_selection_preserves_image_split_and_is_order_independent(self):
        split = dict(train=list(range(700)),test=list(range(700,900)))
        selected = select_images(range(900),split)
        self.assertEqual(selected,select_images(reversed(range(900)),split))
        self.assertEqual(len(selected['train']),400)
        self.assertEqual(len(selected['test']),100)
        self.assertFalse(set(selected['train'])&set(selected['test']))
        self.assertTrue(set(selected['train'])<=set(split['train']))
        self.assertTrue(set(selected['test'])<=set(split['test']))
        with self.assertRaises(ValueError):
            select_images(range(750),split)
        with self.assertRaises(ValueError):
            select_images(range(900),dict(train=[1],test=[1]),1,1)

    def test_frozen_manifest_resume_rejects_changed_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'cohort.json'
            value = dict(train=[1,2],test=[3])
            canonical_json(value,p)
            before = (p.read_bytes(),p.stat().st_mtime_ns)
            canonical_json(value,p)
            self.assertEqual(before,(p.read_bytes(),p.stat().st_mtime_ns))
            with self.assertRaises(ValueError):
                canonical_json(dict(train=[1,3],test=[2]),p)
