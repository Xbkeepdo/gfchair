from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.watch_all_source_progress import MODELS, render, watch


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.now = datetime(2026, 9, 12, 20, 0, tzinfo=timezone.utc)

    def put(self, model, **values):
        path = self.root / 'outputs/ffn_all_source_paths_v1' / model / 'progress.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(values))
        return path

    def test_missing_and_running_have_three_lines_and_do_not_finish(self):
        self.put(MODELS['local'][0], stage='training', status='running', completed=5,
                 total=10, epoch=3, heartbeat=self.now.isoformat())
        text, finished = render(self.root, 'local', self.now)
        self.assertFalse(finished)
        self.assertEqual(len(text.splitlines()), 3)
        self.assertIn('[' + '█' * 10 + '░' * 10 + '] 5/10', text)
        self.assertIn('epoch 3', text)
        self.assertIn('准备中', text)
        self.assertNotIn('无更新', text)

    def test_old_heartbeat_and_missing_heartbeat_use_stale_status(self):
        self.put(MODELS['32678'][0], status='running', heartbeat='2026-09-12T19:00:00Z')
        path = self.put(MODELS['32678'][1], status='running')
        os.utime(path, (self.now.timestamp() - 300,) * 2)
        text, finished = render(self.root, '32678', self.now)
        self.assertEqual(text.count('无更新'), 2)
        self.assertNotIn('failed', text)
        self.assertFalse(finished)

    def test_terminal_states_remain_and_loop_exits_without_sleep(self):
        self.put(MODELS['local'][0], stage='done', status='completed', completed=10,
                 total=10, heartbeat='2020-01-01T00:00:00Z')
        self.put(MODELS['local'][1], stage='audit', status='failed', error='bad\ninput')
        with patch('scripts.watch_all_source_progress.time.sleep') as sleep:
            self.assertEqual(watch(self.root, 'local'), 0)
        sleep.assert_not_called()
        text = (self.root / 'EXPERIMENT_PROGRESS_LOCAL.md').read_text()
        self.assertIn('completed', text)
        self.assertIn('failed', text)
        self.assertIn('error: bad input', text)
        self.assertNotIn('无更新', text)
        self.assertEqual(len(text.splitlines()), 3)
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_corrupt_json_remains_nonterminal_and_once_writes_correct_machine(self):
        path = self.put(MODELS['32678'][0], status='running')
        path.write_text('{')
        text, finished = render(self.root, '32678', self.now)
        self.assertFalse(finished)
        self.assertIn('状态暂不可读', text)
        self.assertEqual(watch(self.root, '32678', once=True), 0)
        self.assertTrue((self.root / 'EXPERIMENT_PROGRESS_32678.md').is_file())
        self.assertFalse((self.root / 'EXPERIMENT_PROGRESS_LOCAL.md').exists())

    def test_lock_prevents_same_machine_writer_but_allows_other_machine(self):
        with (self.root / '.EXPERIMENT_PROGRESS_LOCAL.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(watch(self.root, 'local', once=True), 1)
            self.assertFalse((self.root / 'EXPERIMENT_PROGRESS_LOCAL.md').exists())
            self.assertEqual(watch(self.root, '32678', once=True), 0)

    def test_secondary_study_takes_over_after_primary_completion(self):
        secondary=self.root/'baseline'
        for model in MODELS['local']:
            self.put(model,stage='AE training',status='running',completed=3,total=9,heartbeat=self.now.isoformat())
            path=secondary/model/'progress.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(stage='baseline queued',status='waiting',heartbeat=self.now.isoformat())))
        text,finished=render(self.root,'local',self.now,secondary_progress_root=secondary)
        self.assertIn('AE training',text);self.assertNotIn('baseline queued',text);self.assertFalse(finished)
        for model in MODELS['local']:
            self.put(model,stage='AE done',status='completed',completed=9,total=9)
        text,finished=render(self.root,'local',self.now,secondary_progress_root=secondary)
        self.assertIn('baseline queued',text);self.assertFalse(finished)

    def test_finished_parallel_secondary_does_not_hide_primary_training(self):
        secondary=self.root/'native'
        for model in MODELS['local']:
            self.put(model,stage='shared training',status='running',heartbeat=self.now.isoformat())
            path=secondary/model/'progress.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(stage='native done',status='completed')))
        text,finished=render(self.root,'local',self.now,secondary_progress_root=secondary)
        self.assertIn('shared training',text);self.assertNotIn('native done',text);self.assertFalse(finished)


if __name__ == '__main__':
    unittest.main()
