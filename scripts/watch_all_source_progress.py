"""Refresh one machine's compact progress note from experiment heartbeats."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time


MODELS = {
    'local': ('qwen2_5_vl_7b', 'llava_1_5_7b'),
    '32678': ('qwen3_vl_8b', 'internvl_2_5_8b'),
}
TERMINAL = {'completed', 'failed'}
STALE_SECONDS = 120


def short_text(value, limit=180):
    return ' '.join(str(value).split())[:limit]


def load_progress(path, now):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('progress must be an object')
        stamp = path.stat().st_mtime
    except FileNotFoundError:
        return {'stage': '准备中', 'status': '准备中'}, False
    except (OSError, ValueError) as exc:
        return {'stage': '读取状态', 'status': '状态暂不可读', 'error': str(exc)}, False
    try:
        heartbeat = datetime.fromisoformat(str(value['heartbeat']).replace('Z', '+00:00'))
        stamp = heartbeat.replace(tzinfo=timezone.utc).timestamp() if heartbeat.tzinfo is None else heartbeat.timestamp()
    except (KeyError, ValueError, TypeError):
        pass
    terminal = value.get('status') in TERMINAL
    if not terminal and now.timestamp() - stamp > STALE_SECONDS:
        value['status'] = short_text(value.get('status', '运行中')) + '（无更新）'
    return value, terminal


def count(value):
    try:
        return max(0, int(value)) if math.isfinite(float(value)) else 0
    except (TypeError, ValueError, OverflowError):
        return 0


def render(root, machine, now=None, progress_root=None, secondary_progress_root=None, models=None):
    now = now or datetime.now(timezone.utc)
    lines = [f"UTC 更新时间：{now.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | 服务器：{machine}"]
    finished = []
    for model in (tuple(models) if models else MODELS[machine]):
        path = (Path(progress_root) if progress_root is not None else root / 'outputs/ffn_all_source_paths_v1') / model / 'progress.json'
        value, terminal = load_progress(path, now)
        if secondary_progress_root is not None:
            secondary_path = Path(secondary_progress_root) / model / 'progress.json'
            if secondary_path.exists():
                secondary, secondary_terminal = load_progress(secondary_path, now)
                if value.get('status') == 'completed' or secondary.get('status') not in ('waiting', '准备中', 'completed'):
                    value, terminal = secondary, secondary_terminal
                    path = secondary_path
                elif secondary.get('status') != 'completed':
                    terminal = False
        if value.get('stage') == '图片配对bootstrap' and not terminal:
            try:
                detail = json.loads((path.parent/'bootstrap_progress.json').read_text())
                value.update(completed=detail['completed'], total=detail['total'])
            except (OSError, ValueError, KeyError):
                pass
        finished.append(terminal)
        completed, total = count(value.get('completed')), count(value.get('total'))
        filled = min(20, completed * 20 // total) if total else 0
        bar = '█' * filled + '░' * (20 - filled)
        line = (f"{model} | {short_text(value.get('stage', '准备中'))} | [{bar}] "
                f"{completed}/{total} {100*completed/total if total else 0:.1f}% | {short_text(value.get('status', '准备中'))}")
        if value.get('epoch') is not None:
            line += f" | epoch {short_text(value['epoch'], 40)}"
        if value.get('error'):
            line += f" | error: {short_text(value['error'])}"
        lines.append(line)
    return '\n'.join(lines) + '\n', all(finished)


def write_note(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def watch(root, machine, once=False, progress_root=None, secondary_progress_root=None, models=None):
    root = Path(root).resolve()
    suffix = 'LOCAL' if machine == 'local' else machine
    note = root / f'EXPERIMENT_PROGRESS_{suffix}.md'
    with (root / f'.EXPERIMENT_PROGRESS_{suffix}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'Progress watcher already active for {machine}', file=sys.stderr)
            return 1
        while True:
            content, finished = render(root, machine, progress_root=progress_root,
                                       secondary_progress_root=secondary_progress_root, models=models)
            write_note(note, content)
            if once or finished:
                return 0
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', choices=MODELS, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--progress-root', type=Path)
    parser.add_argument('--secondary-progress-root', type=Path)
    parser.add_argument('--models', nargs='+', help='Override the two model IDs shown for this machine')
    args = parser.parse_args()
    return watch(args.root, args.machine, args.once, args.progress_root, args.secondary_progress_root, args.models)


if __name__ == '__main__':
    raise SystemExit(main())
