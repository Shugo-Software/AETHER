import argparse
import glob
import html
import json
import os
import re
import subprocess
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHASES = [
    ('1', 'Inventory + principles', 'done', 'hardware-inventory.md'),
    ('2', 'Dual builds (Vulkan + CUDA)', 'done', 'build-guide.md'),
    ('3', 'Single-GPU benchmarks', 'done', 'benchmarks.md'),
    ('4', 'Multi-GPU inference', 'done', 'multi-gpu.md'),
    ('5', 'Cline local-provider profile', 'done', 'cline-integration.md'),
    ('6', 'Dataset generation (pilot 75/75)', 'done', 'dataset-generation.md'),
    ('7', 'Student training (LFM2.5 QLoRA)', 'done', 'student-training.md'),
    ('8', 'Model export (merge -> GGUF -> HF)', 'done', 'export.md'),
    ('9', 'DPO / preference optimization', 'todo', None),
    ('10', 'Jadis 1.0 release (turn-shape fix + multi-quant export)', 'done', 'export.md'),
]

LOGS_TO_KEEP = 3


def gpu_state():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 5:
                idx, name, util, used, total = parts
                gpus.append({'index': idx, 'name': name, 'util': float(util), 'used': float(used), 'total': float(total), 'pct': round(100 * float(used) / float(total), 1)})
        return gpus
    except Exception as e:
        return [{'error': str(e)}]


def teacher_state():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5) as r:
            body = r.read().decode('utf-8', 'replace')
        with urllib.request.urlopen('http://127.0.0.1:8080/v1/models', timeout=5) as r:
            models = [m['id'] for m in json.loads(r.read())['data']]
        return {'up': True, 'health': body.strip(), 'models': models}
    except Exception as e:
        return {'up': False, 'error': str(e)}


def dataset_inventory():
    inv = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'datasets', '*.jsonl'))):
        st = os.stat(path)
        n = 0
        try:
            with open(path, encoding='utf-8') as fh:
                for line in fh:
                    if line.strip():
                        n += 1
        except OSError:
            pass
        man = path.replace('.jsonl', '.manifest.json')
        tag = os.path.basename(path)
        if os.path.exists(man):
            try:
                with open(man, encoding='utf-8') as fh:
                    m = json.load(fh)
                tag = '%s (%s/%s)' % (m.get('tag', tag), m['counts']['written'], m['counts']['requested'])
            except (OSError, ValueError, KeyError):
                pass
        inv.append({'file': os.path.basename(path), 'tag': tag, 'samples': n, 'kb': round(st.st_size / 1024, 1)})
    return inv


def _read_jsonl(path):
    rows = []
    try:
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return rows


def training_status():
    out = {'state': 'not started', 'detail': None, 'log_tail': [], 'losses': [],
           'epochs': [], 'metrics': {'steps': [], 'losses': [], 'grad_norms': [],
                                     'lrs': [], 'step_time_s': None,
                                     'eta_sec': None}, 'samples': [], 'config': None}
    states = sorted(glob.glob(os.path.join(ROOT, 'logs', 'train-*.json')), key=os.path.getmtime)
    logs = sorted(glob.glob(os.path.join(ROOT, 'logs', 'train-*.log')), key=os.path.getmtime)
    if states:
        try:
            with open(states[-1], encoding='utf-8') as fh:
                out['detail'] = json.load(fh)
            out['state_file'] = os.path.basename(states[-1])
            out['state'] = out['detail'].get('status', 'unknown')
            out['config'] = out['detail'].get('config')
        except (OSError, ValueError) as e:
            out['detail'] = 'unreadable: %s' % e
    if logs:
        try:
            with open(logs[-1], encoding='utf-8', errors='replace') as fh:
                lines = fh.readlines()
            out['log_tail'] = [ln.rstrip() for ln in lines[-30:]]
            out['log_file'] = os.path.basename(logs[-1])
        except OSError as e:
            out['log_tail'] = ['unreadable: %s' % e]

    # Structured per-step telemetry from train-<tag>.metrics.jsonl (Phase 7 metrics hook).
    metrics_files = sorted(glob.glob(os.path.join(ROOT, 'logs', 'train-*.metrics.jsonl')),
                           key=os.path.getmtime)
    if metrics_files:
        mpath = metrics_files[-1]
        out['metrics_file'] = os.path.basename(mpath)
        rows = _read_jsonl(mpath)
        steps, losses, gnorms, lrs, ts, totals = [], [], [], [], [], []
        for r in rows:
            if r.get('step') is not None:
                steps.append(r['step'])
            if r.get('loss') is not None:
                losses.append(round(float(r['loss']), 4))
            if r.get('grad_norm') is not None:
                gnorms.append(round(float(r['grad_norm']), 4))
            if r.get('learning_rate') is not None:
                lrs.append(round(float(r['learning_rate']), 8))
            if r.get('total') is not None:
                totals.append(int(r['total']))
            ts.append(r.get('ts'))
        m = out['metrics']
        m['steps'], m['losses'], m['grad_norms'], m['lrs'] = steps, losses, gnorms, lrs
        # total optimizer steps: from metrics 'total', else parse the log tail "N/M".
        total = max(totals) if totals else None
        if not total:
            for ln in reversed(out.get('log_tail', [])):
                mt = re.search(r'(\d+)\s*/\s*(\d+)\s*\[', ln)
                if mt:
                    total = int(mt.group(2))
                    break
        m['total'] = total
        # step time = median seconds between consecutive timestamps
        deltas = []
        for a, b in zip(ts, ts[1:]):
            try:
                d = (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
                if 0 < d < 3600:
                    deltas.append(d)
            except (TypeError, ValueError):
                pass
        if deltas:
            deltas.sort()
            m['step_time_s'] = round(deltas[len(deltas) // 2], 2)
        # ETA = (total - current step) * step_time
        if m['step_time_s'] and steps and total:
            m['eta_sec'] = int(max(total - steps[-1], 0) * m['step_time_s'])
        # keep legacy keys for the old loss table
        out['losses'] = losses[-LOGS_TO_KEEP:]
        out['epochs'] = [r.get('epoch') for r in rows if r.get('step') is not None][-LOGS_TO_KEEP:]

    # Live generations from train-<tag>.samples.json (sample_student.py).
    sample_files = sorted(glob.glob(os.path.join(ROOT, 'logs', 'train-*.samples.json')),
                          key=os.path.getmtime)
    if sample_files:
        try:
            with open(sample_files[-1], encoding='utf-8') as fh:
                data = json.load(fh)
            out['samples'] = data.get('samples', [])
            out['samples_file'] = os.path.basename(sample_files[-1])
        except (OSError, ValueError):
            pass
    return out


def control_state():
    """Training-control panel data: dataset list + adapters + active run."""
    import train_control as tc
    active = tc.active_run()
    return {'datasets': tc.list_datasets(),
            'adapters': tc.list_adapters(),
            'active': active,
            'can_start': active is None}


def export_state():
    """Latest export status (logs/export-<adapter>.json) + active run + log tail."""
    import export_control as ec
    out = {'active': ec.active_export(), 'status': None, 'log_tail': []}
    states = sorted(glob.glob(os.path.join(ROOT, 'logs', 'export-*.json')),
                    key=os.path.getmtime)
    states = [s for s in states if not s.endswith('export-active.json')]
    if states:
        try:
            with open(states[-1], encoding='utf-8') as fh:
                out['status'] = json.load(fh)
        except (OSError, ValueError):
            pass
        log = os.path.splitext(states[-1])[0] + '.log'
        try:
            with open(log, encoding='utf-8', errors='replace') as fh:
                out['log_tail'] = fh.read().splitlines()[-15:]
        except OSError:
            pass
    return out


def snapshot():
    return {'updated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'phases': [{'n': n, 'name': name, 'status': s, 'doc': d} for n, name, s, d in PHASES],
            'gpus': gpu_state(),
            'teacher': teacher_state(),
            'datasets': dataset_inventory(),
            'training': training_status(),
            'export': export_state(),
            'control': control_state()}
