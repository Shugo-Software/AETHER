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
    ('11', 'External datasets + recursive loop (pull to G: -> train -> gate -> release)', 'active', 'dataset-generation.md'),
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


def corpus_state():
    """What the model actually learns: label masking + corpus health.

    Every number comes from a file a pipeline stage wrote (the trainer's state,
    verify_dataset's report, fit_corpus's provenance), so the panel shows
    measurements rather than intentions — and `warnings` names the failure modes
    (unmasked loss, answers truncated away, anchor leakage) instead of hiding
    them behind a green tick.
    """
    out = {'train': None, 'verify': None, 'fit': None, 'warnings': [], 'notes': []}
    states = [s for s in sorted(glob.glob(os.path.join(ROOT, 'logs', 'train-*.json')),
                                key=os.path.getmtime)
              if not s.endswith('train-active.json')]
    if states:
        try:
            with open(states[-1], encoding='utf-8') as fh:
                st = json.load(fh)
            cfg = st.get('config') or {}
            # The state file is merged across runs, so train_loss can be the
            # PREVIOUS run's value while this one is still training (seen: the
            # panel showing 5.82 from iteration 1 while iteration 2 was at 0.77).
            # Only report a final loss once the run that wrote the file is done.
            done = st.get('status') in ('done', 'saving')
            out['train'] = {
                'file': os.path.basename(states[-1]), 'tag': st.get('tag'),
                'status': st.get('status'),
                'loss': st.get('train_loss') if done else None,
                'dataset': cfg.get('dataset'), 'max_len': cfg.get('max_len'),
                'max_steps': cfg.get('max_steps'), 'lr': cfg.get('lr'),
                'prompt_loss': cfg.get('prompt_loss'),
                'rows': st.get('rows'), 'rows_used': cfg.get('rows_used'),
                'dropped_no_answer': cfg.get('dropped_no_answer'),
                'answer_token_share': cfg.get('answer_token_share')}
        except (OSError, ValueError):
            pass
    reports = sorted(glob.glob(os.path.join(ROOT, 'logs', 'verify-dataset-*.json')),
                     key=os.path.getmtime)
    if reports:
        try:
            with open(reports[-1], encoding='utf-8') as fh:
                rep = json.load(fh)
            t = rep.get('tokens') or {}
            cnt = rep.get('counts') or (t.get('tokens') or {})
            dup = rep.get('duplicates') or {}
            over = rep.get('overlap') or {}
            out['verify'] = {
                'file': os.path.basename(reports[-1]), 'corpus': rep.get('file'),
                'rows': t.get('rows'), 'max_len': t.get('max_len'),
                'classes': t.get('classes'),
                'answer_cut': (t.get('classes') or {}).get('answer_cut'),
                'terminator_pct': t.get('terminator_pct'),
                'answer_tokens_kept_pct': cnt.get('completion_kept_pct'),
                'answer_token_share': cnt.get('answer_loss_share'),
                'duplicate_groups': dup.get('duplicate_groups'),
                'duplicate_rows': dup.get('rows_in_duplicate_groups'),
                'anchor_overlap': over.get('shared'),
                'len_tokens': t.get('len_tokens')}
        except (OSError, ValueError):
            pass
    fits = sorted(glob.glob(os.path.join(ROOT, 'datasets', '*.filter.json')),
                  key=os.path.getmtime)
    if fits:
        try:
            with open(fits[-1], encoding='utf-8') as fh:
                f = json.load(fh)
            out['fit'] = {'file': os.path.basename(fits[-1]),
                          'rows_in': (f.get('input') or {}).get('rows'),
                          'rows_out': (f.get('output') or {}).get('rows'),
                          'trimmed_rows': f.get('trimmed_rows'),
                          'dropped': f.get('dropped'),
                          'policy': f.get('policy')}
        except (OSError, ValueError):
            pass
    out['warnings'], out['notes'] = _corpus_findings(out)
    return out


def _corpus_findings(out):
    """Warnings/notes derived from the three stage reports."""
    warns, notes = [], []
    tr = out.get('train') or {}
    ve = out.get('verify') or {}
    fi = out.get('fit') or {}
    if tr:
        if tr.get('prompt_loss') == 'full':
            warns.append('prompt_loss=full — the loss covers the prompt too, so '
                         'most of the gradient is reproducing the input rather '
                         'than answering')
        elif tr.get('prompt_loss') == 'masked':
            share = tr.get('answer_token_share')
            notes.append('prompt_loss=masked — only answer tokens carry gradient%s'
                         % (' (answers are %d%% of the tokens kept)'
                            % round(100 * share)
                            if isinstance(share, (int, float)) else ''))
        if (tr.get('dropped_no_answer') or 0) > 0:
            warns.append('%d rows had no answer tokens left at max_len=%s and were '
                         'dropped during training — fit the corpus first'
                         % (tr['dropped_no_answer'], tr.get('max_len')))
        if tr.get('rows_used') and tr.get('rows') and tr['rows_used'] < tr['rows']:
            notes.append('%d of %d corpus rows were used'
                         % (tr['rows_used'], tr['rows']))
    if ve:
        if (ve.get('answer_cut') or 0) > 0:
            warns.append('%d rows lose their answer entirely at max_len=%s — with '
                         'masked loss they teach nothing, and unmasked they teach '
                         'the prompt' % (ve['answer_cut'], ve.get('max_len')))
        if ve.get('anchor_overlap'):
            warns.append('%s prompts also appear in the anchor set — comparisons '
                         'against it are contaminated' % ve['anchor_overlap'])
        if ve.get('duplicate_groups'):
            notes.append('%d duplicate prompt groups cover %s rows'
                         % (ve['duplicate_groups'], ve.get('duplicate_rows')))
        if fi and fi.get('trimmed_rows'):
            notes.append('%d answers trimmed to fit the budget'
                         % fi['trimmed_rows'])
    else:
        warns.append('this corpus has never been measured — run verify_dataset.py '
                     '(--max-len) before trusting a training run on it')
    return warns, notes


def hf_state():
    """Linked Hugging Face datasets: state, disk cost, queue and job feed."""
    try:
        import hf_datasets as hf
    except Exception as e:  # noqa: BLE001 — the panel must not break the page
        return {'error': str(e), 'datasets': [], 'jobs': [], 'storage': {}}
    data = hf.load_registry()
    items = []
    for e in data['datasets']:
        items.append({
            'id': e['id'], 'state': e.get('state'), 'tag': e.get('tag'),
            'rows': e.get('rows'), 'fitted_rows': e.get('fitted_rows'),
            'bytes_human': hf.human(e.get('bytes') or 0),
            'freed_human': (hf.human(e.get('freed_bytes'))
                            if e.get('freed_bytes') else None),
            'corpus': e.get('corpus'), 'fitted': e.get('fitted'),
            'health': e.get('health'), 'audit': e.get('audit'),
            'dropped': e.get('dropped'), 'note': e.get('note'),
            'license': e.get('license'), 'likes': e.get('likes'),
            'downloads': e.get('downloads'), 'split': e.get('pulled_split'),
            'on_disk': os.path.isdir(e.get('local_dir') or ''),
            'gate': e.get('gate'), 'revision': (e.get('revision') or '')[:8]})
    st = hf.storage()
    return {'datasets': items, 'queue': hf.queue_summary(data),
            'jobs': hf.jobs()[:12],
            'storage': {'root': st['root'], 'total_human': st['total_human'],
                        'free_human': st['free_human'],
                        'orphans': st['orphans']}}


def snapshot():
    return {'updated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'phases': [{'n': n, 'name': name, 'status': s, 'doc': d} for n, name, s, d in PHASES],
            'gpus': gpu_state(),
            'teacher': teacher_state(),
            'datasets': dataset_inventory(),
            'training': training_status(),
            'corpus': corpus_state(),
            'hfd': hf_state(),
            'export': export_state(),
            'control': control_state()}
