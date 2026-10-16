import argparse
import glob
import html
import json
import os
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The page polls every 5s, so anything that shells out or touches the network is
# cached: without this, snapshot() cost 6.5s+ (endpoints 4.7s, host 1.7s) and the
# browser had overlapping requests in flight permanently. gpu_state is cheap
# (one nvidia-smi, ~0.15s) and stays live so the numbers still move.
_CACHE = {}


def _cached(key, ttl, fn):
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val

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
    """Live per-GPU telemetry. Temperature/power are optional: Pascal cards can
    report [N/A] for power, so a missing field must not break the panel."""
    try:
        out = subprocess.run(['nvidia-smi',
                              '--query-gpu=index,name,utilization.gpu,memory.used,'
                              'memory.total,temperature.gpu,power.draw,'
                              'power.limit',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10)
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                continue
            idx, name, util, used, total = parts[:5]
            rest = parts[5:]

            def _num(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            gpus.append({'index': idx, 'name': name, 'util': float(util),
                         'used': float(used), 'total': float(total),
                         'pct': round(100 * float(used) / float(total), 1),
                         'temp': _num(rest[0]) if rest else None,
                         'power': _num(rest[1]) if len(rest) > 1 else None,
                         'power_limit': _num(rest[2]) if len(rest) > 2 else None})
        return gpus
    except Exception as e:
        return [{'error': str(e)}]


def host_state():
    """CPU load and RAM. RAM comes from the Win32 struct (instant); CPU needs a
    subprocess, so it is the part worth caching."""
    ram = None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            gb = 1024.0 ** 3
            total = stat.ullTotalPhys / gb
            avail = stat.ullAvailPhys / gb
            ram = {'used_gb': round(total - avail, 1), 'total_gb': round(total, 1),
                   'pct': int(stat.dwMemoryLoad)}
    except Exception:  # noqa: BLE001
        ram = None
    try:
        cpu = subprocess.run(['powershell', '-NoProfile', '-Command',
                              "(Get-CimInstance Win32_Processor | "
                              "Measure-Object -Property LoadPercentage -Average)"
                              ".Average"],
                             capture_output=True, text=True, timeout=15).stdout
        load = float(cpu.strip()) if cpu.strip() else None
    except Exception:  # noqa: BLE001
        load = None
    return {'cpu_pct': load, 'ram': ram}


def _endpoint_procs():
    """Slow part of endpoints_state(): process table + per-PID VRAM.

    Two subprocesses (~2s). Cached separately from the health probe so the
    live/down indicator can refresh often without re-paying for this.
    """
    procs = {}
    try:
        ps = ("Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\""
              " | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
        raw = subprocess.run(['powershell', '-NoProfile', '-Command', ps],
                             capture_output=True, text=True, timeout=20).stdout
        items = json.loads(raw or '[]')
        if isinstance(items, dict):
            items = [items]
        for p in items:
            cmd = p.get('CommandLine') or ''

            def flag(name):
                m = re.search(r'%s\s+("[^"]+"|\S+)' % re.escape(name), cmd)
                return m.group(1).strip('"') if m else None
            gguf = re.search(r'([A-Za-z]:\\[^"]*?\.gguf)', cmd) \
                or re.search(r'(\S+\.gguf)', cmd)
            procs[flag('--port') or str(p.get('ProcessId'))] = {
                'pid': p.get('ProcessId'), 'alias': flag('--alias'),
                'device': flag('--device') or flag('-dev'),
                'ctx': flag('-c') or flag('--ctx-size'),
                'model': os.path.basename(gguf.group(1)) if gguf else None}
    except Exception:  # noqa: BLE001 — the panel must not break the page
        pass
    vram = {}
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            pid, mb = [x.strip() for x in line.split(',')]
            vram[pid] = float(mb)
    except Exception:  # noqa: BLE001
        pass
    return procs, vram


def _probe_port(port, timeout=0.25):
    """Is something listening on this port?

    A raw TCP connect, not an HTTP request: the expensive part of the old version
    (3x urlopen = 4s per snapshot) was HTTP to endpoints that are mid-training or
    down. Everything the panel *shows* comes from the process command line, which
    is already read once and cached; this only answers "is it up right now".
    """
    import socket
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=timeout):
            return True
    except OSError:
        return False


def endpoints_state():
    """Every llama-server endpoint: alias, GPU, model file, VRAM, liveness.

    The dashboard used to show only the teacher. The pipeline now runs three
    endpoints at once (teacher + two students), and after an iteration the
    question that matters is *which weights the students are serving* — so this
    reads each server's own command line rather than guessing.
    """
    procs, vram = _cached('endpoint_procs', 30, _endpoint_procs)
    out = []
    for port in (8080, 8082, 8083):
        info = procs.get(str(port), {})
        out.append({
            'port': port, 'label': ('teacher' if port == 8080
                                    else 'student %s' % ('A' if port == 8082
                                                         else 'B')),
            'alias': info.get('alias'), 'device': info.get('device'),
            'ctx': info.get('ctx'), 'model': info.get('model'),
            'pid': info.get('pid'),
            'vram_mib': vram.get(str(info.get('pid'))) if info.get('pid') else None,
            'up': _probe_port(port)})
    return out


def _flatten_steps(steps):
    """One level of nesting flattened: {'serve': {'a': {'ok': True}}} -> serve.a.

    Without this the serve step (which reports per student) counted as a single
    unnamed entry and a failed student was invisible in the panel.
    """
    out = {}
    for key, val in (steps or {}).items():
        if key == 'review':
            continue
        if isinstance(val, dict):
            if 'ok' in val:
                out[key] = val.get('ok')
            else:
                for sub, sv in val.items():
                    if isinstance(sv, dict) and 'ok' in sv:
                        out['%s.%s' % (key, sub)] = sv.get('ok')
                    elif isinstance(sv, bool):
                        out['%s.%s' % (key, sub)] = sv
        else:
            out[key] = val
    return out


def _loop_tag(path):
    """The tag a loop-<tag>.json belongs to."""
    base = os.path.basename(path or '')
    if base.startswith('loop-') and base.endswith('.json'):
        return base[len('loop-'):-len('.json')]
    return None


def loop_state(path=None, run_tag=None):
    """The recursive loop's own record: iterations, gates, and what it queued.

    `path` is the loop file belonging to the run the rest of the page shows;
    without it the newest loop wins (loop-selftest.json is a fixture, never a
    run, so _newest() never returns it).

    `run_tag` is that run's tag. When the loop has not written its own file yet —
    a first iteration sitting in its multi-hour train step, before this was
    fixed — the newest file belongs to an *earlier* run, and the panel must say
    so instead of letting its iteration read as the current one.
    """
    if path is None:
        path = _newest(['loop-*.json'])
    if not path:
        return {'file': None, 'tag': run_tag, 'iterations': [],
                'expected_tag': run_tag}
    try:
        with open(path, encoding='utf-8') as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        return {'file': os.path.basename(path), 'tag': _loop_tag(path),
                'iterations': [], 'expected_tag': run_tag}
    iters = []
    for it in state.get('iterations', []):
        gate = it.get('gate') or {}
        reviews = ((it.get('steps') or {}).get('review') or {})
        iters.append({
            'iteration': it.get('iteration'), 'status': it.get('status'),
            'corpus': it.get('corpus'), 'started': it.get('started_utc'),
            'mean_pass_rate': it.get('mean_pass_rate'),
            'reviews': {k: {'pass_rate': v.get('pass_rate'),
                            'scored': v.get('scored')}
                        for k, v in reviews.items()
                        if isinstance(v, dict)},
            'steps': _flatten_steps(it.get('steps')),
            'gate': {'passed': gate.get('passed'),
                     'floor': gate.get('floor'),
                     'clears_floor': gate.get('clears_floor'),
                     'no_regression': gate.get('no_regression')}})
    # A loop record that belongs to another run is labelled as such (`stale_run`),
    # never silently presented as this run's iterations.
    tag_seen = state.get('tag') or _loop_tag(path)
    return {'file': os.path.basename(path), 'tag': tag_seen, 'iterations': iters,
            'expected_tag': run_tag,
            'stale_run': bool(run_tag and tag_seen and tag_seen != run_tag)}


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


# States a trainer writes *while it is working*. If the process behind them is
# gone the run was killed, and the panel must say so rather than repeat the claim
# (seen 2026-10-15: status=training, trainer dead, nothing training).
_LIVE_STATES = ('loading_tokenizer', 'loading_model', 'preparing_data',
                'training', 'saving')


def _trainer_alive(pid):
    """Is the process that wrote a training state still running and still ours?"""
    try:
        import hf_datasets as hf
        return hf.process_alive(pid)
    except Exception:  # noqa: BLE001 — never break the page over a liveness probe
        return None


def _newest(patterns, exclude=('train-active.json', 'loop-selftest.json')):
    """The newest file matching any pattern under logs/ (never a runtime pointer)."""
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(ROOT, 'logs', pat))
    files = [f for f in files if os.path.basename(f) not in exclude]
    return max(files, key=os.path.getmtime) if files else None


def _run_files():
    """The newest training run, and the artifacts belonging to *that* run.

    Each panel used to pick its own file by newest mtime, so one page could show
    iteration 2's stat cards beside iteration 1's log tail, a third tag's
    samples and an unrelated loop's gate decision (seen live 2026-10-15). The
    state file names the run; everything else is keyed to its tag.
    """
    state_file = _newest(['train-*.json'])
    tag = None
    if state_file:
        try:
            with open(state_file, encoding='utf-8') as fh:
                tag = (json.load(fh) or {}).get('tag')
        except (OSError, ValueError):
            tag = None
        if not tag:
            base = os.path.basename(state_file)
            if base.startswith('train-') and base.endswith('.json'):
                tag = base[len('train-'):-len('.json')]
    runs = {'log': None, 'metrics': None, 'samples': None, 'loop': None}
    if tag:
        # The recursive loop tees the trainer to loop-<tag>-it<N>-train.log, so
        # a tag whose own train-<tag>.log was never written still has a log.
        runs['log'] = (_newest(['train-%s.log' % tag])
                       or _newest(['loop-%s-it*-train.log' % tag,
                                   'hf-train-%s.log' % tag]))
        runs['metrics'] = _newest(['train-%s.metrics.jsonl' % tag])
        runs['samples'] = _newest(['train-%s.samples.json' % tag])
        runs['loop'] = _newest(['loop-%s.json' % tag])
    return state_file, tag, runs


def training_status(state_file=None, tag=None, runs=None):
    if state_file is None:
        state_file, tag, runs = _run_files()
    runs = runs or {}
    out = {'state': 'not started', 'detail': None, 'log_tail': [], 'losses': [],
           'epochs': [], 'metrics': {'steps': [], 'losses': [], 'grad_norms': [],
                                     'lrs': [], 'step_time_s': None,
                                     'eta_sec': None}, 'samples': [], 'config': None,
           'tag': tag, 'log_file': None, 'log_note': None}
    if state_file:
        try:
            with open(state_file, encoding='utf-8') as fh:
                out['detail'] = json.load(fh)
            out['state_file'] = os.path.basename(state_file)
            out['state'] = out['detail'].get('status', 'unknown')
            out['config'] = out['detail'].get('config')
        except (OSError, ValueError) as e:
            out['detail'] = 'unreadable: %s' % e
    # A state file may also declare itself stale (a run stopped by hand); surface
    # that so the header can say "stopped" instead of repeating the claim.
    out['stale'] = bool(isinstance(out['detail'], dict)
                        and out['detail'].get('stale'))
    # Ask the process that wrote the file whether it is still there before
    # repeating its claim. The probe shells out to tasklist, so it is cached like
    # every other subprocess here rather than run on each 5 s poll.
    if isinstance(out['detail'], dict) and out['state'] in _LIVE_STATES:
        pid = out['detail'].get('pid')
        alive = (None if not pid else
                 _cached('trainer-pid-%s' % pid, 15,
                         lambda: _trainer_alive(pid)))
        if alive is False:
            detail = dict(out['detail'])
            detail['status'] = 'stopped — the trainer process is gone'
            detail['stale'] = True
            out['detail'] = detail
            out['state'] = 'stopped'
            out['stale'] = True
    log_path = runs.get('log')
    if log_path:
        try:
            with open(log_path, encoding='utf-8', errors='replace') as fh:
                lines = fh.readlines()
            out['log_tail'] = [ln.rstrip() for ln in lines[-30:]]
            out['log_file'] = os.path.basename(log_path)
        except OSError as e:
            out['log_tail'] = ['unreadable: %s' % e]
    else:
        # Say so rather than borrowing another run's log (the panel used to show
        # whichever tag trained last, next to these numbers).
        out['log_note'] = ('no log recorded for tag %s' % tag if tag
                           else 'no training run recorded yet')

    # Structured per-step telemetry from train-<tag>.metrics.jsonl (Phase 7 metrics hook).
    mpath = runs.get('metrics')
    if mpath:
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

    # Live generations from train-<tag>.samples.json (sample_student.py) — the
    # same tag the state above came from, so the panel cannot show another run's.
    sfile = runs.get('samples')
    if sfile:
        try:
            with open(sfile, encoding='utf-8') as fh:
                data = json.load(fh)
            out['samples'] = data.get('samples', [])
            out['samples_file'] = os.path.basename(sfile)
        except (OSError, ValueError):
            pass
    return out


def control_state():
    """Training-control panel data: dataset list + adapters + active run.

    `active` can be a run this module never launched — the queue's loop, a sweep —
    so the panel also gets a description naming the role, tag and repo. That is what
    makes Stop a legible button instead of a mystery, and Start's refusal readable.
    """
    import train_control as tc
    active = tc.active_run()
    return {'datasets': tc.list_datasets(),
            'adapters': tc.list_adapters(),
            'active': active,
            'describe': tc.describe_run(active),
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


def corpus_state(state_file=None):
    """What the model actually learns: label masking + corpus health.

    Every number comes from a file a pipeline stage wrote (the trainer's state,
    verify_dataset's report, fit_corpus's provenance), so the panel shows
    measurements rather than intentions — and `warnings` names the failure modes
    (unmasked loss, answers truncated away, anchor leakage) instead of hiding
    them behind a green tick. `state_file` is the run the rest of the page is
    showing; without it the newest one wins.
    """
    out = {'train': None, 'verify': None, 'fit': None, 'warnings': [], 'notes': []}
    state_file = state_file or _newest(['train-*.json'])
    if state_file:
        try:
            with open(state_file, encoding='utf-8') as fh:
                st = json.load(fh)
            cfg = st.get('config') or {}
            # The state file is merged across runs, so train_loss can be the
            # PREVIOUS run's value while this one is still training (seen: the
            # panel showing 5.82 from iteration 1 while iteration 2 was at 0.77).
            # Only report a final loss once the run that wrote the file is done.
            done = st.get('status') in ('done', 'saving')
            out['train'] = {
                'file': os.path.basename(state_file), 'tag': st.get('tag'),
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
            'items': hf.queue_items(data),
            'jobs': hf.jobs()[:12],
            'storage': {'root': st['root'], 'total_human': st['total_human'],
                        'free_human': st['free_human'],
                        'orphans': st['orphans']}}


def snapshot():
    state_file, tag, runs = _run_files()
    return {'updated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'phases': [{'n': n, 'name': name, 'status': s, 'doc': d} for n, name, s, d in PHASES],
            'gpus': gpu_state(),
            'host': _cached('host', 10, host_state),
            'teacher': _cached('teacher', 5, teacher_state),
            'endpoints': _cached('endpoints', 6, endpoints_state),
            'loop': loop_state(runs.get('loop') or _newest(['loop-*.json']),
                               run_tag=tag),
            'run_tag': tag,
            'datasets': _cached('inventory', 15, dataset_inventory),
            'training': training_status(state_file, tag, runs),
            'corpus': corpus_state(state_file),
            'hfd': _cached('hfd', 8, hf_state),
            'export': export_state(),
            'control': control_state()}
