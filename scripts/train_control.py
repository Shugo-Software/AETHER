#!/usr/bin/env python3
"""AETHER dashboard control: start/stop student training from the web UI.

Loopback-only, stdlib-only. The dashboard (dashboard.py) calls start_training /
stop_training; dashdata.py calls active_run / list_datasets for the snapshot.

Safety:
  * training only runs the local venv python on scripts/train_student.py — no
    arbitrary command/path execution (dataset must live under datasets/, tag is
    [A-Za-z0-9][A-Za-z0-9._-]* — dots allowed for release names like jadis-1.0).
  * single concurrent run (one 1060 6GB card); double-start refused.
  * stop only kills a PID whose command line still names train_student.py — it
    never kills a foreign process that reused the PID.
  * all caches/env point at G: (C: is critically small — plan.md); the child is
    detached (CREATE_NEW_PROCESS_GROUP) so it survives a dashboard restart.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
DATASETS = os.path.join(ROOT, "datasets")
ADAPTERS = os.path.join(ROOT, "adapters")
ACTIVE_PATH = os.path.join(LOGS, "train-active.json")
TRAINER = os.path.join(ROOT, "scripts", "train_student.py")
TRAINER_MARKER = "train_student.py"
CREATE_NEW_PROCESS_GROUP = 0x00000200  # detached on Windows

# Env mirroring run-train.cmd: everything on G:, single 1060 6GB card.
G = ROOT
TRAIN_ENV = {
    "HF_HOME": os.path.join(G, "cache", "hf"),
    "HF_HUB_CACHE": os.path.join(G, "cache", "hf"),
    "HF_DATASETS_CACHE": os.path.join(G, "cache", "hf"),
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "PIP_CACHE_DIR": os.path.join(G, "cache", "pip"),
    "TEMP": os.path.join(G, "cache", "temp"),
    "TMP": os.path.join(G, "cache", "temp"),
    "TMPDIR": os.path.join(G, "cache", "temp"),
    "CUDA_VISIBLE_DEVICES": "1",  # the 1060 6GB; teacher stays on CUDA0 (4070 Ti)
}

_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_datasets():
    """[{file, samples, kb}] for the dataset picker (same shape as dashdata)."""
    out = []
    for path in sorted(glob.glob(os.path.join(DATASETS, "*.jsonl"))):
        n = 0
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        n += 1
        except OSError:
            pass
        out.append({"file": os.path.basename(path), "samples": n,
                    "kb": round(os.path.getsize(path) / 1024, 1)})
    return out


def list_adapters():
    """[{dir, mb, modified}] for the resume picker: dirs with adapter_config.json."""
    out = []
    for path in sorted(glob.glob(os.path.join(ADAPTERS, "*"))):
        if not os.path.isdir(path) \
                or not os.path.isfile(os.path.join(path, "adapter_config.json")):
            continue
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(path) for f in fs)
        out.append({"dir": os.path.basename(path),
                    "mb": round(size / 1024 / 1024, 1),
                    "modified": datetime.fromtimestamp(
                        os.path.getmtime(path), timezone.utc
                    ).isoformat(timespec="seconds")})
    return out


def _read_active():
    try:
        with open(ACTIVE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_active(rec):
    os.makedirs(LOGS, exist_ok=True)
    with open(ACTIVE_PATH, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)


def _cmdline(pid):
    """Command line of a PID on Windows (via CIM); '' if gone/unreadable."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter 'ProcessId=%d').CommandLine" % pid],
            capture_output=True, text=True, timeout=15)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _alive_and_ours(pid):
    if not pid:
        return False
    if TRAINER_MARKER not in _cmdline(int(pid)):
        return False
    try:
        os.kill(int(pid), 0)  # signal 0 = existence check
        return True
    except OSError:
        return False


def active_run():
    """Any live training run — ours or the queue's — else None (self-heals).

    This used to read only train-active.json, the record *this* module writes. A run
    the queue started (hf_datasets.train -> recursive_train.py) therefore showed as
    no run at all: the Stop button stayed disabled while Start would have launched a
    second trainer onto the card the loop's trainer already occupied (2026-10-15).
    run_state resolves every launcher's record through the process table.
    """
    rec = _read_active()
    if rec and _alive_and_ours(rec.get("pid")):
        return rec
    if rec:  # stale record (crashed/finished) — clean it up
        try:
            os.remove(ACTIVE_PATH)
        except OSError:
            pass
    try:
        import run_state
        st = run_state.active()
    except Exception:  # noqa: BLE001 — the control plane must never break
        return None
    p = st.get("primary")
    if not p:
        return None
    return {"pid": p["pid"], "tag": p.get("tag"), "dataset": p.get("dataset"),
            "started_utc": p.get("started_utc"), "role": p["role"],
            "source": p["source"], "repo": p.get("repo"), "log": p.get("log"),
            "corpus": p.get("corpus"), "tree": p.get("tree"),
            "cmdline": p.get("cmdline")}


def describe_run(rec):
    """One line naming a live run ("loop pid=25480 tag=… (queue training)").

    Shared by the start/stop messages and the dashboard's control panel, so a user
    can always tell *what* is running and who started it.
    """
    if not rec:
        return "nothing running"
    bits = "%s pid=%s" % (rec.get("role") or "run", rec.get("pid"))
    for key, label in (("tag", "tag"), ("repo", "repo")):
        if rec.get(key):
            bits += " %s=%s" % (label, rec[key])
    if rec.get("source"):
        bits += " (%s)" % rec["source"]
    return bits


def _num(cfg, key, default, cast):
    try:
        return cast(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def start_training(cfg):
    """Validate + launch train_student.py. Returns (ok, payload)."""
    if not isinstance(cfg, dict):
        return False, {"error": "expected a JSON object"}
    live = active_run()
    if live:
        # Name it: it may be a queue/loop run this module never launched, which is
        # exactly the case that used to let a second trainer start on a busy card.
        return False, {"error": "already running — %s. Stop it first."
                                % describe_run(live),
                       "active": live}

    dataset = (cfg.get("dataset") or "").strip()
    if not dataset:
        return False, {"error": "dataset is required"}
    # dataset must be a bare filename under datasets/ (no traversal / injection)
    ds_path = os.path.join(DATASETS, dataset)
    if os.path.basename(dataset) != dataset or not dataset.endswith(".jsonl") \
            or not os.path.exists(ds_path):
        return False, {"error": "unknown dataset: %r" % dataset}

    tag = (cfg.get("tag") or "").strip()
    if not _TAG_RE.match(tag):
        return False, {"error": "tag must match [A-Za-z0-9][A-Za-z0-9._-]*"}

    # Optional: which base model to train. Checked against the registry, because a
    # typo here would otherwise start a multi-hour run against the wrong base — and
    # the adapter it writes then refuses to merge anywhere.
    import aether_models as am
    model = (cfg.get("model") or "").strip()
    if model and not am.known(model):
        return False, {"error": "unknown base model %r — add it to aether_models.py "
                                "first" % model}

    # Optional: adapter dir name. Without it train_student.py uses its legacy
    # adapters/lfm25-<tag>; pass out="jadis-1.0" for a release-style name.
    out = (cfg.get("out") or "").strip()
    if out and (os.path.basename(out) != out or not _TAG_RE.match(out)):
        return False, {"error": "out must be a bare adapter dir name"}

    # Optional: continue training an existing adapter (e.g. jadis on new data).
    resume = (cfg.get("resume") or "").strip()
    if resume:
        rdir = os.path.join(ADAPTERS, resume)
        if (os.path.basename(resume) != resume or not _TAG_RE.match(resume)
                or not os.path.isfile(os.path.join(rdir, "adapter_config.json"))):
            return False, {"error": "unknown adapter to resume: %r" % resume}

    epochs = _num(cfg, "epochs", 3, float)
    lr = _num(cfg, "lr", 2e-4, float)
    max_len = _num(cfg, "max_len", 1024, int)
    lora_r = _num(cfg, "lora_r", 16, int)
    lora_alpha = _num(cfg, "lora_alpha", 32, int)
    batch = _num(cfg, "batch", 1, int)
    grad_accum = _num(cfg, "grad_accum", 16, int)
    max_steps = _num(cfg, "max_steps", -1, int)

    cmd = [sys.executable, "-u", TRAINER,
           "--dataset", ds_path, "--tag", tag,
           "--epochs", str(epochs), "--lr", str(lr),
           "--max-len", str(max_len), "--lora-r", str(lora_r),
           "--lora-alpha", str(lora_alpha), "--batch", str(batch),
           "--grad-accum", str(grad_accum)]
    if out:
        cmd += ["--out", os.path.join(ADAPTERS, out)]
    if model:
        cmd += ["--model", model]
    if max_steps and max_steps > 0:
        cmd += ["--max-steps", str(max_steps)]
    if resume:
        cmd += ["--resume", os.path.join(ADAPTERS, resume)]

    env = dict(os.environ)
    env.update(TRAIN_ENV)
    os.makedirs(os.path.join(G, "cache", "temp"), exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    log_path = os.path.join(LOGS, "train-%s.log" % tag)
    try:
        logf = open(log_path, "w", encoding="utf-8")
    except OSError as e:
        return False, {"error": "cannot open log %s: %s" % (log_path, e)}
    try:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=ROOT, env=env,
                                creationflags=CREATE_NEW_PROCESS_GROUP)
    except Exception as e:  # noqa: BLE001
        logf.close()
        return False, {"error": "failed to launch: %s" % e}

    rec = {"pid": proc.pid, "tag": tag, "dataset": dataset, "started_utc": _now(),
           "log": os.path.basename(log_path),
           "resume": resume or None,
           "out": out or None,
           "cfg": {"epochs": epochs, "lr": lr, "max_len": max_len,
                   "lora_r": lora_r, "lora_alpha": lora_alpha, "batch": batch,
                   "grad_accum": grad_accum, "max_steps": max_steps,
                   "resume": resume or None, "out": out or None}}
    _write_active(rec)
    return True, rec


def stop_training(dry_run=False):
    """Stop the live training run, whoever started it.

    One code path decides what "our job" means: run_state verifies the pid's command
    line and kills the whole subtree (the loop, its trainer and their workers). The
    old version only knew train-active.json, so a queue-started run could not be
    stopped from the UI at all.
    """
    rec = active_run()
    if not rec:
        stale = _read_active()
        if stale:
            try:
                os.remove(ACTIVE_PATH)
            except OSError:
                pass
            return True, {"stopped": False, "note": "run already finished"}
        return True, {"stopped": False, "note": "no active run"}
    pid = int(rec["pid"])
    try:
        import run_state
        res = run_state.stop(rec, dry_run=dry_run)
        res.setdefault("tag", rec.get("tag"))
        if not res.get("ok"):
            return False, res
        if dry_run:
            return True, res
        try:
            os.remove(ACTIVE_PATH)
        except OSError:
            pass
        return True, res
    except Exception:  # noqa: BLE001 — fall back to the narrow, marker-checked kill
        if TRAINER_MARKER not in _cmdline(pid):
            return False, {"error": "PID %d is not our trainer — refusing to kill" % pid}
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=20)
        except Exception as e:  # noqa: BLE001
            return False, {"error": "taskkill failed: %s" % e}
        try:
            os.remove(ACTIVE_PATH)
        except OSError:
            pass
        return True, {"stopped": True, "pid": pid, "tag": rec.get("tag")}


