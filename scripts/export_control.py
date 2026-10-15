#!/usr/bin/env python3
"""Dashboard control: launch/stop model exports (export_model.py).

Same safety pattern as train_control.py: no arbitrary command execution
(adapter must be a bare name under adapters/), single concurrent export,
stop only kills a PID whose command line still names export_model.py.
Auth for Hugging Face is NEVER taken from the web — export_model.py reads
HF_TOKEN or the stored `huggingface-cli login` token.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import export_model as _em  # single source of truth for QUANT_TYPES
import train_control as tc  # shared: ROOT/LOGS/ADAPTERS env + guards

ROOT, LOGS, ADAPTERS = tc.ROOT, tc.LOGS, tc.ADAPTERS
EXPORTER = os.path.join(ROOT, "scripts", "export_model.py")
EXPORTER_MARKER = "export_model.py"
ACTIVE_PATH = os.path.join(LOGS, "export-active.json")
CREATE_NEW_PROCESS_GROUP = 0x00000200  # detached on Windows

_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_QUANTS = ("",) + tuple(sorted(_em.QUANT_TYPES))


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_active():
    try:
        with open(ACTIVE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def active_export():
    """The running export if its PID is alive + ours, else None (self-heals)."""
    rec = _read_active()
    if not rec:
        return None
    pid = rec.get("pid")
    if not pid:
        return None
    try:
        cmdline = tc._cmdline(int(pid))
    except (TypeError, ValueError):
        cmdline = ""
    if EXPORTER_MARKER not in cmdline:  # gone, or PID reused by a foreign process
        try:
            os.remove(ACTIVE_PATH)
        except OSError:
            pass
        return None
    return rec


def start_export(cfg):
    """Validate + launch export_model.py. Returns (ok, payload)."""
    if not isinstance(cfg, dict):
        return False, {"error": "expected a JSON object"}
    if active_export():
        return False, {"error": "an export is already running — stop it first"}

    adapter = (cfg.get("adapter") or "").strip()
    if (os.path.basename(adapter) != adapter or not tc._TAG_RE.match(adapter)
            or not os.path.isfile(os.path.join(ADAPTERS, adapter,
                                               "adapter_config.json"))):
        return False, {"error": "unknown adapter: %r" % adapter}

    gguf = (cfg.get("gguf") or "").strip()
    if gguf not in _QUANTS:
        return False, {"error": "gguf must be one of: (none), %s"
                       % ", ".join(q for q in _QUANTS if q)}
    hub = bool(cfg.get("hub"))
    repo = (cfg.get("repo") or "").strip()
    if repo and not _REPO_RE.match(repo):
        return False, {"error": "repo must look like owner/name"}
    public = bool(cfg.get("public"))

    cmd = [sys.executable, "-u", EXPORTER, "--adapter", adapter]
    if gguf:
        cmd += ["--gguf", gguf]
    if hub:
        cmd += ["--hub"]
        if repo:
            cmd += ["--repo", repo]
        if public:
            cmd.append("--public")

    # Mirror train_control: inherit the real process env, then overlay the
    # G:-cache vars. A bare TRAIN_ENV dict (no SystemRoot/PATH) breaks Winsock
    # (WinError 10106) and DLL loading in the child.
    env = dict(os.environ)
    env.update(tc.TRAIN_ENV)
    os.makedirs(os.path.join(tc.ROOT, "cache", "temp"), exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    log_path = os.path.join(LOGS, "export-%s.log" % adapter)
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

    rec = {"pid": proc.pid, "adapter": adapter, "started_utc": _now(),
           "log": os.path.basename(log_path),
           "cfg": {"gguf": gguf or None, "hub": hub, "repo": repo or None,
                   "public": public}}
    with open(ACTIVE_PATH, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    return True, rec


def stop_export():
    """Stop the running export (only if cmdline still names export_model.py)."""
    rec = _read_active()
    if not rec:
        try:
            os.remove(ACTIVE_PATH)
        except OSError:
            pass
        return True, {"stopped": False, "note": "no active export"}
    pid = int(rec["pid"])
    if EXPORTER_MARKER not in tc._cmdline(pid):
        return False, {"error": "PID %d is not our exporter — refusing to kill" % pid}
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, timeout=20)
    except Exception as e:  # noqa: BLE001
        return False, {"error": "taskkill failed: %s" % e}
    try:
        os.remove(ACTIVE_PATH)
    except OSError:
        pass
    return True, {"stopped": True, "pid": pid, "adapter": rec.get("adapter")}
