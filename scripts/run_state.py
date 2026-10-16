#!/usr/bin/env python3
"""One source of truth for "what is running" — what the dashboard's Stop acts on.

The pipeline grew two launchers and each kept its own private idea of the active
run, so the dashboard's controls disagreed with reality:

  logs/train-active.json     written by train_control.start_training  (UI button)
  logs/hf-train-active.json  written by hf_datasets.train             (queue)  <- read by nothing
  logs/export-active.json    written by export_control.start_export   (UI button)
  logs/train-<tag>.json      written by the trainer itself, while it works
  logs/loop-<tag>.json       written by recursive_train.py per stage
  logs/hf-datasets.json      the queue's own registry row (state, loop_pid)

Measured consequence (2026-10-15): a queue-started MiniMax run was training, yet
`control.active` was null and `control.can_start` was True — the Stop button was
disabled while Start would have launched a second trainer onto the card the loop's
trainer already occupied, and the only way to stop the run was Task Manager.

This module answers both questions from every source at once, verifies liveness
against the process's command line (Windows recycles pids), and exposes one
marker-checked tree kill. It is deliberately dependency-free so train_control,
export_control, dashdata and hf_datasets can all import it.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aether_env  # noqa: E402 - caches/temp stay on the repo drive (never C:)
aether_env.pin_caches()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")

# A pid is only ours if its command line still names one of these. The teacher and
# student llama-servers are deliberately NOT here: those are endpoints, not jobs,
# and they must never be killed by a job stop.
MARKERS = ("recursive_train.py", "train_student.py", "train_sweep.py",
           "hf_datasets.py", "export_model.py",
           "llama-imatrix.exe", "llama-quantize.exe")

# Records that *claim* a run is active. Each is (file, label, role hint).
RECORD_FILES = (
    ("train-active.json", "dashboard training", "trainer"),
    ("hf-train-active.json", "queue training", "loop"),
    ("export-active.json", "dashboard export", "export"),
)

_MATCH = ("recursive_train", "train_student", "train_sweep", "hf_datasets",
          "export_model", "llama-imatrix", "llama-quantize")

# The dashboard polls this every 5 s. An uncached Get-CimInstance over every process
# is not free, and under the disk load of a multi-GB pull it stalled the snapshot
# long enough that the page stopped answering (measured 2026-10-16). Display callers
# take the cached map; a kill forces a fresh one.
_PROC_TTL = 5.0
_PROC_CACHE = {"t": 0.0, "procs": None}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def processes(timeout=25, ttl=None):
    """{pid: {ppid, cmdline, name}} for every process that looks like ours.

    One CIM query instead of one per pid, cached for `_PROC_TTL` seconds (pass
    `ttl=0` for a fresh scan, as a kill does). The result is shared: treat it as
    read-only.
    """
    ttl = _PROC_TTL if ttl is None else ttl
    started = time.monotonic()
    if ttl and _PROC_CACHE["procs"] is not None \
            and (started - _PROC_CACHE["t"]) < ttl:
        return _PROC_CACHE["procs"]
    pattern = "|".join(_MATCH)
    script = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '%s' } "
              "| Select-Object ProcessId,ParentProcessId,Name,CommandLine "
              "| ConvertTo-Json -Compress" % pattern)
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                             capture_output=True, text=True, timeout=timeout)
        raw = out.stdout.strip()
        rows = json.loads(raw) if raw else []
    except Exception:  # noqa: BLE001 — a probe must never break the page
        return _PROC_CACHE["procs"] or {}
    if isinstance(rows, dict):
        rows = [rows]
    procs = {}
    for r in rows:
        try:
            pid = int(r.get("ProcessId"))
        except (TypeError, ValueError):
            continue
        procs[pid] = {"ppid": int(r.get("ParentProcessId") or 0),
                      "cmdline": r.get("CommandLine") or "",
                      "name": r.get("Name") or ""}
    _PROC_CACHE["t"] = time.monotonic()
    _PROC_CACHE["procs"] = procs
    return procs


def is_ours(pid, procs=None):
    """Is this pid one of our jobs right now (marker-checked, not just alive)?"""
    if not pid:
        return False
    procs = processes() if procs is None else procs
    info = procs.get(int(pid))
    if not info:
        return False
    return any(m in info["cmdline"] for m in MARKERS)


def role_of(cmdline):
    """Which stage a command line belongs to (for the panel and the kill order)."""
    c = cmdline or ""
    if "train_sweep.py" in c:
        return "sweep"
    if "recursive_train.py" in c:
        return "loop"
    if "train_student.py" in c:
        return "trainer"
    if "export_model.py" in c:
        return "export"
    if "llama-imatrix.exe" in c:
        return "imatrix"
    if "llama-quantize.exe" in c:
        return "quantize"
    if "hf_datasets.py" in c:
        return "queue-runner"
    return "unknown"


def children(pid, procs):
    """Descendant pids of `pid`, breadth-first (a stage runs as a small tree)."""
    out, frontier = [], [int(pid)]
    while frontier:
        cur = frontier.pop(0)
        for p, info in procs.items():
            if info["ppid"] == cur and p not in out and p != cur:
                out.append(p)
                frontier.append(p)
    return out


def _tag_of(path):
    base = os.path.basename(path or "")
    for pre, suf in (("loop-", ".json"), ("train-", ".json")):
        if base.startswith(pre) and base.endswith(suf):
            return base[len(pre):-len(suf)]
    return None


def _newest(patterns):
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(LOGS, pat))
    return max(files, key=os.path.getmtime) if files else None


def _from_records(procs):
    """Runs claimed by an active-run record (dashboard training, queue, export)."""
    runs = []
    for name, label, role in RECORD_FILES:
        rec = _read_json(os.path.join(LOGS, name))
        if not rec or not is_ours(rec.get("pid"), procs):
            continue          # a stale claim: keep the file, do not report a run
        pid = int(rec["pid"])
        cmdline = procs[pid]["cmdline"]
        runs.append({
            "pid": pid, "role": role_of(cmdline) if role == "loop" else role,
            "source": label, "record": name,
            "tag": rec.get("tag"), "repo": rec.get("repo"),
            "dataset": rec.get("dataset") or rec.get("corpus") or rec.get("adapter"),
            "corpus": rec.get("corpus"), "started_utc": rec.get("started_utc"),
            "log": rec.get("log"), "cmdline": cmdline,
            "tree": [pid] + children(pid, procs),
        })
    return runs


_LIVE_TRAINER_STATES = ("loading_tokenizer", "loading_model", "preparing_data",
                        "training", "saving")


def _from_state_files(procs):
    """Runs only the worker itself knows about (a sweep, or a loop iteration).

    A job launched by neither UI path still writes its own state, and that is
    enough to find and stop it.
    """
    runs, seen = [], set()
    for path in sorted(glob.glob(os.path.join(LOGS, "loop-*.json")),
                       key=os.path.getmtime, reverse=True):
        if os.path.basename(path) == "loop-selftest.json":
            continue
        st = _read_json(path) or {}
        it = (st.get("iterations") or [{}])[-1]
        pid = it.get("loop_pid") or it.get("pid")
        if not str(st.get("status") or "").startswith("running") \
                or not is_ours(pid, procs) or int(pid) in seen:
            continue
        seen.add(int(pid))
        runs.append({"pid": int(pid), "role": "loop", "source": "loop state",
                     "record": os.path.basename(path), "tag": st.get("tag"),
                     "repo": None, "dataset": it.get("corpus"),
                     "corpus": it.get("corpus"), "started_utc": it.get("started_utc"),
                     "log": None, "cmdline": procs[int(pid)]["cmdline"],
                     "tree": [int(pid)] + children(int(pid), procs)})
    for path in sorted(glob.glob(os.path.join(LOGS, "train-*.json")),
                       key=os.path.getmtime, reverse=True):
        st = _read_json(path) or {}
        pid = st.get("pid")
        if not is_ours(pid, procs) or int(pid) in seen \
                or str(st.get("status") or "").lower() not in _LIVE_TRAINER_STATES:
            continue
        seen.add(int(pid))
        runs.append({"pid": int(pid), "role": role_of(procs[int(pid)]["cmdline"]),
                     "source": "trainer state", "record": os.path.basename(path),
                     "tag": st.get("tag"), "repo": None, "dataset": st.get("dataset"),
                     "corpus": st.get("dataset"),
                     "started_utc": st.get("started_at") or st.get("started_utc"),
                     "log": None, "cmdline": procs[int(pid)]["cmdline"],
                     "tree": [int(pid)] + children(int(pid), procs)})
    return runs


def _queue_runner(procs):
    """The queue runner itself, if one is alive.

    Reported but never chosen as the run to stop: it is what moves on to the next
    dataset once a job ends.
    """
    for pid, info in procs.items():
        if "hf_datasets.py" in info["cmdline"] and "--queue-run" in info["cmdline"]:
            return {"pid": pid, "role": "queue-runner", "source": "process scan",
                    "record": None, "tag": None, "repo": None, "dataset": None,
                    "corpus": None, "started_utc": None, "log": None,
                    "cmdline": info["cmdline"],
                    "tree": [pid] + children(pid, procs)}
    return None


# Kill order: tearing down the outer stage takes its children with it.
_PRIORITY = ("sweep", "loop", "trainer", "export", "imatrix", "quantize")


def active(procs=None, with_state=True, fresh=False):
    """Everything that is running right now, whatever launched it.

    Returns {'primary', 'runs', 'runner', 'updated_utc'}; `primary` is the job a
    Stop should act on (never the queue runner), or None. `with_state=False` uses
    only the process table; `fresh=True` re-scans instead of using the 5 s cache
    (what a kill does before it decides).
    """
    procs = processes(ttl=0 if fresh else None) if procs is None else procs
    runs = _from_records(procs) + (_from_state_files(procs) if with_state else [])
    seen, uniq = set(), []
    for r in runs:
        if r["pid"] in seen:
            continue
        seen.add(r["pid"])
        uniq.append(r)
    uniq.sort(key=lambda r: _PRIORITY.index(r["role"])
              if r["role"] in _PRIORITY else len(_PRIORITY))
    primary = next((r for r in uniq if r["role"] in _PRIORITY), None)
    return {"primary": primary, "runs": uniq, "runner": _queue_runner(procs),
            "updated_utc": _now()}


def describe(state=None):
    """One line for the panel: what is running, and what Stop would end."""
    st = active() if state is None else state
    p = st.get("primary")
    if not p:
        text = ("no job running (the queue runner is waiting)"
                if st.get("runner") else "nothing running")
        return {"running": False, "text": text, "role": None, "pid": None,
                "runner_pid": (st.get("runner") or {}).get("pid")}
    bits = "%s pid=%d" % (p["role"], p["pid"])
    if p.get("tag"):
        bits += " tag=%s" % p["tag"]
    if p.get("repo"):
        bits += " repo=%s" % p["repo"]
    if p.get("dataset"):
        bits += " on %s" % os.path.basename(str(p["dataset"]))
    return {"running": True, "text": bits, "role": p["role"], "pid": p["pid"],
            "source": p["source"], "tag": p.get("tag"), "repo": p.get("repo"),
            "dataset": p.get("dataset"), "started_utc": p.get("started_utc"),
            "cmdline": p.get("cmdline"), "tree": p.get("tree"),
            "runner_pid": (st.get("runner") or {}).get("pid")}


def stop(rec, dry_run=False, procs=None):
    """Stop one run: marker-checked, whole subtree, then clear its claims.

    Refuses (and says so) when the pid no longer names one of our scripts — a
    recycled pid must never be killed because of a stale record.
    """
    procs = processes(ttl=0) if procs is None else procs
    pid = int(rec.get("pid") or 0)
    if not pid:
        return {"ok": False, "error": "no pid to stop"}
    info = procs.get(pid)
    if not info or not any(m in info["cmdline"] for m in MARKERS):
        return {"ok": False, "pid": pid,
                "error": "pid %d is not one of our jobs — refusing to kill" % pid}
    tree = rec.get("tree") or ([pid] + children(pid, procs))
    plan = {"ok": True, "pid": pid, "role": rec.get("role"), "tag": rec.get("tag"),
            "repo": rec.get("repo"), "tree": tree, "dry_run": bool(dry_run),
            "stopped": False}
    if dry_run:
        plan["note"] = ("dry run — would kill %d process(es): %s"
                        % (len(tree), ", ".join(str(p) for p in tree)))
        return plan
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, timeout=30)
    except Exception as e:  # noqa: BLE001
        plan["ok"] = False
        plan["error"] = "taskkill failed: %s" % e
        return plan
    plan["stopped"] = True
    plan["cleared"] = clear_claims(rec)
    return plan


def clear_claims(rec):
    """Delete the records that pointed at a stopped run (never the history).

    The trainer's own state and logs are left alone on purpose: they are the record
    of what happened, and dashdata's liveness probe turns a stale 'training' into
    'stopped — the trainer process is gone' from them.
    """
    gone = []
    pid = int(rec.get("pid") or 0)
    tree = set(int(x) for x in (rec.get("tree") or [pid]))
    for name, _label, _role in RECORD_FILES:
        path = os.path.join(LOGS, name)
        cur = _read_json(path)
        if not cur:
            continue
        if int(cur.get("pid") or 0) in tree:
            try:
                os.remove(path)
                gone.append(name)
            except OSError:
                pass
    return gone


def jobs(procs=None):
    """Every live job pid with its role (used by hf_datasets --stop and tests)."""
    procs = processes() if procs is None else procs
    out = []
    for pid, info in sorted(procs.items()):
        role = role_of(info["cmdline"])
        if role != "unknown":
            out.append({"pid": pid, "role": role, "ppid": info["ppid"],
                        "cmdline": info["cmdline"]})
    return out


def selftest():
    """Offline checks of the resolver's judgement (no kill is ever performed)."""
    ok = True

    def check(good, label, extra=""):
        nonlocal ok
        ok = ok and bool(good)
        print("[selftest] %s %s%s" % ("ok  " if good else "FAIL", label,
                                      (" — %s" % extra) if extra else ""))

    procs = processes()
    check(isinstance(procs, dict), "process scan returns a mapping (%d match ours)"
          % len(procs))

    foreign = {os.getpid(): {"ppid": 0, "cmdline": sys.executable, "name": "python"}}
    check(not is_ours(os.getpid(), foreign),
          "a pid whose command line has no marker is not ours")

    check(role_of('python -u "G:\\x\\scripts\\recursive_train.py" --tag t') == "loop",
          "recursive_train.py -> loop")
    check(role_of('python -u "G:\\x\\scripts\\train_student.py" --tag t') == "trainer",
          "train_student.py -> trainer")
    check(role_of('python -u "G:\\x\\scripts\\export_model.py" --hub') == "export",
          "export_model.py -> export")
    check(role_of('python scripts\\hf_datasets.py --queue-run') == "queue-runner",
          "hf_datasets --queue-run -> runner")
    check(role_of("llama-server.exe --port 8082 --alias jadis-a") == "unknown",
          "llama-server is not a job (a stop can never kill an endpoint)")

    refused = stop({"pid": os.getpid(), "role": "trainer"}, dry_run=True, procs=foreign)
    check(refused.get("ok") is False and "refusing" in (refused.get("error") or ""),
          "stopping a non-marker pid is refused", refused.get("error", ""))

    fake = {10: {"ppid": 1, "cmdline": "recursive_train.py", "name": "python"},
            11: {"ppid": 10, "cmdline": "train_student.py", "name": "python"},
            12: {"ppid": 11, "cmdline": "python.exe", "name": "python"}}
    check(children(10, fake) == [11, 12], "the tree walk reaches grandchildren")
    ours = {10: {"ppid": 1, "cmdline": "recursive_train.py --tag t", "name": "python"}}
    dry = stop({"pid": 10, "role": "loop", "tag": "t"}, dry_run=True, procs=ours)
    check(dry.get("ok") and dry.get("stopped") is False and dry.get("dry_run"),
          "a dry run reports the plan and kills nothing")
    check(_from_records({}) == [], "no records, no phantom runs")

    d = describe(active(procs))
    if d["running"]:
        print("[selftest] info live: %s (source: %s)" % (d["text"], d.get("source")))
    else:
        print("[selftest] info nothing is running right now")
    print("[selftest] %s" % ("ALL OK" if ok else "FAILURES"))
    return 0 if ok else 1


def main(argv):
    if "--selftest" in argv:
        return selftest()
    if "--stop" in argv:
        st = active()
        if not st.get("primary"):
            print(json.dumps({"ok": True, "stopped": False,
                              "note": "nothing running"}, indent=2))
            return 0
        print(json.dumps(stop(st["primary"], dry_run="--dry-run" in argv), indent=2))
        return 0
    st = active()
    print(json.dumps({"updated_utc": st["updated_utc"],
                      "describe": describe(st),
                      "runs": [{k: v for k, v in r.items() if k != "cmdline"}
                               for r in st["runs"]],
                      "jobs": jobs(),
                      "runner": (st["runner"] or {}).get("pid")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
