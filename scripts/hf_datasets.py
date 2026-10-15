#!/usr/bin/env python3
"""Hugging Face dataset pipeline: link -> pull to G: -> convert -> verify ->
fit -> train -> verify learned -> delete -> next dataset.

One dataset at a time, whole, on disk, then removed. The G: drive holds the
download; `cache/datasets/<org>--<name>/` is gitignored like the rest of cache/.

State machine per dataset (recorded in logs/hf-datasets.json):

    added -> pulled -> converted -> verified -> fitted -> trained
          -> learned_verified -> dropped

`learned_verified` is set by the review gate (the teacher grading the trained
student), which is what licenses `--drop`: the point of pulling a dataset to
disk is to learn from it, so the local copy is only released once an iteration
has passed the gate on it.

Usage:
  python scripts/hf_datasets.py --add <hub url or org/name>
  python scripts/hf_datasets.py --list
  python scripts/hf_datasets.py --pull org/name --split full
  python scripts/hf_datasets.py --convert org/name --limit 2000 --tag fable10k
  python scripts/hf_datasets.py --verify org/name --max-len 1536
  python scripts/hf_datasets.py --fit org/name --max-len 1536
  python scripts/hf_datasets.py --storage
  python scripts/hf_datasets.py --drop org/name --yes
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
PY = os.path.join(ROOT, "venvs", "ml", "Scripts", "python.exe")
DEST_ROOT = os.path.join(ROOT, "cache", "datasets")
LOGS = os.path.join(ROOT, "logs")
REGISTRY = os.path.join(LOGS, "hf-datasets.json")
DATASETS = os.path.join(ROOT, "datasets")
HF_API = "https://huggingface.co/api/datasets/%s"

STATES = ["added", "pulled", "converted", "verified", "fitted", "training",
          "trained", "learned_verified", "not_trainable", "dropped", "error"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_repo(value):
    """Accept a hub URL, a `datasets/` URL, or a bare `org/name` slug.

    The UI takes whatever the user pastes from the browser address bar. Anything
    that is not a plain slug is reduced to one, so this works:
      --pull https://huggingface.co/datasets/org/name/tree/main
    """
    s = (value or "").strip()
    if not s:
        raise SystemExit("empty dataset reference")
    s = re.sub(r"^[a-z]+://", "", s, flags=re.I)          # scheme
    s = re.sub(r"^(www\.)?huggingface\.co/", "", s, flags=re.I)
    s = re.sub(r"^(api/)?datasets/", "", s, flags=re.I)
    s = s.split("?", 1)[0].split("#", 1)[0]
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 3 and parts[2] == "tree":            # org/name/tree/main/...
        parts = parts[:2]
    if not parts:
        raise SystemExit("cannot parse a dataset id from %r" % value)
    if len(parts) == 1:
        parts = ["", parts[0]]
    repo = "/".join(parts[:2])
    if not re.match(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", repo):
        raise SystemExit("that does not look like a dataset id: %r" % repo)
    return repo


def slug_tag(repo, limit=48):
    """A short, filesystem-safe tag for datasets/<tag>.jsonl."""
    name = repo.split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return name[:limit].rstrip("-")


def api_info(repo, timeout=30):
    """Dataset metadata from the hub API (public repos need no auth)."""
    req = urllib.request.Request(
        HF_API % repo, headers={"User-Agent": "aether-dataset-pipeline"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def split_files(info, split):
    """File paths to download for a split name ('full', 'train', 'all').

    Cards may list several configs (gsm8k ships `main/` and `socratic/` under the
    same split name), so the FIRST config that matches wins — pulling both would
    silently double the corpus with a different recipe.
    """
    configs = (info.get("cardData", {}) or {}).get("configs", []) or []
    if split in ("all", "*"):
        seen, out = set(), []
        for cfg in configs:
            for f in cfg.get("data_files", []) or []:
                if f.get("path") and f["path"] not in seen:
                    seen.add(f["path"])
                    out.append(f["path"])
        if not out:
            for s in info.get("siblings", []) or []:
                if s.get("rfilename", "").endswith((".jsonl", ".parquet")):
                    out.append(s["rfilename"])
        return out
    for cfg in configs:
        out = []
        for f in cfg.get("data_files", []) or []:
            path, name = f.get("path"), f.get("split")
            if not path:
                continue
            if split in (name, os.path.basename(os.path.dirname(path)), path):
                out.append(path)
        if out:
            return out
    return []


def expand_files(root, patterns):
    """Resolve card patterns (`main/train-*`, `**/*.parquet`) to real files."""
    import glob as _glob
    out, seen = [], set()
    for p in patterns or []:
        hits = _glob.glob(os.path.join(root, p), recursive=True)
        if not hits:
            literal = os.path.join(root, p)
            hits = [literal] if os.path.isfile(literal) else []
        for f in sorted(hits):
            if os.path.isfile(f) and f not in seen:
                seen.add(f)
                out.append(f)
    return out


def count_rows(path):
    """Rows in a local file (JSONL by line count, Parquet via the datasets lib)."""
    if path.lower().endswith(".parquet"):
        from datasets import Dataset
        return Dataset.from_parquet(path).num_rows
    n = 0
    with open(path, "rb") as fh:
        for _ in fh:
            n += 1
    return n


def dir_bytes(path):
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return "%d B" % n if unit == "B" else "%.2f %s" % (n, unit)
        n /= 1024.0
    return "%0.2f TB" % n


def load_registry():
    try:
        with open(REGISTRY, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("datasets"), list):
            return data
    except (OSError, ValueError):
        pass
    return {"datasets": []}


def save_registry(data):
    os.makedirs(LOGS, exist_ok=True)
    tmp = REGISTRY + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, REGISTRY)


def entry_of(data, repo):
    for e in data["datasets"]:
        if e["id"] == repo:
            return e
    return None


def upsert(data, repo, note=None, **kw):
    e = entry_of(data, repo)
    if e is None:
        e = {"id": repo, "added_utc": now(), "state": "added", "history": []}
        data["datasets"].append(e)
    e.update(kw)
    if note:
        e["note"] = note          # the dashboard's "why" column reads this
        e["history"].append({"utc": now(), "note": note})
    return e


def local_dir_for(repo):
    return os.path.join(DEST_ROOT, repo.replace("/", "--"))


def disk_free(path):
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def _official_rows(info):
    """Split -> declared path, as the dataset card describes them."""
    out = {}
    for cfg in (info.get("cardData", {}) or {}).get("configs", []) or []:
        for f in (cfg.get("data_files") or []):
            out[f.get("split") or f.get("path")] = f.get("path")
    return out


def pull(repo, split="full", log=print):
    """Download the whole requested split into cache/datasets/<org>--<name>/."""
    try:
        info = api_info(repo)
    except Exception as e:  # noqa: BLE001 — network/auth problems must be visible
        raise SystemExit("cannot read dataset info for %s: %s" % (repo, e))
    files = split_files(info, split)
    if not files:
        files = ["%s/train.jsonl" % split]
    dest = local_dir_for(repo)
    os.makedirs(dest, exist_ok=True)
    free = disk_free(dest)
    want = sum(s.get("size") or 0 for s in (info.get("siblings") or []))
    if free is not None and want and free < want * 1.2:
        raise SystemExit("not enough space on the dataset drive: %s free, "
                         "~%s needed" % (human(free), human(want)))
    log("[hf] %s -> %s" % (repo, os.path.relpath(dest, ROOT)))
    log("[hf] files: %s" % ", ".join(files))
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=repo, repo_type="dataset",
                             local_dir=dest, allow_patterns=files,
                             max_workers=4)
    # Card patterns are globs (`main/train-*`): resolve them to the files that
    # actually landed so counting, converting and dropping all name real paths.
    real = [os.path.relpath(f, path) for f in expand_files(path, files)]
    if not real:
        real = files
    total = dir_bytes(path)
    rows = 0
    for f in real:
        p = os.path.join(path, f)
        if os.path.isfile(p):
            n = count_rows(p)
            rows += n
            log("[hf]   %s: %d rows, %s" % (f, n, human(os.path.getsize(p))))
    data = load_registry()
    e = upsert(data, repo, state="pulled", revision=info.get("sha"),
               pulled_split=split, pulled_files=real, local_dir=path,
               bytes=total, rows=rows,
               license=(info.get("cardData", {}) or {}).get("license"),
               likes=info.get("likes"), downloads=info.get("downloads"),
               official_rows=_official_rows(info),
               note="pulled split=%s rows=%d bytes=%d" % (split, rows, total))
    save_registry(data)
    log("[hf] pulled %s rows, %s on disk" % (rows, human(total)))
    return e


def convert(repo, tag=None, limit=2000, map_="chat", log=print,
            max_prompt_chars=4000, max_completion_chars=4000,
            max_reasoning_chars=2000):
    """Map the local copy into AETHER's training schema (datasets/<tag>.jsonl).

    Reads the pulled copy, never the hub: the file is on the dataset drive, so
    re-streaming it would waste exactly the bandwidth the pull step paid for.
    """
    data = load_registry()
    e = entry_of(data, repo) or upsert(data, repo)
    tag = tag or slug_tag(repo)
    local = e.get("local_dir") or local_dir_for(repo)
    if not os.path.isdir(local):
        raise SystemExit("no local copy for %s — run --pull first" % repo)
    cmd = [PY, "-u", os.path.join(SCRIPTS, "ingest_hf_dataset.py"),
           "--repo", repo, "--map", map_, "--tag", tag,
           "--from-local", local, "--limit", str(limit),
           "--max-prompt-chars", str(max_prompt_chars),
           "--max-completion-chars", str(max_completion_chars),
           "--max-reasoning-chars", str(max_reasoning_chars)]
    # Convert exactly the files the pull fetched. With --split all several
    # splits land on disk (5k/ and 10k/ here) and only the full one should be
    # mapped; naming the files keeps that decision explicit and auditable.
    files = e.get("pulled_files") or []
    if len(files) > 1:
        big = max(files, key=lambda f: os.path.getsize(os.path.join(local, f))
                  if os.path.isfile(os.path.join(local, f)) else 0)
        files = [big]
    if files:
        cmd += ["--from-local-files", ",".join(files)]
        cmd += ["--split", e.get("pulled_split") or "train"]
    log("[hf] $ %s" % " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    out = os.path.join(DATASETS, tag + ".jsonl")
    if proc.returncode != 0 or not os.path.isfile(out):
        upsert(data, repo, state="error",
               note="convert failed rc=%d" % proc.returncode)
        save_registry(data)
        raise SystemExit("convert failed (rc=%d)" % proc.returncode)
    rows = count_rows(out)
    e = upsert(data, repo, state="converted", tag=tag,
               corpus=os.path.relpath(out, ROOT), corpus_rows=rows,
               corpus_bytes=os.path.getsize(out),
               note="converted -> %s (%d rows)" % (os.path.basename(out), rows))
    save_registry(data)
    log("[hf] corpus %s: %d rows" % (os.path.relpath(out, ROOT), rows))
    return e


def verify(repo, max_len=1536, anchor="datasets/phase6_pilot1.jsonl", log=print):
    """Measure the corpus with the trainer's own renderer (verify_dataset.py)."""
    data = load_registry()
    e = entry_of(data, repo)
    if not e or not e.get("corpus"):
        raise SystemExit("nothing converted for %s yet" % repo)
    report = os.path.join(LOGS, "verify-dataset-%s.json" % e.get("tag", "x"))
    cmd = [PY, "-u", os.path.join(SCRIPTS, "verify_dataset.py"),
           "--jsonl", e["corpus"], "--max-len", str(max_len),
           "--json", report]
    anchor_path = os.path.join(ROOT, anchor)
    if os.path.isfile(anchor_path):
        cmd += ["--overlap", anchor_path]
    log("[hf] $ %s" % " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    try:
        with open(report, encoding="utf-8") as fh:
            rep = json.load(fh)
    except (OSError, ValueError):
        raise SystemExit("verify produced no report (rc=%d)" % proc.returncode)
    t = rep.get("tokens", {})
    cnt = rep.get("counts") or (t.get("tokens") or {})
    dup = rep.get("duplicates") or {}
    over = rep.get("overlap") or {}
    health = {"rows": t.get("rows"), "max_len": t.get("max_len"),
              "classes": t.get("classes"),
              "answer_cut": (t.get("classes") or {}).get("answer_cut"),
              "terminator_pct": t.get("terminator_pct"),
              "answer_token_share": cnt.get("answer_loss_share"),
              "answer_tokens_kept_pct": cnt.get("completion_kept_pct"),
              "duplicate_groups": dup.get("duplicate_groups"),
              "anchor_overlap": over.get("shared"),
              "len_tokens": t.get("len_tokens")}
    e = upsert(data, repo, state="verified", health=health,
               verify_report=os.path.relpath(report, ROOT),
               note="verified at max_len=%d" % max_len)
    save_registry(data)
    log("[hf] health: %s" % json.dumps(health))
    return e


def fit(repo, max_len=1536, anchor="datasets/phase6_pilot1.jsonl", log=print,
        max_per_prompt=4):
    """Fit the corpus to the budget (fit_corpus.py) -> datasets/<tag>-fit.jsonl."""
    data = load_registry()
    e = entry_of(data, repo)
    if not e or not e.get("corpus"):
        raise SystemExit("nothing converted for %s yet" % repo)
    out = e["corpus"].replace(".jsonl", "-fit.jsonl")
    cmd = [PY, "-u", os.path.join(SCRIPTS, "fit_corpus.py"),
           "--in", e["corpus"], "--out", out, "--max-len", str(max_len),
           "--max-per-prompt", str(max_per_prompt)]
    anchor_path = os.path.join(ROOT, anchor)
    if os.path.isfile(anchor_path):
        cmd += ["--holdout", anchor_path]
    log("[hf] $ %s" % " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0 or not os.path.isfile(out):
        upsert(data, repo, state="error",
               note="fit failed rc=%d" % proc.returncode)
        save_registry(data)
        raise SystemExit("fit failed (rc=%d)" % proc.returncode)
    prov = {}
    try:
        with open(out + ".filter.json", encoding="utf-8") as fh:
            prov = json.load(fh)
    except (OSError, ValueError):
        pass
    rows = count_rows(out)
    e = upsert(data, repo, state="fitted", fitted=os.path.relpath(out, ROOT),
               fitted_rows=rows, dropped=(prov.get("dropped") or {}),
               trimmed_rows=prov.get("trimmed_rows"),
               note="fitted -> %s (%d rows)" % (os.path.basename(out), rows))
    save_registry(data)
    log("[hf] fitted %s: %d rows (dropped %s)"
        % (os.path.relpath(out, ROOT), rows, prov.get("dropped")))
    return e


def rewrite(repo, tag=None, limit=200, max_len=1536, log=print):
    """Teacher-distil a trace dataset into lessons, then verify + fit them.

    For datasets that hold no prompt->answer pairs (agentic traces), the teacher
    reads each trace digest and writes a self-contained lesson instead. The
    lessons become the corpus, so the usual verify/fit stages still apply.
    """
    data = load_registry()
    e = entry_of(data, repo) or upsert(data, repo)
    local = e.get("local_dir") or local_dir_for(repo)
    if not os.path.isdir(local):
        raise SystemExit("no local copy for %s — run --pull first" % repo)
    tag = tag or (slug_tag(repo)[:32] + "-lessons")
    files = e.get("pulled_files") or []
    if len(files) > 1:
        files = [max(files, key=lambda f: os.path.getsize(os.path.join(local, f))
                     if os.path.isfile(os.path.join(local, f)) else 0)]
    cmd = [PY, "-u", os.path.join(SCRIPTS, "rewrite_traces.py"),
           "--repo", repo, "--from-local", local, "--tag", tag,
           "--limit", str(limit)]
    if files:
        cmd += ["--from-local-files", ",".join(files)]
    log("[hf] $ %s" % " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    out = os.path.join(DATASETS, tag + ".jsonl")
    if proc.returncode != 0 or not os.path.isfile(out) or not count_rows(out):
        upsert(data, repo, state="not_trainable",
               note="rewrite produced no lessons (rc=%d) — see logs/rewrite-%s.log"
                    % (proc.returncode, tag))
        save_registry(data)
        raise SystemExit("rewrite produced no lessons for %s" % repo)
    rows = count_rows(out)
    e = upsert(data, repo, state="converted", tag=tag, rewritten=True,
               corpus=os.path.relpath(out, ROOT), corpus_rows=rows,
               corpus_bytes=os.path.getsize(out),
               note="teacher rewrite -> %s (%d lessons)" % (os.path.basename(out),
                                                            rows))
    save_registry(data)
    log("[hf] %d lessons in %s" % (rows, os.path.relpath(out, ROOT)))
    log("[hf] === %s: verify (lessons) ===" % repo)
    verify(repo, max_len=max_len, log=log)
    log("[hf] === %s: fit (lessons) ===" % repo)
    return fit(repo, max_len=max_len, log=log)


def mark(repo, state, **kw):
    """Set a dataset's state (the loop/dashboard call this when a stage ends)."""
    data = load_registry()
    e = upsert(data, repo, state=state, **kw)
    save_registry(data)
    return e


def drop(repo, force=False, log=print):
    """Delete the local copy (only once an iteration has learned from it)."""
    data = load_registry()
    e = entry_of(data, repo)
    if e is None:
        raise SystemExit("unknown dataset %s" % repo)
    if not force and e.get("state") != "learned_verified":
        raise SystemExit(
            "refusing to drop %s: state=%r — a dataset is released only after an "
            "iteration passes the gate on it (set learned_verified), or pass "
            "--force" % (repo, e.get("state")))
    path = e.get("local_dir") or local_dir_for(repo)
    freed = dir_bytes(path) if os.path.isdir(path) else 0
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        log("[hf] nothing on disk at %s" % path)
    upsert(data, repo, state="dropped", freed_bytes=freed, local_dir=None,
           dropped_utc=now(),
           note="dropped local copy, freed %s" % human(freed))
    save_registry(data)
    log("[hf] dropped %s, freed %s" % (repo, human(freed)))
    return freed


def storage():
    """Per-dataset bytes on disk + total + free space on the dataset drive."""
    data = load_registry()
    items, total = [], 0
    for e in data["datasets"]:
        path = e.get("local_dir") or local_dir_for(e["id"])
        present = os.path.isdir(path)
        n = dir_bytes(path) if present else 0
        total += n
        items.append({"id": e["id"], "state": e.get("state"), "present": present,
                      "bytes": n, "human": human(n), "tag": e.get("tag"),
                      "rows": e.get("rows") or e.get("fitted_rows"),
                      "path": os.path.relpath(path, ROOT) if present else None})
    orphans = []
    if os.path.isdir(DEST_ROOT):
        known = {os.path.normcase(i["path"] or "") for i in items}
        for d in sorted(os.listdir(DEST_ROOT)):
            p = os.path.join(DEST_ROOT, d)
            rel = os.path.relpath(p, ROOT)
            if os.path.isdir(p) and os.path.normcase(rel) not in known:
                n = dir_bytes(p)
                total += n
                orphans.append({"dir": d, "bytes": n, "human": human(n)})
    free = disk_free(DEST_ROOT) or 0
    return {"items": items, "orphans": orphans, "total_bytes": total,
            "total_human": human(total), "free_bytes": free,
            "free_human": human(free), "root": os.path.relpath(DEST_ROOT, ROOT)}


def train(repo, tag=None, iteration=1, max_steps=-1, max_len=1536, sample=40,
          min_pass=0.6, log=print):
    """Launch one recursive-loop iteration on this dataset's fitted corpus.

    Returns (pid, tag). The loop trains -> exports -> serves -> reviews -> gates,
    and its state file is what `learned_verified()` reads.
    """
    data = load_registry()
    e = entry_of(data, repo)
    if e is None:
        raise SystemExit("unknown dataset %s" % repo)
    corpus = e.get("fitted") or e.get("corpus")
    if not corpus:
        raise SystemExit("nothing converted for %s yet" % repo)
    tag = tag or e.get("tag") or slug_tag(repo)
    cmd = [PY, "-u", os.path.join(SCRIPTS, "recursive_train.py"),
           "--iteration", str(iteration), "--tag", tag,
           "--corpus", corpus, "--max-len", str(max_len),
           "--sample", str(sample), "--min-pass", str(min_pass)]
    if max_steps and max_steps > 0:
        cmd += ["--max-steps", str(max_steps)]
    os.makedirs(LOGS, exist_ok=True)
    log_path = os.path.join(LOGS, "hf-train-%s.log" % tag)
    env = dict(os.environ)
    env.setdefault("AETHER_LLAMA_CPP_SRC",
                   os.path.join(os.path.dirname(ROOT), "Personality", "runtime",
                                "llama.cpp"))
    log("[hf] $ %s" % " ".join(cmd))
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf,
                                stderr=subprocess.STDOUT,
                                creationflags=0x00000200)  # detached
    active = {"pid": proc.pid, "repo": repo, "tag": tag, "corpus": corpus,
              "started_utc": now(), "log": os.path.relpath(log_path, ROOT)}
    with open(os.path.join(LOGS, "hf-train-active.json"), "w",
              encoding="utf-8") as fh:
        json.dump(active, fh, indent=1)
    upsert(data, repo, state="training", tag=tag, loop_pid=proc.pid,
           loop_log=active["log"],
           note="training tag=%s pid=%d corpus=%s" % (tag, proc.pid, corpus))
    save_registry(data)
    log("[hf] training pid=%d tag=%s (log %s)" % (proc.pid, tag, active["log"]))
    return proc.pid, tag


def learned_verified(repo, tag=None, log=print):
    """Mark learned_verified if the loop's gate passed on this dataset.

    This is the licence to delete the local copy: the teacher graded the trained
    student on held-out rows and the iteration cleared its floor without
    regressing. Returns (ok, detail).
    """
    data = load_registry()
    e = entry_of(data, repo)
    if e is None:
        raise SystemExit("unknown dataset %s" % repo)
    tag = tag or e.get("tag") or slug_tag(repo)
    path = os.path.join(LOGS, "loop-%s.json" % tag)
    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        return False, {"error": "no loop state at %s" % os.path.relpath(path, ROOT)}
    iters = state.get("iterations") or []
    best = None
    for it in iters:
        g = it.get("gate") or {}
        if g.get("passed"):
            best = it
    if best is None:
        marks = [(it.get("iteration"), (it.get("gate") or {}).get("mean_pass"),
                  (it.get("gate") or {}).get("floor"))
                 for it in iters]
        return False, {"error": "no iteration passed the gate", "iterations": marks}
    upsert(data, repo, state="learned_verified",
           learned_verified_utc=now(), learned_iteration=best.get("iteration"),
           gate=best.get("gate"),
           note="gate passed on iteration %s (mean_pass=%s)"
                % (best.get("iteration"), (best.get("gate") or {}).get("mean_pass")))
    save_registry(data)
    log("[hf] %s: learned verified on iteration %s"
        % (repo, best.get("iteration")))
    return True, {"iteration": best.get("iteration"), "gate": best.get("gate")}


def advance(repo, split="full", limit=2000, map_="chat", max_len=1536,
            tag=None, log=print):
    """Data stages only: pull -> convert -> verify -> fit. Returns the registry entry."""
    log("[hf] === %s: pull ===" % repo)
    pull(repo, split=split, log=log)
    log("[hf] === %s: convert ===" % repo)
    convert(repo, tag=tag, limit=limit, map_=map_, log=log)
    log("[hf] === %s: verify ===" % repo)
    verify(repo, max_len=max_len, log=log)
    log("[hf] === %s: fit ===" % repo)
    return fit(repo, max_len=max_len, log=log)


def selftest(log=print):
    """Offline checks for the parsing/registry/gating logic that matter.

    Catches the mistakes that would otherwise surface as a wrong download or a
    dataset deleted too early: URL parsing, split selection, registry round-trip,
    and the drop gate refusing an unlearned dataset.
    """
    import tempfile
    ok = True
    cases = [
        ("MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x",
         "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x"),
        ("https://huggingface.co/datasets/MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x",
         "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x"),
        ("https://huggingface.co/datasets/MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x/tree/main",
         "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x"),
        ("huggingface.co/datasets/FineEnvs/openai-math",
         "FineEnvs/openai-math"),
        ("api/datasets/org/name?x=1", "org/name"),
    ]
    for raw, want in cases:
        got = resolve_repo(raw)
        good = got == want
        ok = ok and good
        log("[selftest] %s resolve_repo(%r) -> %s%s"
            % ("ok  " if good else "FAIL", raw, got,
               "" if good else " (want %s)" % want))
    tag = slug_tag("MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x")
    good = tag == "fable-5-1-max-reasoning-filtered-10000x"
    ok = ok and good
    log("[selftest] %s slug_tag -> %s" % ("ok  " if good else "FAIL", tag))

    # registry round-trip in a temp file (never touches the live registry)
    global REGISTRY, DEST_ROOT
    keep_reg, keep_dest = REGISTRY, DEST_ROOT
    tmp = tempfile.mkdtemp(prefix="aether-hf-selftest-")
    try:
        REGISTRY = os.path.join(tmp, "hf-datasets.json")
        DEST_ROOT = os.path.join(tmp, "datasets")
        data = load_registry()
        upsert(data, "org/one", state="pulled", bytes=123, rows=4)
        save_registry(data)
        again = load_registry()
        good = len(again["datasets"]) == 1 and again["datasets"][0]["bytes"] == 123
        ok = ok and good
        log("[selftest] %s registry round-trip" % ("ok  " if good else "FAIL"))
        good = queue_pending(again) and queue_pending(again)[0]["id"] == "org/one"
        ok = ok and bool(good)
        log("[selftest] %s pulled dataset is queued" % ("ok  " if good else "FAIL"))
        # a dataset that has not been learned from must not be deletable
        os.makedirs(os.path.join(DEST_ROOT, "org--one"), exist_ok=True)
        with open(os.path.join(DEST_ROOT, "org--one", "x.jsonl"), "w") as fh:
            fh.write("x\n")
        try:
            drop("org/one", log=lambda *a: None)
            good = False
        except SystemExit:
            good = True
        ok = ok and good
        log("[selftest] %s drop refuses an unlearned dataset"
            % ("ok  " if good else "FAIL"))
        mark("org/one", "learned_verified")
        freed = drop("org/one", log=lambda *a: None)
        good = freed > 0 and not os.path.isdir(os.path.join(DEST_ROOT, "org--one"))
        ok = ok and good
        log("[selftest] %s drop releases a learned dataset (freed %d bytes)"
            % ("ok  " if good else "FAIL", freed))
    finally:
        REGISTRY, DEST_ROOT = keep_reg, keep_dest
        shutil.rmtree(tmp, ignore_errors=True)
    log("[selftest] %s" % ("ALL OK" if ok else "FAILURES"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="HF dataset pipeline (G:-resident)")
    ap.add_argument("--add", metavar="REF", help="register a dataset (URL or org/name)")
    ap.add_argument("--pull", metavar="REF", help="download the whole split to G:")
    ap.add_argument("--convert", metavar="REF", help="map the local copy to our schema")
    ap.add_argument("--verify", metavar="REF", help="measure the corpus (tokens)")
    ap.add_argument("--fit", metavar="REF", help="fit the corpus to --max-len")
    ap.add_argument("--rewrite", metavar="REF",
                    help="teacher-distil traces into lessons (for datasets with "
                         "no prompt/answer pairs), then verify + fit")
    ap.add_argument("--train", metavar="REF", help="launch a loop iteration")
    ap.add_argument("--learned-verified", metavar="REF",
                    help="mark learned_verified if the gate passed")
    ap.add_argument("--drop", metavar="REF", help="delete the local copy")
    ap.add_argument("--advance", metavar="REF", help="pull+convert+verify+fit")
    ap.add_argument("--queue-run", action="store_true",
                    help="process every queued dataset end to end")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--storage", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="offline checks (URL parsing, registry, drop gating)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--split", default="full", help="full|train|all (dataset card)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--map", dest="map_", default="chat",
                    help="schema mapping: chat | flat")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--min-pass", type=float, default=0.6)
    ap.add_argument("--iteration", type=int, default=0, help="0 = auto-continue")
    ap.add_argument("--yes", action="store_true", help="confirm --drop")
    ap.add_argument("--after-pid", type=int, default=0,
                    help="wait for this PID to exit first — chains a queue "
                         "behind a loop that is still running")
    ap.add_argument("--force", action="store_true",
                    help="--drop even without a verified iteration")
    a = ap.parse_args()

    def quiet(*_args, **_kw):
        """--json must stay parseable, so silence the human log lines."""
        pass
    log = quiet if a.json else print

    if a.after_pid:
        print("[hf] waiting for pid %d to exit before starting (the review/loop "
              "that owns the student cards)" % a.after_pid, flush=True)
        wait_for_pid(a.after_pid, poll=30, log=print)
        print("[hf] pid %d is gone - proceeding" % a.after_pid, flush=True)

    if a.selftest:
        return selftest(log=print)
    if a.add:
        repo = resolve_repo(a.add)
        data = load_registry()
        e = upsert(data, repo, note="linked from %s" % a.add)
        save_registry(data)
        if a.json:
            print(json.dumps({"ok": True, "dataset": e}, indent=1))
        else:
            print("[hf] registered %s (state=%s)" % (repo, e["state"]))
        return 0
    if a.list:
        data = load_registry()
        if a.json:
            print(json.dumps(data, indent=1))
        elif not data["datasets"]:
            print("[hf] no datasets linked yet")
        else:
            for e in data["datasets"]:
                print("%-52s %-16s %s rows  %s"
                      % (e["id"], e.get("state"), e.get("rows") or "-",
                         human(e.get("bytes") or 0)))
        return 0
    if a.storage:
        s = storage()
        if a.json:
            print(json.dumps(s, indent=1))
        else:
            print("[hf] dataset drive %s: %s held, %s free"
                  % (s["root"], s["total_human"], s["free_human"]))
            for i in s["items"]:
                print("  %-52s %-16s %-9s %s"
                      % (i["id"], i["state"], i["human"],
                         "on disk" if i["present"] else "-"))
            for o in s["orphans"]:
                print("  %-52s %-16s %-9s %s"
                      % (o["dir"], "unregistered", o["human"], "on disk"))
        return 0
    if a.queue_run:
        return queue_run(split=a.split, limit=a.limit, map_=a.map_,
                         max_len=a.max_len, max_steps=a.max_steps,
                         sample=a.sample, min_pass=a.min_pass, log=print)

    repo = None
    for ref in (a.pull, a.convert, a.verify, a.fit, a.train, a.drop,
                a.advance, a.learned_verified, a.rewrite):
        if ref:
            repo = resolve_repo(ref)
    if repo is None:
        ap.error("pick an action: --add/--list/--pull/--convert/--verify/--fit/"
                 "--train/--advance/--drop/--storage/--queue-run")

    if a.pull:
        e = pull(repo, split=a.split, log=log)
    elif a.convert:
        e = convert(repo, tag=a.tag, limit=a.limit, map_=a.map_, log=log)
    elif a.verify:
        e = verify(repo, max_len=a.max_len, log=log)
    elif a.fit:
        e = fit(repo, max_len=a.max_len, log=log)
    elif a.rewrite:
        e = rewrite(repo, tag=a.tag, limit=a.limit, max_len=a.max_len, log=log)
    elif a.train:
        tag = a.tag or (entry_of(load_registry(), repo) or {}).get("tag") \
            or slug_tag(repo)
        pid, tag = train(repo, tag=tag,
                         iteration=a.iteration or next_iteration(tag),
                         max_steps=a.max_steps, max_len=a.max_len,
                         sample=a.sample, min_pass=a.min_pass, log=log)
        if a.json:
            print(json.dumps({"ok": True, "pid": pid, "tag": tag}, indent=1))
        return 0
    elif a.learned_verified:
        ok, detail = learned_verified(repo, tag=a.tag, log=log)
        if a.json:
            print(json.dumps({"ok": ok, "detail": detail}, indent=1))
        return 0 if ok else 2
    elif a.drop:
        if not a.yes:
            raise SystemExit("--drop needs --yes (it deletes the local copy)")
        freed = drop(repo, force=a.force, log=log)
        if a.json:
            print(json.dumps({"ok": True, "freed_bytes": freed}, indent=1))
        return 0
    else:
        e = advance(repo, split=a.split, limit=a.limit, map_=a.map_,
                    max_len=a.max_len, tag=a.tag, log=log)
    if a.json:
        print(json.dumps({"ok": True, "dataset": e}, indent=1))
    return 0


JOBS_PATH = os.path.join(LOGS, "hf-jobs.json")
JOBS = ["add", "pull", "convert", "verify", "fit", "rewrite", "train",
        "learned_verified", "drop", "advance", "queue_run"]


def _load_jobs():
    try:
        with open(JOBS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
    except (OSError, ValueError):
        pass
    return []


def _save_jobs(jobs):
    os.makedirs(LOGS, exist_ok=True)
    tmp = JOBS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(jobs[-40:], fh, indent=1)   # keep the last 40
    os.replace(tmp, JOBS_PATH)


def pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % int(pid), "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
        return str(pid) in out
    except Exception:  # noqa: BLE001
        return False


def spawn(action, repo=None, **kw):
    """Launch one pipeline action as a detached job (what the dashboard calls).

    Runs this same script as a subprocess so the UI path and the CLI path are
    literally the same code, and so a slow pull cannot block the web handler.
    """
    if action not in JOBS:
        raise SystemExit("unknown action %r" % action)
    repo = resolve_repo(repo) if repo else None
    cmd = [PY, "-u", os.path.abspath(__file__)]
    flag = "--" + action.replace("_", "-")
    if action == "queue_run":
        cmd.append("--queue-run")
    elif repo:
        cmd += [flag, repo]
    else:
        raise SystemExit("%s needs a dataset" % action)
    for key in ("split", "tag", "map_", "limit", "max_len", "max_steps",
                "sample", "min_pass", "iteration"):
        val = kw.get(key)
        if val in (None, ""):
            continue
        cmd += ["--" + key.rstrip("_").replace("_", "-"), str(val)]
    if action == "drop":
        cmd.append("--yes")
    slug = (repo or "queue").replace("/", "--")
    log_path = os.path.join(LOGS, "hf-%s-%s.log" % (action, slug[:40]))
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf,
                                stderr=subprocess.STDOUT,
                                creationflags=0x00000200)  # detached
    job = {"id": "%s-%d" % (action, proc.pid), "action": action, "repo": repo,
           "pid": proc.pid, "started_utc": now(), "cwd": ROOT,
           "log": os.path.relpath(log_path, ROOT), "cmd": cmd[2:]}
    jobs = _load_jobs()
    jobs.append(job)
    _save_jobs(jobs)
    return job


def jobs():
    """Recorded jobs, each with liveness and a log tail (dashboard feed)."""
    out = []
    for j in reversed(_load_jobs()):
        path = os.path.join(ROOT, j.get("log") or "")
        tail = ""
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                tail = "\n".join(fh.read().splitlines()[-4:])
        except OSError:
            pass
        out.append(dict(j, running=pid_alive(j.get("pid")),
                        tail=tail[-400:]))
    return out


def wait_for_pid(pid, poll=30, log=print):
    """Block until a detached child exits (a loop iteration runs for hours)."""
    while True:
        try:
            out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
                                 capture_output=True, text=True,
                                 timeout=60).stdout
        except Exception:  # noqa: BLE001 — no tasklist => assume it finished
            return None
        if str(pid) not in out:
            return True
        time.sleep(poll)



def next_iteration(tag):
    """Continue an existing run's numbering instead of restarting at 1."""
    path = os.path.join(LOGS, "loop-%s.json" % tag)
    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
        iters = [it.get("iteration") or 0 for it in (state.get("iterations") or [])]
        return (max(iters) + 1) if iters else 1
    except (OSError, ValueError):
        return 1


def queue_pending(data=None):
    """Datasets still to process, in add order (the 'next dataset' queue)."""
    data = data or load_registry()
    done = {"learned_verified", "dropped", "error", "not_trainable"}
    return [e for e in data["datasets"] if e.get("state") not in done]


def queue_summary(data=None):
    """Counts for the dashboard: what is waiting, what is done, what failed."""
    data = data or load_registry()
    out = {}
    for e in data["datasets"]:
        out[e.get("state") or "unknown"] = out.get(e.get("state") or "unknown", 0) + 1
    pend = queue_pending(data)
    return {"by_state": out, "pending": len(pend),
            "next": pend[0]["id"] if pend else None}



def queue_run(split="full", limit=2000, map_="chat", max_len=1536, max_steps=-1,
              sample=40, min_pass=0.6, log=print):
    """One dataset at a time: pull -> convert -> verify -> fit -> train -> gate
    -> learned_verified -> delete the local copy -> next dataset.

    Stops (deleting nothing) the moment a gate does not pass: keeping the local
    copy is what makes another attempt possible, so it is only released on a
    verified win.
    """
    while True:
        pending = queue_pending()
        if not pending:
            log("[hf] queue empty — nothing left to learn from")
            return 0
        e, repo, st = pending[0], pending[0]["id"], pending[0].get("state")
        log("[hf] === queue: %s (state=%s, %d pending) ==="
            % (repo, st, len(pending)))
        if st == "pulled":
            convert(repo, limit=limit, map_=map_, log=log)
            verify(repo, max_len=max_len, log=log)
            fit(repo, max_len=max_len, log=log)
        elif st == "converted":
            verify(repo, max_len=max_len, log=log)
            fit(repo, max_len=max_len, log=log)
        elif st == "verified":
            fit(repo, max_len=max_len, log=log)
        elif st != "fitted":
            advance(repo, split=split, limit=limit, map_=map_,
                    max_len=max_len, log=log)
        tag = (entry_of(load_registry(), repo) or {}).get("tag") or slug_tag(repo)
        pid, tag = train(repo, tag=tag, iteration=next_iteration(tag),
                         max_steps=max_steps, max_len=max_len, sample=sample,
                         min_pass=min_pass, log=log)
        log("[hf] waiting for the loop (pid %d): train/export/serve/review/gate"
            % pid)
        wait_for_pid(pid, log=log)
        ok, detail = learned_verified(repo, tag=tag, log=log)
        if not ok:
            log("[hf] GATE NOT PASSED for %s: %s" % (repo, json.dumps(detail)))
            log("[hf] keeping %s on disk so another iteration can try"
                % (e.get("local_dir") or local_dir_for(repo)))
            return 2
        drop(repo, log=log)
        log("[hf] moved on — %d dataset(s) left in the queue"
            % len(queue_pending()))


if __name__ == "__main__":
    raise SystemExit(main())






