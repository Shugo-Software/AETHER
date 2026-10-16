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
  python scripts/hf_datasets.py --convert org/name --limit 0 --tag fable10k
  python scripts/hf_datasets.py --verify org/name --max-len 1536
  python scripts/hf_datasets.py --fit org/name --max-len 1536
  python scripts/hf_datasets.py --storage
  python scripts/hf_datasets.py --drop org/name --yes
  python scripts/hf_datasets.py --queue-plan       # what the queue would do (dry run)
  python scripts/hf_datasets.py --queue-run        # do it, one dataset at a time
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


def cache_env():
    """Environment for anything that downloads, converts or counts a dataset.

    The ingestion path used to inherit the venv's defaults, which put the HF
    caches on C: — a drive this project keeps critically small (plan.md). On
    2026-10-15 that cost a real failure: counting a parquet shard materialised it
    into the C: arrow cache and died with ENOSPC while G: had 1.6 TB free.
    `train_control.TRAIN_ENV` is the canonical G:-resident set; CUDA_VISIBLE_DEVICES
    is deliberately dropped, because the GPU mask belongs to the trainer only
    (hiding devices from llama-server/llama-imatrix breaks their device names —
    recursive_train splits it the same way).
    """
    env = dict(os.environ)
    try:
        import train_control as tc
        env.update({k: v for k, v in tc.TRAIN_ENV.items()
                    if k != "CUDA_VISIBLE_DEVICES"})
    except Exception:  # noqa: BLE001 — keep working if that import breaks
        env.setdefault("HF_HOME", os.path.join(ROOT, "cache", "hf"))
        env.setdefault("HF_DATASETS_CACHE", os.path.join(ROOT, "cache", "hf"))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def pin_caches():
    """Point this process's own caches at G:.

    `snapshot_download` and `count_rows` run in-process, so they cannot be given
    a child environment — they read os.environ.
    """
    for key, val in cache_env().items():
        if key.startswith(("HF_", "PIP_")) or key in ("TEMP", "TMP", "TMPDIR"):
            os.environ[key] = val


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
            # No card-declared data files: fall back to the repo's own files, but
            # never to an auxiliary parquet. openai-math ships `manifest.parquet`,
            # and pulling it "succeeded" only for every row to fail mapping.
            for s in info.get("siblings", []) or []:
                name = os.path.basename(s.get("rfilename", "")).lower()
                if name.startswith(("manifest", "readme", "dataset_infos",
                                    "index")):
                    continue
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


# A pull bigger than this is a corpus, not a training set: starcoderdata is 863
# shards of source code (tens of GB), and the "full" fallback will happily fetch
# all of it. Ask first.
BIG_PULL_FILES = 50
BIG_PULL_BYTES = 50 * 1024 ** 3


def pull_files(info, split):
    """The files a pull would fetch — the same decision `pull()` then makes.

    Returns (files, matched). `matched` is False when the card declares nothing
    for this split name, which means the fallback (every file the card declares)
    is what gets fetched — worth saying out loud, because that is how a "full"
    pull quietly turns into a whole-repo download.
    """
    files = split_files(info, split)
    if files:
        return files, True
    if split not in ("all", "*"):
        every = split_files(info, "all")
        if every:
            return every, False
    return [], False


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
    """Rows in a local file. Parquet reads its footer, never the data.

    `Dataset.from_parquet(path).num_rows` materialises the whole file into the
    datasets arrow cache first: counting 863 shards of a code corpus that way
    meant hundreds of GB of cache writes for a row count, and on 2026-10-15 it
    died with `OSError: [Errno 28] No space left on device` on a C: holding
    0.5 GB free. The parquet footer already has the number.
    """
    if path.lower().endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
            return pq.ParquetFile(path).metadata.num_rows
        except Exception:  # noqa: BLE001 — fall back to the datasets reader
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


def pull(repo, split="full", log=print, allow_big=False):
    """Download the whole requested split into cache/datasets/<org>--<name>/.

    `snapshot_download` runs in this process, so the caches are pinned to G:
    before it starts rather than being inherited from whatever launched us.
    """
    pin_caches()
    try:
        info = api_info(repo)
    except Exception as e:  # noqa: BLE001 — network/auth problems must be visible
        raise SystemExit("cannot read dataset info for %s: %s" % (repo, e))
    files, matched = pull_files(info, split)
    if not files:
        files = ["%s/train.jsonl" % split]
    if not matched:
        # Nothing on the card is called `full`/`train`: everything the card
        # declares gets pulled instead of guessing `<split>/train.jsonl`, which
        # downloads nothing and reports "pulled 0 rows" (seen on gsm8k).
        log("[hf] the card declares no %r split — pulling all %d declared "
            "file(s)" % (split, len(files)))
    dest = local_dir_for(repo)
    os.makedirs(dest, exist_ok=True)
    size_of = {s.get("rfilename"): (s.get("size") or 0)
               for s in (info.get("siblings") or [])}
    want = sum(size_of.get(f) or 0 for f in files)
    free = disk_free(dest)
    if free is not None and want and free < want * 1.2:
        raise SystemExit("not enough space on the dataset drive: %s free, "
                         "~%s needed" % (human(free), human(want)))
    log("[hf] %s -> %s" % (repo, os.path.relpath(dest, ROOT)))
    log("[hf] %d file(s) to fetch%s"
        % (len(files), (", %s" % human(want)) if want
           else ", size not reported by the card"))
    if not want:
        # The space guard above needs a size; without one it silently does
        # nothing — say that rather than implying the pull is small.
        log("[hf] WARNING: the card reports no sizes — the space check is "
            "skipped and the download size is unknown until it lands")
    if (len(files) > BIG_PULL_FILES or want > BIG_PULL_BYTES) \
            and not (allow_big or os.environ.get("AETHER_ALLOW_BIG_PULL")):
        raise SystemExit(
            "refusing a large pull: %d file(s)%s — this is a corpus, not a "
            "training set (starcoderdata is 863 shards of source code). Re-run "
            "with --allow-big-pull (or AETHER_ALLOW_BIG_PULL=1) if that is what "
            "you want." % (len(files), (", %s" % human(want)) if want else ""))
    if len(files) <= 12:
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


def _shard_group(files, local):
    """The pulled files of ONE split: every shard of it, but not two splits.

    A repo shipping the same rows twice (fable's `5k/` and `10k/`) must not be
    trained on twice — but a sharded split (say 20 parquet files) has to be read
    whole, and `convert` used to keep only the single largest file, silently
    losing every other shard. Group by directory, take the heaviest group, and
    keep all of it (sorted, for a stable command line).
    """
    def size(rel):
        p = os.path.join(local, rel)
        return os.path.getsize(p) if os.path.isfile(p) else 0
    groups = {}
    for rel in files:
        groups.setdefault(os.path.dirname(rel), []).append(rel)
    best = max(groups, key=lambda d: sum(size(r) for r in groups[d]))
    return sorted(groups[best]), sorted(groups)


def convert(repo, tag=None, limit=0, map_="auto", log=print,
            max_prompt_chars=4000, max_completion_chars=4000,
            max_reasoning_chars=2000):
    """Map the local copy into AETHER's training schema (datasets/<tag>.jsonl).

    Reads the pulled copy, never the hub: the file is on the dataset drive, so
    re-streaming it would waste exactly the bandwidth the pull step paid for.

    `limit=0` (the default) ingests the whole copy, and the scan is never capped
    — a cap here is the one thing that makes a full pull look like a partial
    dataset, so it stays opt-in.
    """
    data = load_registry()
    e = entry_of(data, repo) or upsert(data, repo)
    tag = tag or slug_tag(repo)
    local = e.get("local_dir") or local_dir_for(repo)
    if not os.path.isdir(local):
        raise SystemExit("no local copy for %s — run --pull first" % repo)
    cmd = [PY, "-u", os.path.join(SCRIPTS, "ingest_hf_dataset.py"),
           "--repo", repo, "--map", map_, "--tag", tag,
           "--from-local", local, "--limit", str(limit), "--scan-limit", "0",
           "--max-prompt-chars", str(max_prompt_chars),
           "--max-completion-chars", str(max_completion_chars),
           "--max-reasoning-chars", str(max_reasoning_chars)]
    # Convert exactly the files the pull fetched — and ALL shards of the chosen
    # split. Naming them keeps the decision explicit and auditable.
    files = e.get("pulled_files") or []
    if len(files) > 1:
        files, groups = _shard_group(files, local)
        if len(groups) > 1:
            log("[hf] %d splits on disk (%s) — converting %s (%d file(s)); the "
                "same rows are not ingested twice"
                % (len(groups), ", ".join(g or "(root)" for g in groups),
                   os.path.dirname(files[0]) or "(root)", len(files)))
    if files:
        cmd += ["--from-local-files", ",".join(files)]
        cmd += ["--split", e.get("pulled_split") or "train"]
    log("[hf] $ %s" % " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, env=cache_env())
    out = os.path.join(DATASETS, tag + ".jsonl")
    if proc.returncode != 0 or not os.path.isfile(out):
        # "convert failed rc=1" hides three different things: a *verdict* (the
        # mapping refused every row and counted why), a *crash*, and a missing
        # output — indistinguishable in the panel. The ingester writes a manifest
        # either way, so read it and record what actually happened.
        counts = _manifest_counts(tag)
        if counts and not counts.get("written"):
            why = _drop_reason(counts.get("dropped") or {})
            upsert(data, repo, state="not_trainable", tag=tag,
                   scanned=counts.get("scanned"),
                   dropped=dict(counts.get("dropped") or {}),
                   note="mapped 0 of %s rows — %s"
                        % (counts.get("scanned"), why))
            save_registry(data)
            log("[hf] %s: not trainable as SFT — %s" % (repo, why))
            return entry_of(load_registry(), repo)
        upsert(data, repo, state="error",
               note="convert failed rc=%d (no manifest written)" % proc.returncode)
        save_registry(data)
        raise SystemExit("convert failed (rc=%d)" % proc.returncode)
    rows = count_rows(out)
    agentic = (_manifest(tag) or {}).get("agentic_rows") or 0
    note = "converted -> %s (%d rows)" % (os.path.basename(out), rows)
    extra = {}
    if agentic:
        # Say it where the operator reads it: the HF panel's "why" column and the
        # queue history both show this note.
        extra["agentic_rows"] = agentic
        note += (" — %d of them mined from agentic traces (prompts carry tool "
                 "results the student cannot produce)" % agentic)
    e = upsert(data, repo, state="converted", tag=tag,
               corpus=os.path.relpath(out, ROOT), corpus_rows=rows,
               corpus_bytes=os.path.getsize(out), note=note, **extra)
    save_registry(data)
    log("[hf] corpus %s: %d rows%s"
        % (os.path.relpath(out, ROOT), rows,
           " (%d from agentic traces)" % agentic if agentic else ""))
    return e


def _manifest(tag):
    """The ingester's manifest for a tag ({} when there is none)."""
    path = os.path.join(DATASETS, tag + ".manifest.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def _manifest_counts(tag):
    """The manifest's counts for a tag: scanned / written / dropped.

    Written by ingest_hf_dataset.write_output even when the mapping kept nothing,
    which is what makes a verdict distinguishable from a crash.
    """
    return (_manifest(tag) or {}).get("counts") or {}


_DROP_HINTS = {
    "tools_present": "every row carries a tool schema (agentic traces, not "
                     "prompt→answer pairs)",
    "assistant_tool_calls": "every assistant turn is a tool call",
    "agentic_marker": "every row looks like repo-scale agentic scaffolding",
    "no_prompt_pair": "no recognised prompt/answer column pair — for a code "
                      "corpus that is expected: it is pretraining data, not SFT",
    "no_user_or_assistant": "the rows carry no user/assistant messages",
    "unparsable_json": "no row parsed as JSON",
    "empty_completion": "every assistant answer was empty",
    "empty_prompt": "every rendered prompt was empty",
    "duplicate_pair": "every row was a duplicate of another",
    "prompt_too_long": "every prompt exceeded the length budget",
}


def _drop_reason(dropped):
    """One sentence for why a mapping kept nothing, from its drop counts."""
    if not dropped:
        return "the mapping kept no rows and recorded no reason"
    total = sum(dropped.values()) or 1
    bits = []
    for key, n in sorted(dropped.items(), key=lambda kv: -kv[1])[:3]:
        hint = _DROP_HINTS.get(key)
        bits.append("%s (%d/%d%s)" % (key, n, total, " — " + hint if hint else ""))
    return "; ".join(bits)


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
    proc = subprocess.run(cmd, cwd=ROOT, env=cache_env())
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
    proc = subprocess.run(cmd, cwd=ROOT, env=cache_env())
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
    proc = subprocess.run(cmd, cwd=ROOT, env=cache_env())
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
    _record_job("train", repo, proc.pid, log_path, cmd)
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


def advance(repo, split="full", limit=0, map_="auto", max_len=1536,
            tag=None, log=print, allow_big=False):
    """Data stages only: pull -> convert -> verify -> fit. Returns the registry entry."""
    log("[hf] === %s: pull ===" % repo)
    pull(repo, split=split, log=log, allow_big=allow_big)
    log("[hf] === %s: convert ===" % repo)
    e = convert(repo, tag=tag, limit=limit, map_=map_, log=log)
    if (e or {}).get("state") == "not_trainable":
        # A verdict, not a crash: verify/fit would only fail on a corpus that was
        # never written, and the reason is already recorded in the registry.
        log("[hf] %s is not trainable as SFT — stopping the data stages" % repo)
        return e
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
        # A killed loop leaves state=training behind. It must not lock the queue
        # forever (found live 2026-10-15: openai/gsm8k sat in "training" with no
        # process behind it, so the operator could not reorder or skip the row).
        upsert(again, "org/one", state="training", loop_pid=999999)
        good = stalled(entry_of(again, "org/one"))
        ok = ok and good
        log("[selftest] %s a dead runner is reported stalled"
            % ("ok  " if good else "FAIL"))
        try:
            _locked("org/one", again)
            unlocked = True
        except SystemExit:
            unlocked = False
        good = unlocked and not queue_items(again)[0]["locked"]
        ok = ok and good
        log("[selftest] %s a stalled dataset is not locked"
            % ("ok  " if good else "FAIL"))
        good = (resume_state({"corpus": os.path.join("datasets", "x.jsonl")})
                == "fitted" and resume_state({}) == "added")
        ok = ok and good
        log("[selftest] %s a stalled dataset resumes at the stage it reached"
            % ("ok  " if good else "FAIL"))
        # ...but a runner that IS alive, and still ours, holds the lock.
        global _cmdline
        upsert(again, "org/one", state="training", loop_pid=os.getpid())
        keep_cmdline = _cmdline
        try:
            _cmdline = lambda pid, timeout=15: (
                r"C:\aether\venvs\ml\Scripts\python.exe -u "
                r"C:\aether\scripts\recursive_train.py --tag x")
            _OURS_CACHE.clear()
            try:
                _locked("org/one", again)
                locked = False
            except SystemExit:
                locked = True
            good = locked and not stalled(entry_of(again, "org/one"))
        finally:
            _cmdline = keep_cmdline
            _OURS_CACHE.clear()
        ok = ok and good
        log("[selftest] %s a live runner still locks its dataset"
            % ("ok  " if good else "FAIL"))
        # ...and a recycled pid is not a live runner (the 2026-10-15 failure:
        # the loop pid had come back as Code.exe, so the run looked alive).
        try:
            _cmdline = lambda pid, timeout=15: r"C:\Apps\Code.exe --type=utility"
            _OURS_CACHE.clear()
            good = (stalled(entry_of(again, "org/one"))
                    and not _is_runner_cmdline("C:\\Apps\\Code.exe --type=utility"))
        finally:
            _cmdline = keep_cmdline
            _OURS_CACHE.clear()
        ok = ok and good
        log("[selftest] %s a recycled pid is not mistaken for a live runner"
            % ("ok  " if good else "FAIL"))
        upsert(again, "org/one", state="pulled", bytes=123, rows=4, loop_pid=None)
        plan = queue_plan(again)
        good = (plan["pending"] == 1 and plan["next"] == "org/one"
                and plan["items"][0]["steps"][:4]
                == ["convert", "verify", "fit", "train"]
                and plan["items"][0]["evidence"]["local_dir"].endswith("org--one"))
        ok = ok and good
        log("[selftest] %s the plan lists the stages the runner would walk"
            % ("ok  " if good else "FAIL"))
        # The full-copy defaults: a cap is the one thing that makes a full pull
        # look like a partial dataset, so it must stay opt-in (0 = all rows).
        import inspect as _inspect
        good = (all(_inspect.signature(fn).parameters["limit"].default == 0
                    for fn in (convert, advance, queue_run))
                and all(_inspect.signature(fn).parameters["map_"].default == "auto"
                        for fn in (convert, advance, queue_run)))
        ok = ok and good
        log("[selftest] %s ingestion defaults to the whole copy (limit=0, map=auto)"
            % ("ok  " if good else "FAIL"))
        # A manifest/aux parquet is not data: openai-math ships only
        # manifest.parquet, and pulling it "succeeded" while every row failed.
        info = {"siblings": [{"rfilename": "manifest.parquet", "size": 10},
                             {"rfilename": "data/train.jsonl", "size": 20}],
                "cardData": {}}
        good = pull_files(info, "full")[0] == ["data/train.jsonl"]
        ok = ok and good
        log("[selftest] %s a manifest parquet is never pulled as data"
            % ("ok  " if good else "FAIL"))
        # A 0-row conversion has to explain itself, not just report rc=1.
        why = _drop_reason({"tools_present": 1306})
        good = "tools_present" in why and "tool schema" in why
        ok = ok and good
        log("[selftest] %s a 0-row conversion explains itself"
            % ("ok  " if good else "FAIL"))
        # Every shard of the chosen split, but never two splits of the same rows.
        for sub, names in (("5k", ["a.jsonl"]), ("10k", ["a.jsonl", "b.jsonl"])):
            os.makedirs(os.path.join(DEST_ROOT, sub), exist_ok=True)
            for n in names:
                with open(os.path.join(DEST_ROOT, sub, n), "wb") as fh:
                    fh.write(b"x" * (40 if sub == "10k" else 10))
        picked, groups = _shard_group(["5k/a.jsonl", "10k/a.jsonl", "10k/b.jsonl"],
                                      DEST_ROOT)
        good = picked == ["10k/a.jsonl", "10k/b.jsonl"] and len(groups) == 2
        ok = ok and good
        log("[selftest] %s all shards of one split are read, not just the biggest"
            % ("ok  " if good else "FAIL"))
        # The teacher stops the gate, never ingestion.
        blockers, gate, _warn = _preflight({"state": "added"}, ["convert"],
                                           "org/missing")
        good = (any("local copy" in b for b in blockers)
                and not any("8080" in b for b in blockers)
                and len(gate) <= 1)
        ok = ok and good
        log("[selftest] %s ingestion blockers stay separate from the gate"
            % ("ok  " if good else "FAIL"))
        good = (stage_steps({"state": "added"})
                == ["pull", "convert", "verify", "fit", "train"]
                and stage_steps({"state": "fitted"}) == ["train"])
        ok = ok and good
        log("[selftest] %s the ladder is the runner's own step list"
            % ("ok  " if good else "FAIL"))
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
    ap.add_argument("--queue-plan", action="store_true",
                    help="dry run: print what the queue would do, item by item, "
                         "with its preconditions checked (no writes, no network)")
    ap.add_argument("--remote", action="store_true",
                    help="with --queue-plan: also ask the hub what a pull would "
                         "fetch (file count and size)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--storage", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="offline checks (URL parsing, registry, drop gating)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--split", default="full", help="full|train|all (dataset card)")
    ap.add_argument("--allow-big-pull", action="store_true",
                    help="permit a pull larger than %d files / %s (a corpus "
                         "rather than a training set)" % (BIG_PULL_FILES,
                                                          human(BIG_PULL_BYTES)))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--map", dest="map_", default="auto",
                    help="schema mapping: auto (detect) | chat | flat")
    ap.add_argument("--limit", type=int, default=0,
                    help="rows to convert (0 = all rows in the pulled copy)")
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
    if a.queue_plan:
        plan = queue_plan(settings={"split": a.split, "limit": a.limit,
                                    "map": a.map_, "max_len": a.max_len,
                                    "max_steps": a.max_steps, "sample": a.sample,
                                    "min_pass": a.min_pass},
                          remote=a.remote)
        if a.json:
            print(json.dumps(plan, indent=1))
            return 0 if plan["runnable"] else 1
        print("[hf] queue plan: %d pending, next %s"
              % (plan["pending"], plan["next"] or "-"))
        if not plan["items"]:
            print("[hf] queue is empty — nothing to run")
            return 0
        st = plan["settings"] or {}
        print("[hf] a run would use: split=%s map=%s limit=%s max_len=%s "
              "max_steps=%s"
              % (st.get("split"), st.get("map"),
                 "all" if not st.get("limit") else st["limit"],
                 st.get("max_len"),
                 "from epochs" if (st.get("max_steps") or -1) < 0
                 else st.get("max_steps")))
        for it in plan["items"]:
            ev = it["evidence"]
            print("\n  %d. %s  [%s]%s"
                  % (it["pos"] + 1, it["id"], it["state"],
                     "  STALLED (its runner is gone)" if it["stalled"] else ""))
            print("     steps    : %s" % " -> ".join(it["steps"]))
            pull = it.get("pull")
            if pull:
                if pull.get("error"):
                    print("     pull     : size unknown (%s)" % pull["error"])
                else:
                    print("     pull     : %d file(s), %s%s"
                          % (pull["files"],
                             pull.get("bytes_human") or "size not reported",
                             "   <-- a corpus: needs --allow-big-pull"
                             if pull.get("big") else ""))
            print("     on disk  : %s %s%s"
                  % (ev["local_dir"], ev["on_disk_human"],
                     "" if ev["on_disk"] else "  (absent)"))
            print("     corpus   : %s (%s rows)%s"
                  % (ev["corpus"] or "-",
                     ev["corpus_rows"] if ev["corpus_rows"] is not None else "-",
                     "" if ev["corpus_present"] or not ev["corpus"] else "  MISSING"))
            print("     fitted   : %s (%s rows)%s"
                  % (ev["fitted"] or "-",
                     ev["fitted_rows"] if ev["fitted_rows"] is not None else "-",
                     "" if ev["fitted_present"] or not ev["fitted"] else "  MISSING"))
            print("     artifacts: adapter=%s  gguf=%s  gate=%s"
                  % (ev["adapter_present"], ",".join(ev["gguf"]) or "-",
                     ev["gate"] if ev["gate"] is not None else "-"))
            for b in it["blockers"]:
                print("     BLOCKER  : %s" % b)
            for b in it["gate_blockers"]:
                print("     BLOCKS   : training/gate only — %s" % b)
            for w in it["warnings"]:
                print("     warn     : %s" % w)
        print()
        if not plan["runnable"]:
            print("[hf] queue would NOT run — resolve the blockers above")
        elif not plan["can_train"]:
            print("[hf] ingestion is runnable (pull -> convert -> verify -> fit), "
                  "but the training loop is blocked — see BLOCKS above")
        else:
            print("[hf] queue is fully runnable")
        return 0 if plan["runnable"] else 1
    if a.queue_run:
        # Recorded like any other job so the dashboard's Pipeline jobs table can
        # see a queue run started from a console (its log is wherever the caller
        # sent stdout; an empty tail is better than an invisible job).
        out_name = getattr(sys.stdout, "name", "") or ""
        if out_name.startswith("<"):     # a console, not a redirected log file
            out_name = ""
        if not os.environ.get("AETHER_JOB"):
            # A dashboard-spawned queue run is already recorded by spawn(); only a
            # console start needs to record itself.
            _record_job("queue_run", None, os.getpid(), out_name, ["queue-run"])
        return queue_run(split=a.split, limit=a.limit, map_=a.map_,
                         max_len=a.max_len, max_steps=a.max_steps,
                         sample=a.sample, min_pass=a.min_pass, log=print,
                         force=a.force, allow_big=a.allow_big_pull)

    repo = None
    for ref in (a.pull, a.convert, a.verify, a.fit, a.train, a.drop,
                a.advance, a.learned_verified, a.rewrite):
        if ref:
            repo = resolve_repo(ref)
    if repo is None:
        ap.error("pick an action: --add/--list/--pull/--convert/--verify/--fit/"
                 "--train/--advance/--drop/--storage/--queue-plan/--queue-run")

    if a.pull:
        e = pull(repo, split=a.split, log=log, allow_big=a.allow_big_pull)
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
                    max_len=a.max_len, tag=a.tag, log=log,
                    allow_big=a.allow_big_pull)
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


def _record_job(action, repo, pid, log_path, cmd):
    """Add one running job to the feed the dashboard's Pipeline jobs table reads.

    Every detached child the pipeline starts belongs here, not only the ones the
    web handler spawns: `train()` and `--queue-run` are started from a console
    too, and the table was blind to them — seen live 2026-10-15, when a queue
    run and its training loop were in flight and the panel showed only a stale
    `add` row from six hours earlier.
    """
    job = {"id": "%s-%d" % (action, pid), "action": action, "repo": repo,
           "pid": pid, "started_utc": now(), "cwd": ROOT,
           "log": os.path.relpath(log_path, ROOT) if log_path else "",
           "cmd": list(cmd[2:])}
    jobs = _load_jobs()
    jobs.append(job)
    _save_jobs(jobs)
    return job


def pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % int(pid), "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
        return str(pid) in out
    except Exception:  # noqa: BLE001
        return False


# A live pid is NOT proof of life: Windows recycles pids, and the loop pid
# recorded for openai/gsm8k on 2026-10-15 came back as Code.exe, which kept a
# dead run looking alive. So a pid only counts as a runner if its command line
# still names one of our scripts — the same reasoning as
# train_control.stop_training, which refuses to kill a PID that no longer names
# its own trainer.
_OWNER_MARKERS = ("recursive_train.py", "hf_datasets.py", "train_student.py")
_OURS_CACHE = {}      # pid -> (monotonic, bool)


def _cmdline(pid, timeout=15):
    """Command line of a PID on Windows (via CIM); '' if gone/unreadable."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter 'ProcessId=%d').CommandLine"
             % int(pid)],
            capture_output=True, text=True, timeout=timeout).stdout
        return out.strip()
    except Exception:  # noqa: BLE001
        return ""


def _is_runner_cmdline(cmdline):
    """Does a command line belong to one of our detached runners?"""
    return any(m in (cmdline or "") for m in _OWNER_MARKERS)


def process_alive(pid):
    """Is this pid alive *and* still one of ours? None if it cannot be checked.

    Windows recycles pids, so existence alone proves nothing (a dead run's pid
    came back as Code.exe on 2026-10-15). `None` means "no pid to check", which
    callers treat as "cannot tell" rather than "dead".
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if not pid:
        return None
    return pid_alive(pid) and _ours(pid)


def _ours(pid):
    """Is a live pid still one of our runners? Unreadable => assume ours.

    Never steal a lock on a guess: if the command line cannot be read, the
    conservative answer is "it may still be ours".
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    now = time.monotonic()
    hit = _OURS_CACHE.get(pid)
    if hit and now - hit[0] < 10:
        return hit[1]
    cmd = _cmdline(pid)
    ours = True if not cmd else _is_runner_cmdline(cmd)
    _OURS_CACHE[pid] = (now, ours)
    return ours


def live_pids():
    """Every running PID in ONE tasklist call (jobs() used to shell out per job)."""
    pids = set()
    try:
        out = subprocess.run(["tasklist", "/NH", "/FO", "CSV"],
                             capture_output=True, text=True, timeout=60).stdout
        for line in out.splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) > 1 and parts[1].isdigit():
                pids.add(parts[1])
    except Exception:  # noqa: BLE001
        pass
    return pids


def jobs():
    """Recorded jobs, each with liveness and a log tail (dashboard feed)."""
    out = []
    alive = live_pids()
    for j in reversed(_load_jobs()):
        path = os.path.join(ROOT, j.get("log") or "")
        tail = ""
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                tail = "\n".join(fh.read().splitlines()[-4:])
        except OSError:
            pass
        running = (str(j.get("pid")) in alive) if alive else pid_alive(j.get("pid"))
        if running and not _ours(j.get("pid")):
            running = False      # the pid was recycled by something else
        out.append(dict(j, running=running, tail=tail[-400:]))
    return out


def live_job_for(repo):
    """The recorded job currently working on this repo, if any.

    The queue refuses to touch an item that is in flight; the row buttons had no
    such guard, so a second click started a second job on the same dataset — two
    `snapshot_download`s into one directory and two writers on the registry.
    """
    for j in jobs():
        if j.get("running") and j.get("repo") == repo:
            return j
    return None


def spawn(action, repo=None, **kw):
    """Launch one pipeline action as a detached job (what the dashboard calls).

    Runs this same script as a subprocess so the UI path and the CLI path are
    literally the same code, and so a slow pull cannot block the web handler.
    """
    if action not in JOBS:
        raise SystemExit("unknown action %r" % action)
    repo = resolve_repo(repo) if repo else None
    if repo:
        busy = live_job_for(repo)
        if busy:
            raise SystemExit(
                "%s is already being worked on by %s (pid %s) — wait for it, or "
                "read its log first" % (repo, busy.get("action"), busy.get("pid")))
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
    if kw.get("allow_big"):
        # The CLI flag is --allow-big-pull, which the generic key->flag loop
        # above cannot derive from "allow_big".
        cmd.append("--allow-big-pull")
    slug = (repo or "queue").replace("/", "--")
    log_path = os.path.join(LOGS, "hf-%s-%s.log" % (action, slug[:40]))
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # Tell the child it was spawned (and is therefore already in the job feed):
    # without this its own --queue-run self-record wrote a second row for the
    # same click, which reads like two downloads.
    env["AETHER_JOB"] = "1"
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf,
                                stderr=subprocess.STDOUT,
                                creationflags=0x00000200)  # detached
    return _record_job(action, repo, proc.pid, log_path, cmd)


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
    """Datasets still to process, in queue order (skipped/excluded ones omitted)."""
    data = data or load_registry()
    return [e for e in data["datasets"]
            if (e.get("state") or "") not in DONE_STATES]


IN_FLIGHT_STATES = {"pulling", "converting", "verifying", "fitting", "rewriting",
                    "training"}
DONE_STATES = {"learned_verified", "dropped", "error", "not_trainable", "skipped"}


def _owner_pid(e):
    """The pid that claimed a dataset's current state, if one was recorded.

    `train()` stores loop_pid (the recursive loop it launched); the stage
    functions would store runner_pid. Without a pid there is nothing to check
    against, so the state stays authoritative and the item stays locked.
    """
    for key in ("loop_pid", "runner_pid", "pid"):
        try:
            if e.get(key):
                return int(e[key])
        except (TypeError, ValueError):
            continue
    return None


def _owner_alive(e):
    """Is the process that claimed this state still alive, and still ours?"""
    pid = _owner_pid(e)
    if not pid:
        return True          # nothing claimed it: the state stays authoritative
    return pid_alive(pid) and _ours(pid)


def stalled(e):
    """A state that says work is happening, with no live process behind it.

    A runner can be killed mid-step (machine sleep, a closed console, an OOM)
    and nothing rewrites the state afterwards: `learned_verified()` only runs
    once the loop reaches the gate. Seen live 2026-10-15 — openai/gsm8k sat in
    state=training with its loop gone, which locked its queue row forever and
    made the panel claim work that was not happening. The owner's pid is
    checked for *ownership*, not just existence, because Windows had already
    recycled that pid onto an unrelated process by the time we looked.
    """
    if (e.get("state") or "") not in IN_FLIGHT_STATES:
        return False
    return bool(_owner_pid(e)) and not _owner_alive(e)

# What the queue will do with an item that is in a given state.
NEXT_STEP = {
    "added": "pull", "pulled": "convert", "converted": "verify",
    "verified": "fit", "fitted": "train", "training": "review + gate",
    "skipped": "(skipped — files kept)", "learned_verified": "delete local copy",
    "dropped": "(released)", "not_trainable": "(excluded — see audit)",
    "error": "(failed — see history)", "rewriting": "teacher rewrite",
    "removed": "(removed from queue)",
}


def resume_state(e):
    """Where an entry whose runner died re-enters the queue.

    The fitted corpus is still on disk, so a killed loop resumes at train rather
    than paying for the pull/convert/verify/fit chain again; only an entry that
    never got that far restarts from the top.
    """
    return "fitted" if (e.get("fitted") or e.get("corpus")) else "added"


# How much of the pipeline is already behind us, per state. This is the single
# source of truth for `--queue-plan` AND for `queue_run`: the plan is printed from
# it and the runner walks it, so a plan cannot describe a sequence the runner
# would not actually perform.
LADDER = {
    "added": ("pull", "convert", "verify", "fit"),
    "pulled": ("convert", "verify", "fit"),
    "converted": ("verify", "fit"),
    "verified": ("fit",),
    "fitted": (),
    "rewriting": ("verify", "fit"),
    "training": (),        # a live loop owns this dataset right now
}


def stage_steps(e):
    """The stages `queue_run` would execute for this entry, in order.

    A dead runner resumes where its corpus actually got to (resume_state), a live
    one is waited for rather than duplicated, and every item ends at the gate.
    """
    st = e.get("state") or ""
    live = st in IN_FLIGHT_STATES and not stalled(e)
    if stalled(e):
        st = resume_state(e)
    steps = list(LADDER.get(st, LADDER["added"]))
    if not live:
        steps.append("train")
    return steps


def _runner_pid(repo):
    """The pid of the live runner working on this dataset, if we can name it."""
    for j in jobs():
        if j.get("running") and j.get("repo") == repo and j.get("pid"):
            try:
                return int(j["pid"])
            except (TypeError, ValueError):
                pass
    e = entry_of(load_registry(), repo)
    return _owner_pid(e) if (e and not stalled(e)) else None


def active_repo():
    """The dataset the pipeline is working on right now, if any.

    Used to lock the queue: reordering or skipping the item that is mid-pull or
    mid-training would desynchronise the runner from what is actually running.
    A *stalled* entry (state says in-flight, owner pid gone) is not work in
    progress, so it does not lock anything.
    """
    for j in jobs():
        if j.get("running") and j.get("repo"):
            return j["repo"]
    for e in load_registry()["datasets"]:
        if (e.get("state") or "") in IN_FLIGHT_STATES and not stalled(e):
            return e["id"]
    return None


def queue_items(data=None):
    """The ordered queue with, per item, what happens next and whether it is locked."""
    data = data or load_registry()
    act = active_repo()
    out = []
    for pos, e in enumerate(data["datasets"]):
        st = e.get("state") or "unknown"
        is_stalled = stalled(e)
        out.append({
            "pos": pos, "id": e["id"], "state": st,
            "pending": st not in DONE_STATES,
            "rows": e.get("rows"), "fitted_rows": e.get("fitted_rows"),
            "bytes_human": human(e.get("bytes") or 0),
            "next_step": ("stalled — resumes at train" if is_stalled
                          else NEXT_STEP.get(st, "?")),
            "active": act == e["id"],
            "stalled": is_stalled,
            "locked": ((act == e["id"])
                       or (st in IN_FLIGHT_STATES and not is_stalled)),
            "tag": e.get("tag"), "note": e.get("note")})
    return out


def _locked(repo, data):
    e = entry_of(data, repo)
    if e is None:
        raise SystemExit("unknown dataset %s" % repo)
    st = e.get("state") or ""
    if active_repo() == repo or (st in IN_FLIGHT_STATES and not stalled(e)):
        raise SystemExit(
            "refusing to change %s: the pipeline is working on it right now "
            "(state=%s). Wait for the step to finish." % (repo, st))
    return e


def queue_move(repo, direction, data=None):
    """Swap an item with the next/previous PENDING item. Refuses when locked."""
    data = data or load_registry()
    _locked(repo, data)
    items = list(data["datasets"])
    target = entry_of(data, repo)
    idx = items.index(target)
    step = -1 if direction == "up" else 1
    j = idx + step
    while 0 <= j < len(items):
        other = items[j]
        if (other.get("state") or "") not in DONE_STATES:
            items[idx], items[j] = items[j], items[idx]
            data["datasets"] = items
            save_registry(data)
            return {"ok": True, "moved": repo, "direction": direction,
                    "swapped_with": other["id"]}
        j += step
    return {"ok": True, "moved": None,
            "note": "%s is already at the %s end of the queue" % (repo, direction)}


def queue_skip(repo, skip=True, data=None):
    """Take an item out of the queue (or put it back) WITHOUT touching its files.

    Distinct from `drop`: skipping keeps the local copy and its provenance, it
    just stops the runner from working on it.
    """
    data = data or load_registry()
    e = _locked(repo, data)
    st = e.get("state") or ""
    if skip:
        if st == "skipped":
            return {"ok": True, "note": "already skipped"}
        e["skipped_from"] = st
        e["state"] = "skipped"
        upsert(data, repo, note="skipped by operator (was %s) — files kept" % st)
    else:
        back = e.pop("skipped_from", None) or "added"
        e["state"] = back
        upsert(data, repo, note="unskipped -> %s" % back)
    save_registry(data)
    return {"ok": True, "id": repo, "state": e["state"]}


def queue_add(ref, data=None):
    """Append a dataset to the queue (registers it; pull/convert follow)."""
    repo = resolve_repo(ref)
    data = data or load_registry()
    e = upsert(data, repo, note="added to the queue")
    save_registry(data)
    return {"ok": True, "id": repo, "state": e.get("state"),
            "position": len(data["datasets"]) - 1}


def queue_summary(data=None):
    """Counts for the dashboard: what is waiting, what is done, what failed."""
    data = data or load_registry()
    out = {}
    for e in data["datasets"]:
        out[e.get("state") or "unknown"] = out.get(e.get("state") or "unknown", 0) + 1
    pend = queue_pending(data)
    return {"by_state": out, "pending": len(pend),
            "next": pend[0]["id"] if pend else None,
            "active": active_repo(),
            "stalled": [e["id"] for e in data["datasets"] if stalled(e)]}


def _port_open(port, timeout=0.4):
    """Is something listening on this local port? (raw connect, no HTTP.)"""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _rel(path):
    """ROOT-relative path for display; the absolute path if it is off-drive.

    os.path.relpath raises ValueError across Windows drives — which is exactly
    what the selftest hit, since its temp dataset drive is on C: while the repo
    lives on G:.
    """
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def _evidence(e, repo):
    """What is on disk for this entry, next to what the registry claims.

    This is the "were the datasets actually processed?" answer: the recorded
    rows/files against what exists right now, plus the artifacts a stage writes.
    """
    local = e.get("local_dir") or local_dir_for(repo)
    tag = e.get("tag")
    corpus = e.get("corpus")
    fitted = e.get("fitted")
    present = os.path.isdir(local)
    exports = os.path.join(ROOT, "exports", tag) if tag else None
    return {
        "local_dir": _rel(local),
        "on_disk": present,
        "on_disk_human": human(dir_bytes(local)) if present else "0 B",
        "corpus": corpus,
        "corpus_present": bool(corpus) and
        os.path.isfile(os.path.join(ROOT, corpus)),
        "corpus_rows": e.get("corpus_rows"),
        "fitted": fitted,
        "fitted_present": bool(fitted) and
        os.path.isfile(os.path.join(ROOT, fitted)),
        "fitted_rows": e.get("fitted_rows"),
        "adapter_present": bool(tag) and os.path.isfile(
            os.path.join(ROOT, "adapters", tag, "adapter_config.json")),
        "gguf": (sorted(f for f in os.listdir(exports)
                        if f.endswith(".gguf"))
                 if exports and os.path.isdir(exports) else []),
        "gate": (e.get("gate") or {}).get("mean_pass"),
        "learned_iteration": e.get("learned_iteration"),
    }


def _preflight(e, steps, repo):
    """Preconditions for the stages about to run: files and tools only.

    Returns `(blockers, gate_blockers, warnings)`. The split matters: a *blocker*
    stops the item, while a *gate blocker* (the teacher being down) only stops the
    training/gate tail — pull, convert, verify and fit are local work on a dataset
    that is already on G: and must never be held up by a service that is not
    needed until hours later.

    The stages are walked *in order* and what each earlier one leaves behind
    counts as satisfied for the next, so an `added` dataset is not "blocked" by
    the local copy that its own pull creates. No writes and no hub traffic: what
    cannot be checked offline (the hub reachable, a card's split list) is
    reported as *not checked* rather than assumed good.
    """
    blockers, gate_blockers, warnings = [], [], []
    local = e.get("local_dir") or local_dir_for(repo)
    corpus, fitted = e.get("corpus"), e.get("fitted")
    will_have = set()
    if os.path.isdir(local):
        will_have.add("local")
    if corpus and os.path.isfile(os.path.join(ROOT, corpus)):
        will_have.add("corpus")
    training_on = fitted or corpus
    if training_on and os.path.isfile(os.path.join(ROOT, training_on)):
        will_have.add("trainable")
    for stage in steps:
        if stage == "pull":
            if not os.path.isfile(PY):
                blockers.append("venv python missing at %s (only it has "
                                "huggingface_hub)" % _rel(PY))
            free = disk_free(DEST_ROOT)
            if free is not None and free < 2 * 1024 ** 3:
                warnings.append("only %s free on the dataset drive" % human(free))
            warnings.append("pull needs huggingface.co reachable (not checked)")
            will_have.add("local")
        elif stage == "convert":
            if "local" not in will_have:
                blockers.append("no local copy at %s — pull has not run"
                                % _rel(local))
            will_have.add("corpus")
        elif stage in ("verify", "fit"):
            if "corpus" not in will_have:
                blockers.append("no corpus recorded — convert has not run")
            if stage == "fit":
                will_have.add("trainable")
        elif stage == "train":
            for name in ("recursive_train.py", "train_student.py",
                         "export_model.py", "review_students.py",
                         "serve_student.ps1"):
                if not os.path.isfile(os.path.join(SCRIPTS, name)):
                    blockers.append("scripts\\%s missing" % name)
            if not os.path.isfile(PY):
                blockers.append("venv python missing at %s" % _rel(PY))
            if "trainable" not in will_have:
                blockers.append("nothing to train on — convert/fit first")
            bin_dir = os.path.join(ROOT, "builds", "cuda", "bin")
            for exe in ("llama-imatrix.exe", "llama-quantize.exe"):
                if not os.path.isfile(os.path.join(bin_dir, exe)):
                    warnings.append("%s missing — the loop cannot build the "
                                    "iq4_xs card weights" % exe)
            if not os.path.isfile(os.path.join(ROOT, "cache", "quants",
                                               "calib.txt")):
                warnings.append("no calibration text cache\\quants\\calib.txt — "
                                "the iq4_xs quant is skipped")
    # The gate is the teacher grading the students, so it stops the TRAINING
    # step, never ingestion: an item whose runner is not going to train is still
    # worth pulling, converting, verifying and fitting.
    if not _port_open(8080):
        gate_blockers.append("teacher endpoint :8080 is down — the review step "
                             "would score nothing, so the gate cannot pass")
    return blockers, gate_blockers, warnings


def queue_plan(data=None, settings=None, remote=False):
    """What the queue will do, item by item, without doing any of it.

    The point is being able to check the queue behaves *before* paying for a
    multi-GB pull or a multi-hour training run: the stages come from the same
    ladder the runner walks (stage_steps), and each item's preconditions are
    checked against this machine — files, tooling, the teacher port. Read-only.

    `remote=True` additionally asks the hub what a pull would fetch (file count
    and bytes) for items that are about to pull, because "how big is this?"
    cannot be answered from disk. It is opt-in so the default plan stays offline.
    """
    data = data or load_registry()
    pend = queue_pending(data)
    act = active_repo()
    split = (settings or {}).get("split") or "full"
    items = []
    for pos, e in enumerate(pend):
        repo = e["id"]
        live = act == repo
        steps = stage_steps(e)
        blockers, gate_blockers, warnings = _preflight(e, steps, repo)
        pull_info = None
        if remote and "pull" in steps:
            try:
                info = api_info(repo)
                files, matched = pull_files(info, split)
                want = sum((s.get("size") or 0)
                           for s in (info.get("siblings") or [])
                           if s.get("rfilename") in files)
                pull_info = {"files": len(files),
                             "bytes": want or None,
                             "bytes_human": human(want) if want else None,
                             "split_declared": matched,
                             "big": len(files) > BIG_PULL_FILES
                             or want > BIG_PULL_BYTES}
                if pull_info["big"]:
                    warnings.append(
                        "a pull would fetch %d file(s)%s — a corpus, not a "
                        "training set; it needs --allow-big-pull"
                        % (len(files), (", %s" % human(want)) if want
                           else " (size not reported by the card)"))
                elif not matched:
                    warnings.append(
                        "the card declares no %r split — a pull would fetch all "
                        "%d declared file(s)" % (split, len(files)))
            except Exception as exc:  # noqa: BLE001 — offline/gated is not fatal
                pull_info = {"error": str(exc)[:140]}
                warnings.append("could not read the card for pull sizes: %s"
                                % str(exc)[:80])
        items.append({
            "pos": pos, "id": repo, "state": e.get("state"),
            "stalled": stalled(e), "locked": bool(live),
            "runner_pid": _runner_pid(repo) if live else None,
            "next_step": ("stalled — resumes at train" if stalled(e)
                          else NEXT_STEP.get(e.get("state") or "", "?")),
            "steps": (["wait for the live runner"] if live else steps) +
                     ["review + gate", "delete the local copy (only on a pass)"],
            "blockers": blockers, "gate_blockers": gate_blockers,
            "warnings": warnings, "pull": pull_info,
            "evidence": _evidence(e, repo)})
    first = items[0] if items else None
    return {
        "updated_utc": now(), "pending": len(pend),
        "state_counts": queue_summary(data)["by_state"],
        "next": first["id"] if first else None,
        "settings": settings or {}, "items": items,
        # runnable  = the item can start (pull -> convert -> verify -> fit)
        # can_train = it can also finish, which needs the teacher for the gate
        "runnable": bool(first) and not first["blockers"],
        "can_train": bool(first) and not (first["blockers"]
                                          or first["gate_blockers"])}




def queue_run(split="full", limit=0, map_="auto", max_len=1536, max_steps=-1,
              sample=40, min_pass=0.6, log=print, force=False, allow_big=False):
    """One dataset at a time: pull -> convert -> verify -> fit -> train -> gate
    -> learned_verified -> delete the local copy -> next dataset.

    Stops (deleting nothing) the moment a gate does not pass: keeping the local
    copy is what makes another attempt possible, so it is only released on a
    verified win.

    Before touching an item it prints the stages it is about to run — the same
    list `--queue-plan` prints, from `stage_steps` — and refuses to start when a
    precondition fails (`force`, i.e. `--force`, overrides). An item that is
    already being worked on is waited for, never duplicated.
    """
    while True:
        pending = queue_pending()
        if not pending:
            log("[hf] queue empty — nothing left to learn from")
            return 0
        e, repo = pending[0], pending[0]["id"]
        if active_repo() == repo:
            pid = _runner_pid(repo)
            if not pid:
                log("[hf] %s is already being worked on and its runner cannot be "
                    "named — leaving it alone" % repo)
                return 0
            log("[hf] %s is already being worked on (pid %d) — waiting for it "
                "rather than starting a second runner" % (repo, pid))
            wait_for_pid(pid, log=log)
            continue                      # re-read the state and re-plan
        if stalled(e):
            log("[hf] %s: the runner that had this is gone (state was %s) — "
                "resuming where its corpus got to" % (repo, e.get("state")))
        steps = stage_steps(e)
        blockers, gate_blockers, warnings = _preflight(e, steps, repo)
        log("[hf] === queue: %s (state=%s, %d pending) ==="
            % (repo, e.get("state"), len(pending)))
        log("[hf] %s: %s" % (repo, " -> ".join(steps + ["review + gate"])))
        for w in warnings:
            log("[hf]   warn: %s" % w)
        if blockers and not force:
            # Refusing beats spending hours and then failing the gate.
            for b in blockers:
                log("[hf]   BLOCKER: %s" % b)
            log("[hf] refusing to run %s — resolve the above, or re-run with "
                "--force to try anyway" % repo)
            return 3
        stage_fns = {"pull": lambda: pull(repo, split=split, log=log,
                                          allow_big=allow_big),
                     "convert": lambda: convert(repo, limit=limit, map_=map_,
                                                log=log),
                     "verify": lambda: verify(repo, max_len=max_len, log=log),
                     "fit": lambda: fit(repo, max_len=max_len, log=log)}
        for stage in steps:
            if stage == "train":
                # Ingestion is done and stays done; the gate cannot pass without
                # the teacher, so stop here rather than after hours of training.
                # The corpus is fitted on disk, so a later run resumes at train.
                if gate_blockers and not force:
                    for b in gate_blockers:
                        log("[hf]   BLOCKER: %s" % b)
                    log("[hf] %s: ingested and fitted — refusing to start the "
                        "training loop until the above is resolved (--force to "
                        "run it anyway)" % repo)
                    return 3
                break
            log("[hf] === %s: %s ===" % (repo, stage))
            stage_fns[stage]()
        after = entry_of(load_registry(), repo) or {}
        if (after.get("state") or "") == "not_trainable":
            # The mapping refused the data and said why — that is a verdict about
            # the dataset, so move on instead of failing the run.
            log("[hf] %s: not trainable as SFT — next dataset. drops: %s"
                % (repo, json.dumps(after.get("dropped") or {})))
            continue
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






