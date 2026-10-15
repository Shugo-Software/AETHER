#!/usr/bin/env python3
"""Phase 11 — ingest external HF datasets into AETHER's dataset schema.

Two shapes, deliberately kept apart:

  --source fable        MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x
                        -> datasets/<tag>.jsonl       (TRAINING: filtered subset)
  --source openai-math  FineEnvs/openai-math
                        -> datasets/eval/<tag>.jsonl  (EVAL ONLY — never trained)

Records match `generate_dataset.py`: {id, source, recipe, system, prompt,
completion, reasoning?, provenance, meta}, with a manifest beside the JSONL
(params, counts, drop reasons, char stats, sha256, source revision).

Why filtering matters for `fable`: those rows are agentic SWE traces (repo-scale
authoring tasks, tool schemas, multi-step scaffolding). A 2.6B student cannot do
them, so the ingester keeps only self-contained rows — one user turn, no tool
calls, bounded length, no repo-scale markers — and records every drop reason in
the manifest so the subset stays auditable.

`openai-math` is Lean4/Harbor environment scaffolding, not prompt->answer data:
it is ingested for measurement (approach/explanation quality vs a reference
solution), never for training, and lives under datasets/eval/ which the training
control plane refuses to launch.

Usage:
  python scripts/ingest_hf_dataset.py --source fable --limit 500
  python scripts/ingest_hf_dataset.py --source openai-math
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(ROOT, "datasets")
EVAL_DIR = os.path.join(DATASETS, "eval")

FABLE_REPO = "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x"
MATH_REPO = "FineEnvs/openai-math"

# Repo-scale / agentic scaffolding that a 2.6B student cannot act on.
AGENTIC_MARKERS = (
    "seed_spec", "wave4_usecases", "you are authoring seeds",
    "autonomous coding agent", "repo root", "task.toml", "dockerfile",
    "pull request", "git commit", "run the tests", "test_maintlog",
)
MATH_SYSTEM = (
    "You are a mathematician. Given a formal proof problem, give a short, "
    "concrete plan: the key objects, the main theorem or invariant to use, and "
    "the first two proof steps. No preamble, no restating the question."
)


def ensure_utf8_mode():
    """Re-exec under UTF-8 mode if needed.

    `FineEnvs/openai-math` ships a custom `harbor` loader that reads its files
    with the *platform* default codec — on Windows that is cp1252, so ingestion
    dies on valid UTF-8. Python's UTF-8 mode fixes it; enabling it early and
    re-executing keeps `python scripts/ingest_hf_dataset.py ...` working as
    documented instead of requiring a magic env var.
    """
    if sys.flags.utf8_mode or os.environ.get("AETHER_UTF8_REEXEC"):
        return
    env = dict(os.environ, PYTHONUTF8="1", AETHER_UTF8_REEXEC="1")
    # subprocess with a LIST keeps the spaced path ("Program Prototype") intact;
    # os.execve would let the interpreter re-split it — the classic trap here.
    raise SystemExit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)


def pin_caches():
    """All HF caches on the repo drive (C: is critically small — plan.md)."""
    hf = os.path.join(ROOT, "cache", "hf")
    os.makedirs(hf, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf)
    os.environ.setdefault("HF_HUB_CACHE", hf)
    os.environ.setdefault("HF_DATASETS_CACHE", hf)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TEMP", os.path.join(ROOT, "cache", "temp"))
    os.environ.setdefault("TMP", os.environ["TEMP"])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args():
    ap = argparse.ArgumentParser(description="Ingest an external HF dataset")
    ap.add_argument("--source", required=True, choices=["fable", "openai-math"])
    ap.add_argument("--tag", default=None,
                    help="output name (default: fable-pilot / openai-math)")
    ap.add_argument("--split", default=None,
                    help="override the split (fable: train|full, math: test)")
    ap.add_argument("--limit", type=int, default=500,
                    help="max records to WRITE (fable default 500)")
    ap.add_argument("--scan-limit", type=int, default=20000,
                    help="max source rows to examine")
    ap.add_argument("--max-prompt-chars", type=int, default=4000)
    ap.add_argument("--max-completion-chars", type=int, default=4000)
    ap.add_argument("--max-fences", type=int, default=3,
                    help="drop rows whose prompt has more fenced blocks than this")
    return ap.parse_args()


def _text(msg):
    """Message content as plain text (handles the list-of-parts shape)."""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(part.get("text", "") for part in c
                         if isinstance(part, dict))
    return ""


def _reasoning(msg):
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def ingest_fable(args, seen, drops):
    """Self-contained rows only; every drop is counted by reason."""
    from datasets import load_dataset
    split = args.split or "train"
    stream = load_dataset(FABLE_REPO, split=split, streaming=True)
    records, scanned = [], 0
    for row in stream:
        scanned += 1
        if scanned > args.scan_limit or len(records) >= args.limit:
            break
        if row.get("tools"):
            drops["tools_present"] += 1
            continue
        msgs = [m for m in (row.get("messages") or []) if isinstance(m, dict)]
        users = [m for m in msgs if m.get("role") == "user"]
        assts = [m for m in msgs if m.get("role") == "assistant"]
        if len(users) != 1 or not assts:
            drops["not_single_user_turn"] += 1
            continue
        if any(m.get("tool_calls") for m in assts):
            drops["assistant_tool_calls"] += 1
            continue
        system = next((_text(m) for m in msgs if m.get("role") == "system"), "")
        prompt = _text(users[0]).strip()
        completion = _text(assts[-1]).strip()
        reason = _reasoning(assts[-1])
        if not prompt or not completion:
            drops["empty_side"] += 1
            continue
        blob = (prompt + "\n" + completion).lower()
        if any(marker in blob for marker in AGENTIC_MARKERS):
            drops["agentic_marker"] += 1
            continue
        if prompt.count("```") > args.max_fences \
                or completion.count("```") > args.max_fences:
            drops["too_many_code_fences"] += 1
            continue
        if len(prompt) > args.max_prompt_chars:
            drops["prompt_too_long"] += 1
            continue
        if len(completion) > args.max_completion_chars:
            drops["completion_too_long"] += 1
            continue
        key = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        if key in seen:
            drops["duplicate_prompt"] += 1
            continue
        seen.add(key)
        rec = {
            "id": "f-%06d" % len(records),
            "source": "fable-5.1-reasoning",
            "recipe": "fable_code" if "```" in prompt else "fable_reason",
            "system": system or ("You are a concise, precise assistant. "
                                 "No preamble, no meta-commentary."),
            "prompt": prompt,
            "completion": completion,
            "provenance": {"repo": FABLE_REPO, "split": split,
                           "scan_index": scanned - 1},
            "meta": {"ingested_utc": now(), "prompt_chars": len(prompt),
                     "completion_chars": len(completion),
                     "reasoning_chars": len(reason or "")},
        }
        if reason:
            rec["reasoning"] = reason      # format_record renders <think>…</think>
        records.append(rec)
    return records, scanned


def ingest_math(args, seen, drops):
    """Eval-only: prompts are trimmed to the students' context budget."""
    from datasets import load_dataset
    stream = load_dataset(MATH_REPO, split=args.split or "test", streaming=True)
    records = []
    for i, row in enumerate(stream):
        if len(records) >= args.limit:
            break
        meta = row.get("metadata") or {}
        instr = (row.get("instruction") or "").strip()
        desc = (row.get("description") or "").strip()
        prompt = instr + ("\n\nContext: " + desc if desc else "")
        if len(prompt) > args.max_prompt_chars:
            drops["prompt_truncated"] += 1
            prompt = prompt[:args.max_prompt_chars] + "\n\n[... truncated ...]"
        location = row.get("location") or "?"
        records.append({
            "id": "m-%03d" % i,
            "source": "openai-math",
            "recipe": "proof_approach",
            "system": MATH_SYSTEM,
            "prompt": prompt,
            "completion": ("[reference: %s/solution — Lean4, oracle=%s, "
                           "difficulty=%s]"
                           % (location, meta.get("oracle_result"),
                              meta.get("difficulty"))),
            "provenance": {"repo": MATH_REPO, "split": args.split or "test",
                           "location": location},
            "meta": {"ingested_utc": now(), "name": row.get("name"),
                     "keywords": row.get("keywords"),
                     "prompt_chars": len(prompt), "metadata": meta},
        })
    return records, len(records)


def revision(repo):
    """Dataset revision (commit sha) — pins exactly what we ingested."""
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo).sha
    except Exception:  # noqa: BLE001 — offline / gated: revision is optional
        return None


def write_output(args, records, drops, scanned):
    repo = FABLE_REPO if args.source == "fable" else MATH_REPO
    if args.source == "fable":
        out_dir, tag = DATASETS, (args.tag or "fable-pilot")
        purpose = "training"
    else:
        out_dir, tag = EVAL_DIR, (args.tag or "openai-math")
        purpose = "eval-only — never used for training"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, tag + ".jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    comp = sorted(len(r["completion"]) for r in records) or [0]
    manifest = {
        "tag": tag,
        "created_utc": now(),
        "purpose": purpose,
        "params": dict(vars(args)),
        "source": {"repo": repo, "split": args.split or "default",
                   "revision": revision(repo)},
        "counts": {"scanned": scanned, "written": len(records),
                   "dropped": {k: v for k, v in sorted(drops.items())}},
        "completion_chars": {"min": comp[0], "median": comp[len(comp) // 2],
                             "max": comp[-1]},
        "output": {"jsonl": os.path.relpath(out_path, ROOT),
                   "sha256": sha256_file(out_path)},
    }
    man_path = os.path.join(out_dir, tag + ".manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return out_path, man_path, manifest


EVAL_README = """# `datasets/eval/` — measurement sets, never training data

These datasets exist to **measure** the students, not to teach them:

| File | Source | Why it is not training data |
|---|---|---|
| `openai-math.jsonl` | `FineEnvs/openai-math` (apache-2.0) | The source is Lean4/Harbor **RL environment** scaffolding (Dockerfile, `Challenge.lean`, `grade.py`), not prompt→answer pairs. It is used to score *approach/explanation quality* of a student that cannot possibly prove research-level theorems. |

The training control plane refuses anything under this directory: datasets are
launched by bare filename from `datasets/`, so `eval/openai-math.jsonl` cannot be
selected from the dashboard or the API.

Ingest (re)creates these files plus their manifests:

```powershell
python scripts\\ingest_hf_dataset.py --source openai-math
```
"""


def main():
    args = parse_args()
    ensure_utf8_mode()
    pin_caches()
    import collections
    drops = collections.Counter()
    seen = set()
    if args.source == "fable":
        records, scanned = ingest_fable(args, seen, drops)
    else:
        records, scanned = ingest_math(args, seen, drops)
    if not records:
        print("[ingest] nothing matched. drops: %s"
              % (dict(drops) or "{}"), file=sys.stderr)
        if args.source == "fable":
            print(
                "[ingest] NOTE: every row of Fable-5.1 (measured: 700/700 rows "
                "across both splits) is an agentic tool-calling transcript — "
                "multi-turn assistant/tool exchanges with 45-85 messages, "
                "`tools` schemas, and repo-scale authoring preambles. There are "
                "no self-contained prompt/answer pairs to select, and in the "
                "`full` split the final assistant turn is usually empty. Do not "
                "train a small student on this without a teacher rewrite step; "
                "use the teacher recipes in generate_dataset.py instead. "
                "See docs/dataset-generation.md.", file=sys.stderr)
        return 1
    out_path, man_path, manifest = write_output(args, records, drops, scanned)
    if args.source == "openai-math":
        with open(os.path.join(EVAL_DIR, "README.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(EVAL_README)
    print("[ingest] source=%s  purpose=%s" % (args.source, manifest["purpose"]))
    print("[ingest] scanned=%d written=%d dropped=%s"
          % (scanned, len(records), dict(drops) or "{}"))
    print("[ingest] completion chars: median %d (min %d / max %d)"
          % (manifest["completion_chars"]["median"],
             manifest["completion_chars"]["min"],
             manifest["completion_chars"]["max"]))
    print("[ingest] jsonl    -> %s" % out_path)
    print("[ingest] manifest -> %s" % man_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
