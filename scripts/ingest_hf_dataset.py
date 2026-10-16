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
  python scripts/ingest_hf_dataset.py --from-local <dir> --repo org/name --map chat
  python scripts/ingest_hf_dataset.py --source fable           # --limit 0 = all rows
  python scripts/ingest_hf_dataset.py --source openai-math

`--limit` / `--scan-limit` default to 0 (no cap): the pulled copy is on G: and
free to read, so a cap there only loses data. Give them a number to build a small
pilot corpus on purpose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aether_env  # noqa: E402 - caches/temp stay on the repo drive (never C:)
aether_env.pin_caches()

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
    """All HF caches on the repo drive.

    Delegates to `aether_env`, the single place that keeps every cache and temp
    path off C: (see its docstring: a default cache there already cost a 289 GB
    pull, which died with ENOSPC on the row count).
    """
    import aether_env
    return aether_env.pin_caches()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    """Run-level UTC stamp — for the manifest, never inside a row.

    A per-row stamp (`meta.ingested_utc`, dropped 2026-10-15) made every
    `--convert` produce a different sha256 for identical input, so the manifest's
    fingerprint could not tell a re-run from real drift. Rows are now byte-exact
    for a fixed input + flags; the run time is the manifest's `created_utc`.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args():
    ap = argparse.ArgumentParser(description="Ingest an external HF dataset")
    ap.add_argument("--source", default=None, choices=["fable", "openai-math"],
                    help="pinned sources (fable = the Fable-5.1 training path, "
                         "openai-math = eval only). Omit and use --repo to link "
                         "any dataset by id/URL.")
    ap.add_argument("--repo", default=None,
                    help="any HF dataset id (org/name) to ingest")
    ap.add_argument("--map", dest="map_", default="auto",
                    choices=["auto", "chat", "flat"],
                    help="chat: rows carry a `messages` list (one record per "
                         "assistant turn). flat: plain prompt/answer columns.")
    ap.add_argument("--turn-mode", default="last", choices=["last", "all"],
                    help="chat mapping: emit only the final assistant turn "
                         "(clearest target) or every assistant turn")
    ap.add_argument("--agentic", choices=["keep", "drop"], default="keep",
                    help="agentic rows (a tool schema plus tool-call turns): "
                         "'keep' extracts their plain assistant answers and marks "
                         "them in provenance (default), 'drop' refuses the row")
    ap.add_argument("--from-local", default=None,
                    help="dir of a pulled copy — read it instead of the hub")
    ap.add_argument("--from-local-files", default=None,
                    help="comma-separated files inside --from-local to use")
    ap.add_argument("--local-pattern", default=None,
                    help="glob inside --from-local (default: the largest .jsonl)")
    ap.add_argument("--eval-only", action="store_true",
                    help="write to datasets/eval/ (never trained on)")
    ap.add_argument("--tag", default=None,
                    help="output name (default: fable-pilot / openai-math)")
    ap.add_argument("--split", default=None,
                    help="override the split (fable: train|full, math: test)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max records to WRITE (0 = all rows; the local copy is "
                         "free to read, so a cap here only loses data)")
    ap.add_argument("--scan-limit", type=int, default=0,
                    help="max source rows to examine (0 = all)")
    ap.add_argument("--max-prompt-chars", type=int, default=4000)
    ap.add_argument("--max-completion-chars", type=int, default=4000)
    ap.add_argument("--max-fences", type=int, default=3,
                    help="drop rows whose prompt has more fenced blocks than this")
    ap.add_argument("--max-reasoning-chars", type=int, default=2000,
                    help="drop a turn's reasoning above this (the answer is the "
                         "learning target; counted as reasoning_trimmed)")
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
        if (args.scan_limit and scanned > args.scan_limit) or \
                (args.limit and len(records) >= args.limit):
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
            "meta": {"prompt_chars": len(prompt),
                     "completion_chars": len(completion),
                     "reasoning_chars": len(reason or "")},
        }
        if reason:
            rec["reasoning"] = reason      # format_record renders <think>…</think>
        records.append(rec)
    return records, scanned


FLAT_PAIRS = (
    ("prompt", "completion"), ("instruction", "response"), ("instruction", "output"),
    ("instruction", "answer"), ("question", "answer"), ("question", "response"),
    ("problem", "solution"), ("query", "response"), ("input", "output"),
    ("prompt", "response"), ("prompt", "answer"), ("user", "assistant"),
)
DEFAULT_SYSTEM = ("You are a concise, precise assistant. No preamble, "
                  "no meta-commentary.")


def slug_tag(name, limit=48):
    """Filesystem-safe tag from a repo name (mirrors hf_datasets.slug_tag)."""
    out, dash = [], False
    for ch in (name or "").lower():
        if ch.isalnum():
            out.append(ch)
            dash = False
        elif not dash:
            out.append("-")
            dash = True
    return "".join(out).strip("-")[:limit].rstrip("-")


def normalize_repo(ref):
    """Reduce a hub URL or slug to `org/name`."""
    s = (ref or "").strip()
    for pre in ("https://", "http://"):
        if s.lower().startswith(pre):
            s = s[len(pre):]
    if s.lower().startswith("www."):
        s = s[4:]
    if s.lower().startswith("huggingface.co/"):
        s = s[len("huggingface.co/"):]
    if s.lower().startswith("api/datasets/"):
        s = s[len("api/datasets/"):]
    if s.lower().startswith("datasets/"):
        s = s[len("datasets/"):]
    s = s.split("?")[0].split("#")[0]
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 3 and parts[2] == "tree":
        parts = parts[:2]
    if len(parts) < 2:
        raise SystemExit("that is not a dataset id: %r" % ref)
    return "/".join(parts[:2])


def local_jsonl_files(args):
    """Data files to read from a pulled copy (JSONL or Parquet).

    Card entries are globs (`main/train-*`) and many datasets ship Parquet, so
    patterns are expanded and both formats are accepted. When several files match
    and none was named explicitly, the largest is used: a repo that ships `5k/`
    and `10k/` of the same data should not be trained on twice.
    """
    import glob as _glob
    root = args.from_local
    if args.from_local_files:
        pats = [p.strip() for p in args.from_local_files.split(",") if p.strip()]
    elif args.local_pattern:
        pats = [args.local_pattern]
    else:
        pats = ["**/*.jsonl", "**/*.parquet"]
    out, seen = [], set()
    for pat in pats:
        hits = _glob.glob(os.path.join(root, pat), recursive=True)
        if not hits:
            literal = os.path.join(root, pat)
            hits = [literal] if os.path.isfile(literal) else []
        for f in sorted(hits):
            if os.path.isfile(f) and f not in seen:
                seen.add(f)
                out.append(f)
    if not args.from_local_files and not args.local_pattern and len(out) > 1:
        # A repo that ships the same rows twice (5k/ and 10k/) must not be
        # ingested twice — but every SHARD of the chosen split has to be read, or
        # a sharded dataset silently loses rows (20 parquet files became one).
        groups = {}
        for f in out:
            groups.setdefault(os.path.dirname(f), []).append(f)

        def total(paths):
            return sum(os.path.getsize(p) for p in paths)
        best = max(groups, key=lambda d: total(groups[d]))
        if len(groups) > 1:
            print("[ingest] %d splits on disk (%s) — reading %s (%d file(s))"
                  % (len(groups), ", ".join(sorted(groups)), best or "(root)",
                     len(groups[best])), flush=True)
        out = sorted(groups[best])
    if not out:
        raise SystemExit("no .jsonl/.parquet found under %s" % root)
    return out


def iter_source(args, repo):
    """Yield (index, row) from the pulled copy when present, else a hub stream."""
    if args.from_local:
        files = local_jsonl_files(args)
        print("[ingest] local files: %s"
              % ", ".join(os.path.relpath(f, ROOT) for f in files), flush=True)
        i = 0
        for path in files:
            if path.lower().endswith(".parquet"):
                # Parquet is the hub's default format: read it with `datasets`
                # rather than pretending every dataset is JSONL.
                from datasets import Dataset
                for row in Dataset.from_parquet(path):
                    yield i, dict(row)
                    i += 1
                continue
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        row = None       # counted as unparsable_json downstream
                    yield i, row
                    i += 1
        return
    from datasets import load_dataset
    stream = load_dataset(repo, split=args.split or "train", streaming=True)
    for i, row in enumerate(stream):
        yield i, row


def render_transcript(msgs):
    """Prior turns as compact text — the context a turn is answering."""
    lines = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") == "system":
            continue
        txt = _text(m).strip()
        if not txt:
            continue
        lines.append("%s: %s"
                     % ("User" if m.get("role") == "user" else "Assistant", txt))
    return "\n\n".join(lines)


def _passes_filters(prompt, completion, args, drops):
    """Length/fence/marker gates shared by every mapping."""
    blob = (prompt + "\n" + completion).lower()
    if any(marker in blob for marker in AGENTIC_MARKERS):
        drops["agentic_marker"] += 1
        return False
    if prompt.count("```") > args.max_fences \
            or completion.count("```") > args.max_fences:
        drops["too_many_code_fences"] += 1
        return False
    if len(prompt) > args.max_prompt_chars:
        drops["prompt_too_long"] += 1
        return False
    if len(completion) > args.max_completion_chars:
        drops["completion_too_long"] += 1
        return False
    return True


def map_chat(row, args, seen, drops, repo, split, idx):
    """A chat/agentic transcript -> one record per assistant turn.

    This is the self-contained-pair extractor: a whole agentic trace is not a
    prompt->answer pair, but each assistant turn inside it is (context -> that
    turn), which is what a 2.6B student can actually learn.
    """
    if row is None:
        drops["unparsable_json"] += 1
        return []
    # An agentic row (a tool schema, tool-call turns, tool results) is still worth
    # mining: its PLAIN assistant turns are self-contained answers, and every
    # tool-call turn is skipped below. `--agentic drop` restores the old refusal.
    # The caveat is recorded per record (provenance.agentic) and counted in the
    # manifest, because such a prompt shows tool results the student cannot
    # produce itself.
    agentic = bool(row.get("tools")) or any(
        (m or {}).get("tool_calls")
        for m in (row.get("messages") or []) if isinstance(m, dict))
    if agentic and args.agentic == "drop":
        drops["tools_present"] += 1
        return []
    msgs = [m for m in (row.get("messages") or []) if isinstance(m, dict)]
    turns = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if not any(m.get("role") == "user" for m in msgs) or not turns:
        drops["no_user_or_assistant"] += 1
        return []
    if args.turn_mode == "last":
        turns = turns[-1:]
    system = next((_text(m) for m in msgs if m.get("role") == "system"), "")
    out = []
    for t in turns:
        m = msgs[t]
        if m.get("tool_calls"):
            drops["assistant_tool_calls"] += 1
            continue
        completion = _text(m).strip()
        if not completion:
            drops["empty_completion"] += 1
            continue
        prompt = render_transcript(msgs[:t]).strip()
        if not prompt:
            drops["empty_prompt"] += 1
            continue
        reason = _reasoning(m)
        if reason and len(reason) > args.max_reasoning_chars:
            drops["reasoning_too_long"] += 1
            reason = None
        if not _passes_filters(prompt, completion, args, drops):
            continue
        key = hashlib.sha1(("%s\x00%s" % (prompt, completion[:200]))
                           .encode("utf-8")).hexdigest()
        if key in seen:
            drops["duplicate_pair"] += 1
            continue
        seen.add(key)
        rec = {
            "id": "h-%06d" % len(out),
            "source": "hf-" + slug_tag(repo.split("/")[-1], 24),
            "recipe": "chat_code" if "```" in prompt else "chat_reason",
            "system": system or DEFAULT_SYSTEM,
            "prompt": prompt,
            "completion": completion,
            "provenance": {"repo": repo, "split": split, "row_index": idx,
                           "turn_index": t, "mapping": "chat",
                           "turn_mode": args.turn_mode},
            "meta": {"prompt_chars": len(prompt),
                     "completion_chars": len(completion),
                     "reasoning_chars": len(reason or ""),
                     "messages_before": t, "messages_total": len(msgs)},
        }
        if agentic:
            # Per-record, so the caveat travels with the data and the corpus can
            # be filtered on it later.
            rec["provenance"]["agentic"] = True
        if reason:
            rec["reasoning"] = reason    # format_record renders <think>…</think>
        out.append(rec)
    return out


def map_flat(row, args, seen, drops, repo, split, idx):
    """A plain table -> prompt/completion from the first recognised column pair."""
    if row is None:
        drops["unparsable_json"] += 1
        return []
    cols = None
    for pc, cc in FLAT_PAIRS:
        a, b = row.get(pc), row.get(cc)
        if isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip():
            cols = (pc, cc)
            break
    if cols is None:
        drops["no_prompt_pair"] += 1
        return []
    prompt, completion = row[cols[0]].strip(), row[cols[1]].strip()
    if not _passes_filters(prompt, completion, args, drops):
        return []
    key = hashlib.sha1(("%s\x00%s" % (prompt, completion[:200]))
                       .encode("utf-8")).hexdigest()
    if key in seen:
        drops["duplicate_pair"] += 1
        return []
    seen.add(key)
    system = row.get("system") if isinstance(row.get("system"), str) else ""
    reason = None
    for k in ("reasoning", "thinking", "reasoning_content", "cot"):
        if isinstance(row.get(k), str) and row[k].strip():
            reason = row[k].strip()
            break
    if reason and len(reason) > args.max_reasoning_chars:
        drops["reasoning_too_long"] += 1
        reason = None
    rec = {
        "id": "h-%06d" % len(seen),
        "source": "hf-" + slug_tag(repo.split("/")[-1], 24),
        "recipe": "flat_pair",
        "system": system or DEFAULT_SYSTEM,
        "prompt": prompt,
        "completion": completion,
        "provenance": {"repo": repo, "split": split, "row_index": idx,
                       "mapping": "flat", "columns": list(cols)},
        "meta": {"prompt_chars": len(prompt),
                 "completion_chars": len(completion),
                 "reasoning_chars": len(reason or "")},
    }
    if reason:
        rec["reasoning"] = reason
    return [rec]


def ingest_generic(args, seen, drops):
    """Ingest any linked dataset with an explicit mapping."""
    repo = args.repo
    split = args.split or "train"
    mapping = args.map_
    if mapping in (None, "", "auto"):
        # Detect from the first row instead of guessing: `--map chat` on a flat
        # dataset (gsm8k's question/answer parquet) drops every single row, which
        # looks exactly like "the dataset is unusable" when it is not.
        mapping = None
    records, scanned = [], 0
    for idx, row in iter_source(args, repo):
        scanned += 1
        if (args.scan_limit and scanned > args.scan_limit) or \
                (args.limit and len(records) >= args.limit):
            break
        if mapping is None and isinstance(row, dict):
            mapping = "chat" if isinstance(row.get("messages"), list) else "flat"
            print("[ingest] auto-detected mapping: %s" % mapping, flush=True)
        if mapping == "chat":
            recs = map_chat(row, args, seen, drops, repo, split, idx)
        else:
            recs = map_flat(row, args, seen, drops, repo, split, idx)
        for r in recs:
            if args.limit and len(records) >= args.limit:
                break
            r["id"] = "h-%06d" % len(records)
            records.append(r)
    return records, scanned


def ingest_math(args, seen, drops):
    """Eval-only: prompts are trimmed to the students' context budget."""
    from datasets import load_dataset
    stream = load_dataset(MATH_REPO, split=args.split or "test", streaming=True)
    records = []
    for i, row in enumerate(stream):
        if args.limit and len(records) >= args.limit:
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
            "meta": {"name": row.get("name"),
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
    if args.source == "fable":
        repo, out_dir = FABLE_REPO, DATASETS
        tag = args.tag or "fable-pilot"
        purpose = "training"
    elif args.source == "openai-math":
        repo, out_dir = MATH_REPO, EVAL_DIR
        tag = args.tag or "openai-math"
        purpose = "eval-only — never used for training"
    else:
        repo = normalize_repo(args.repo)
        out_dir = EVAL_DIR if args.eval_only else DATASETS
        tag = args.tag or slug_tag(repo.split("/")[-1])
        purpose = ("eval-only — never used for training" if args.eval_only
                   else "training")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, tag + ".jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    comp = sorted(len(r["completion"]) for r in records) or [0]
    # Rows mined out of agentic traces (tool schemas / tool-call turns): their
    # plain answers are kept, but their prompts show tool results a student cannot
    # produce, so the count travels with the manifest.
    agentic = sum(1 for r in records
                  if (r.get("provenance") or {}).get("agentic"))
    manifest = {
        "tag": tag,
        "created_utc": now(),
        "purpose": purpose,
        "agentic_rows": agentic,
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
    if agentic:
        print("[ingest] NOTE: %d of %d rows come from agentic traces — their "
              "prompts contain tool results the student cannot produce itself; "
              "weigh the gate accordingly (--agentic drop excludes them)."
              % (agentic, len(records)), flush=True)
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
    if not args.source and not args.repo:
        raise SystemExit("give --source (pinned) or --repo (any dataset id/URL)")
    ensure_utf8_mode()
    pin_caches()
    import collections
    drops = collections.Counter()
    seen = set()
    if args.source == "fable":
        records, scanned = ingest_fable(args, seen, drops)
    elif args.source == "openai-math":
        records, scanned = ingest_math(args, seen, drops)
    else:
        args.repo = normalize_repo(args.repo)
        records, scanned = ingest_generic(args, seen, drops)
    if not records:
        print("[ingest] nothing matched. drops: %s"
              % (dict(drops) or "{}"), file=sys.stderr)
        if args.source == "fable":
            print(
                "[ingest] NOTE: Fable-5.1 rows are agentic tool-calling "
                "transcripts (multi-turn assistant/tool exchanges, `tools` "
                "schemas, repo-scale preambles). Whole rows are not "
                "prompt/answer pairs; only the per-turn mapping is. Use "
                "`--repo MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x "
                "--map chat` to extract context->turn pairs, and check the "
                "drop counts above. See docs/dataset-generation.md.",
                file=sys.stderr)
        else:
            print("[ingest] for --map flat, recognised column pairs are: %s"
                  % ", ".join("%s/%s" % pc for pc in FLAT_PAIRS[:6]),
                  file=sys.stderr)
            print("[ingest] inspect one row with: python -c \"import json;"
                  "print(sorted(json.loads(open(r'<file>').readline()).keys()))\"",
                  file=sys.stderr)
        return 1
    out_path, man_path, manifest = write_output(args, records, drops, scanned)
    if args.source == "openai-math":
        with open(os.path.join(EVAL_DIR, "README.md"), "w",
                  encoding="utf-8") as fh:
            fh.write(EVAL_README)
    print("[ingest] source=%s  purpose=%s"
          % (manifest.get("source", {}).get("repo") or args.source,
             manifest["purpose"]))
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
