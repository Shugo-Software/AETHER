#!/usr/bin/env python3
"""Phase 6 — SamurAI dataset generation: teacher transforms + pure generation.

Two sources (the user's Phase 6 requirement — NOT just generation):

  A. starcoderdata (bigcode/starcoderdata, gated-ToU PASSED via our HF account)
     python code files -> teacher transforms (rotating recipes):
       - explain_code : concise mid-level explanation + one pitfall
       - qa_pairs     : 3 developer Q&A pairs about the code, answered from the code
       - code_review  : concrete issues (bug/style/security) + fixes
  B. pure teacher generation from built-in seed prompts.

Output: JSONL (one record per sample) + manifest.json (params, counts, sha256,
teacher identity, versions) — hash-pinned, reproduce-don't-inherit discipline.

Design notes:
  * chat_template_kwargs enable_thinking=false (verified on this llama-server):
    2 tokens vs 64 for trivial prompts — data-gen runs ~30x cheaper; --think opts in.
  * Deterministic row ORDER (seeded) over the shard; append/resume via --tag.
  * Loopback-only endpoint must already be up (scripts/start-teacher-server.ps1).

Usage:
  python scripts/generate_dataset.py --n-a 50 --n-b 25                 # pilot
  python scripts/generate_dataset.py --n-a 2000 --recipes qa_pairs     # focused batch
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SHARD = os.path.join(ROOT, "corpus", "starcoderdata", "python",
                             "train-00000-of-00059.parquet")
OUT_DIR = os.path.join(ROOT, "datasets")
ENDPOINT = "http://127.0.0.1:8080/v1"
TEACHER_MODEL = "qwen3.5-9b-teacher"

SYSTEM = ("You are a precise senior Python engineer producing high-quality "
          "training material. Use only what is given. Be concrete and correct. "
          "No preamble, no meta-commentary, no markdown code fences around your "
          "whole answer unless the recipe asks for code.")

RECIPES = {
    "explain_code":
        "Explain this Python code for a mid-level developer. Cover: (1) purpose, "
        "(2) the key functions/classes and what they do, (3) one notable pattern "
        "or idiom used, (4) one likely pitfall or bug risk when reusing it. "
        "Keep it under 200 words.\n\n<code>\n{code}\n</code>",
    "qa_pairs":
        "Write exactly 3 high-quality Q&A pairs a developer would ask about this "
        "code, then answer each accurately using only what the code shows. "
        "Format:\nQ1: ...\nA1: ...\nQ2: ...\nA2: ...\nQ3: ...\nA3: ...\n\n"
        "<code>\n{code}\n</code>",
    "code_review":
        "Review this Python code. List concrete issues only (bugs, security, "
        "correctness, serious style problems) with brief line/context references, "
        "then give the corrected code for the most important issue. If the code "
        "is clean, say so and note the single biggest maintainability risk.\n\n"
        "<code>\n{code}\n</code>",
}

# Force all HF caches onto G: (C: is critically small/volatile — plan.md).
# Must be set before the `datasets` library is imported/used.
_HF = os.path.join(ROOT, "cache", "hf")
for _v in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE"):
    os.environ.setdefault(_v, _HF)

SEEDS = [
    "Explain the difference between Python's threading and asyncio, and when each is the right choice.",
    "Write a Python function that retries a flaky network call with exponential backoff and jitter, and explain the design.",
    "What makes a Python context manager (`__enter__`/`__exit__`) safer than try/finally for resource handling? Give one non-obvious example.",
    "Describe how you would structure a small CLI tool that processes files in parallel on Windows, and why you would choose ProcessPoolExecutor over ThreadPoolExecutor.",
    "Explain what an LLM agent loop is (observe-think-act-observe) and give one example of a safe guardrail for the act step.",
    "Write a short Python snippet that atomically writes a JSON file (crash-safe), then explain why os.replace is used.",
    "What is the difference between shallow and deep copy in Python? Give an example where the shallow copy causes a real bug.",
    "Explain how prompt caching works at a high level and why it lowers the cost of repeated context.",
    "Describe one way to make an ML training script reproducible (seeds, deterministic ops) and the trade-off involved.",
    "Write a Python function that merges overlapping intervals, and walk through the edge cases it handles.",
]



# --------------------------------------------------------------------------- client

def chat(prompt: str, *, endpoint: str, model: str, max_tokens: int,
         temperature: float, think: bool, retries: int = 3) -> tuple[str, dict]:
    """One teacher call. Returns (completion, usage). Raises on final failure."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": bool(think)},
    }
    body = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                f"{endpoint}/chat/completions", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = (data["choices"][0]["message"].get("content") or "").strip()
            if not content:
                raise RuntimeError("empty completion")
            return content, data.get("usage", {})
        except Exception as e:  # noqa: BLE001 — retries then raise
            last_err = e
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"teacher call failed after {retries} attempts: {last_err}")


# ------------------------------------------------------------------------- sources

def load_source_a(shard: str, n: int, recipes: list[str], seed: int) -> list[dict]:
    """Deterministic row picks from the starcoderdata shard (bounded chars)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(shard)
    rng = random.Random(seed)
    candidates: list[dict] = []
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg)
        for row in tbl.to_pylist():
            code = row.get("content") or ""
            if 400 <= len(code) <= 9000:          # sweet spot for recipes
                candidates.append({
                    "repo": row.get("max_stars_repo_name"),
                    "path": row.get("max_stars_repo_path"),
                    "stars": row.get("max_stars_count"),
                    "code": code,
                })
        if len(candidates) >= n * 20:             # enough to sample from
            break
    rng.shuffle(candidates)
    picks = candidates[:n]
    # rotate recipes deterministically
    return [dict(p, recipe=recipes[i % len(recipes)]) for i, p in enumerate(picks)]


def load_source_b(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed ^ 0xB1B1)
    picks = []
    for i in range(n):
        prompt = SEEDS[i % len(SEEDS)] if i < len(SEEDS) else rng.choice(SEEDS)
        picks.append({"recipe": "seed_gen", "prompt": prompt})
    return picks


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_jsonl(path: str) -> tuple[int, list[int]]:
    count, lens = 0, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("completion"):
                raise RuntimeError(f"record {count} missing completion")
            lens.append(len(rec["completion"]))
            count += 1
    return count, lens


# ---------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6 dataset generation")
    ap.add_argument("--n-a", type=int, default=50, help="starcoderdata-derived samples")
    ap.add_argument("--n-b", type=int, default=25, help="pure-generation samples")
    ap.add_argument("--recipes", default="explain_code,qa_pairs,code_review")
    ap.add_argument("--shard", default=DEFAULT_SHARD)
    ap.add_argument("--endpoint", default=ENDPOINT)
    ap.add_argument("--model", default=TEACHER_MODEL)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--think", action="store_true",
                    help="enable teacher thinking (default: off, ~30x cheaper)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = ap.parse_args()

    recipes = [r.strip() for r in args.recipes.split(",") if r.strip()]
    for r in recipes:
        if r not in RECIPES:
            ap.error(f"unknown recipe {r!r}; choices: {', '.join(RECIPES)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"phase6_{args.tag}.jsonl")
    manifest_path = os.path.join(OUT_DIR, f"phase6_{args.tag}.manifest.json")

    try:
        with urllib.request.urlopen(f"{args.endpoint}/models", timeout=10) as r:
            ids = [m["id"] for m in json.loads(r.read())["data"]]
    except Exception as e:  # noqa: BLE001
        print(f"[fatal] teacher endpoint unreachable: {e}\n"
              f"        start: powershell -File {ROOT}\\scripts\\start-teacher-server.ps1",
              file=sys.stderr)
        return 2
    if args.model not in ids:
        print(f"[fatal] model {args.model!r} not served (have: {ids})", file=sys.stderr)
        return 2

    jobs: list[dict] = []
    corpus_meta: dict = {}
    if args.n_a:
        if not os.path.exists(args.shard):
            print(f"[fatal] shard missing: {args.shard}", file=sys.stderr)
            return 2
        for i, p in enumerate(load_source_a(args.shard, args.n_a, recipes, args.seed)):
            prompt = RECIPES[p["recipe"]].format(code=p["code"])
            jobs.append({"id": f"a-{i:06d}", "source": "starcoderdata",
                         "recipe": p["recipe"], "prompt": prompt,
                         "provenance": {"repo": p["repo"], "path": p["path"],
                                        "stars": p["stars"]}})
    if args.n_b:
        for i, p in enumerate(load_source_b(args.n_b, args.seed)):
            jobs.append({"id": f"b-{i:06d}", "source": "teacher-gen",
                         "recipe": p["recipe"], "prompt": p["prompt"],
                         "provenance": None})
    if os.path.exists(args.shard):
        corpus_meta = {"repo": "bigcode/starcoderdata", "gated": "auto (ToU accepted)",
                       "shard": os.path.relpath(args.shard, ROOT),
                       "sha256": sha256_file(args.shard)}

    print(f"[plan] {len(jobs)} samples -> {out_path}")
    print(f"[plan] model={args.model} think={args.think} temp={args.temperature} "
          f"max_tokens={args.max_tokens}")
    return _run(args, jobs, out_path, manifest_path, corpus_meta)


def _run(args, jobs: list[dict], out_path: str, manifest_path: str,
         corpus_meta: dict | None = None) -> int:
    t0 = time.time()
    fails = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for i, job in enumerate(jobs, 1):
            try:
                completion, usage = chat(
                    job["prompt"], endpoint=args.endpoint, model=args.model,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    think=args.think)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"[{i}/{len(jobs)}] FAIL {job['id']}: {e}")
                continue
            rec = {
                "id": job["id"], "source": job["source"], "recipe": job["recipe"],
                "system": SYSTEM, "prompt": job["prompt"],
                "completion": completion, "provenance": job["provenance"],
                "meta": {
                    "model": args.model, "endpoint": args.endpoint,
                    "temperature": args.temperature, "max_tokens": args.max_tokens,
                    "think": args.think,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            rate = i / max(time.time() - t0, 1e-9)
            print(f"[{i}/{len(jobs)}] ok {job['id']} "
                  f"({job['source']}/{job['recipe']}) "
                  f"tok={usage.get('completion_tokens')} {rate:.2f} samples/s")

    n_ok, lens = validate_jsonl(out_path)
    manifest = {
        "tag": args.tag,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {k: getattr(args, k) for k in
                   ("n_a", "n_b", "recipes", "source", "subset", "max_rows",
                    "scan_limit", "endpoint", "model", "max_tokens", "temperature",
                    "think", "seed")},
        "system_prompt": SYSTEM,
        "counts": {"requested": len(jobs), "written": n_ok, "failed": fails},
        "completion_chars": {"min": min(lens), "median": sorted(lens)[len(lens) // 2],
                             "max": max(lens)} if lens else None,
        "elapsed_sec": round(time.time() - t0, 1),
        "corpus": corpus_meta if corpus_meta is not None else {
            "repo": "bigcode/starcoderdata", "gated": "auto (ToU accepted)",
            "shard": os.path.relpath(args.shard, ROOT) if os.path.exists(args.shard) else None,
            "sha256": sha256_file(args.shard) if os.path.exists(args.shard) else None},
        "output": {"jsonl": os.path.basename(out_path),
                   "sha256": sha256_file(out_path)},
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"[done] wrote {n_ok}/{len(jobs)} (failed={fails}) "
          f"in {manifest['elapsed_sec']}s")
    print(f"[done] jsonl    -> {out_path}")
    print(f"[done] manifest -> {manifest_path}")
    if fails and n_ok and fails / max(len(jobs), 1) > 0.2:
        print("[warn] >20% failures — investigate before scaling up", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
