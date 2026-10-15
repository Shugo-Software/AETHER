# Phase 6 — dataset generation

Teacher: `qwen3.5-9b-teacher` (Qwen3.5-9B Q4_K_M on CUDA0, loopback `:8080`).
Student target: LFM2.5-2.6B QLoRA. Two sources, per requirement:

- **A. starcoderdata transforms** (`bigcode/starcoderdata`, ToU-gated,
  PASSED via our HF account): python shard
  `corpus/starcoderdata/python/train-00000-of-00059.parquet` (~758 MB,
  3.1 m rows) → rotating recipes `explain_code` / `qa_pairs` / `code_review`.
- **B. pure teacher generation** from 10 built-in seed prompts (recipe `seed_gen`).

## Harness

`scripts/generate_dataset.py` (stdlib-only client): deterministic seeded row
order, append/resume via `--tag`, `enable_thinking=false` (~30x cheaper),
JSONL + `manifest.json` (params, counts, sha256, teacher identity).
Endpoint must be up: `powershell -File scripts/start-teacher-server.ps1`.
Always launch pilot/background runs with `python -u` (unbuffered) so the
redirected log streams live. Throughput measured: **≈ 0.14 samples/s**
(1024-token cap) — plan ≈ 8.5 min per 75 samples.

## Pilot run (2026-10-14, tag `pilot1`) — 75/75, 0 failed, 541.5 s (~9 min)

`generate_dataset.py --n-a 50 --n-b 25` → `datasets/phase6_pilot1.jsonl`
(336.8 KB) + manifest. Recipe split: 17 explain_code / 17 qa_pairs /
16 code_review / 25 seed_gen. Completion chars: min 480 / median 1337 /
max 4729. Spot-checks (a-000000 explain on a TCP-server file, b-000000
threading-vs-asyncio) read concrete and correct. Rate ≈ 0.14 samples/s.

## ~~Prose recipes (EleutherAI/pile)~~ — removed 2026-10-15

The Pile track (`--source pile`) was removed to keep the tree small: datasets,
HF cache stubs, progress tracking, and the `load_source_pile` /
`PROSE_RECIPES` code are gone from `generate_dataset.py`. Jadis was trained on
that data (adapter `adapters/lfm25-jadis` survives); the source JSONLs were
deleted. Only the two StarCoder-backed sources (A/B below the title) remain.

## Quality gates (before any full-scale run)

1. **Spot-read ≥ 5 records per recipe** — concrete, grounded in the code,
   no hallucinated APIs.
2. **Truncation audit** — fraction of completions hitting `max_tokens`
   (pilot: several at exactly 1024 tokens → consider `--max-tokens 1536`
   for review-heavy batches).
3. **De-dupe check** — same `(repo, path)` shouldn't appear twice in one
   batch (seeded order makes this auditable).
4. Record any recipe wording changes in the manifest's `system_prompt`
   (edit + re-run from a new tag — never silently mutate).
