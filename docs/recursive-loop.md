# Phase 11 — the recursive train → review → gate loop

> Purpose: make "the teacher trained the student" an **auditable claim** instead of
> a loss number. The teacher grades every student answer against the corpus
> reference, the gate is a recorded decision, and the failures become the next
> iteration's training data.

## Endpoints (all three GPUs busy at once)

| Role | Device | Port | Weights | Start |
|---|---|---|---|---|
| teacher | CUDA0 (4070 Ti) | 8080 | `Qwen3.5-9B-Q4_K_M` | `scripts\start-teacher-server.ps1` |
| student A | **CUDA2 (GTX 1060 3 GB)** | 8082 | `Jadis-<tag>-iq4_xs.gguf` (1.41 GiB) | `scripts\serve_student.ps1 -Device CUDA2 -Port 8082` |
| student B | **CUDA1 (GTX 1060 6 GB)** | 8083 | `Jadis-<tag>-f16.gguf` (5.03 GiB) | `scripts\serve_student.ps1 -Device CUDA1 -Port 8083` |

Measured on this rig: f16 peaks at **5463 MiB** on the 6 GB card at 4 k context
(27 t/s), iq4_xs peaks at **1657 MiB** on the 3 GB card at 4 k (44 t/s) — see
`benchmarks.md`.

## One iteration (`scripts/recursive_train.py`)

```
train (QLoRA, 6 GB card) → export (merge → f16 + iq4_xs) → serve (restart both)
   → review (teacher grades a corpus sample, per student)
   → gate (mean pass rate ≥ floor AND ≥ previous iteration)
   → fix (teacher writes the answers for the failed rows → next corpus)
```

Every step is a subprocess call to the same script a human would run, so any
iteration can be re-run by hand (`--skip-train --skip-export` …) and the state
(`logs/loop-<tag>.json`) records each step's exit status plus the gate decision.

```powershell
# 1. verify + fit the corpus first (see docs/dataset-generation.md)
python scripts\verify_dataset.py --jsonl datasets\phase6_pro1.jsonl --max-len 1536 --overlap datasets\phase6_pilot1.jsonl
python scripts\fit_corpus.py --in datasets\phase6_pro1.jsonl --out datasets\phase6_pro1-fit.jsonl --max-len 1536 --holdout datasets\phase6_pilot1.jsonl

# 2. one iteration
python scripts\recursive_train.py --iteration 2 --tag pro1 --corpus datasets\phase6_pro1-fit.jsonl --max-steps 64 --max-len 1536 --sample 40 --min-pass 0.6
```

## Preconditions the loop enforces (2026-10-15)

- **The corpus must be fitted to `--max-len`.** With completion-only label
  masking, a row whose prompt fills the budget has nothing left to learn from —
  and rather than training on junk, the trainer drops such rows and reports
  `dropped_no_answer`. `fit_corpus.py` prevents ever handing it one.
- **`--prompt-loss masked` is pinned** as an explicit argument (not left to the
  trainer's default) so a future default change cannot silently alter what the
  loop trains. Every run's setting is recorded in `logs/loop-<tag>.json`.
- **The 6 GB card is masked** for training: the loop reuses `train_control.TRAIN_ENV`
  (`CUDA_VISIBLE_DEVICES=1`), the same env `run-train.cmd` uses. Without it,
  accelerate shards the dataloader across all three GPUs (500 rows became 167
  batches = 3x fewer steps) and the model can land on the 4070 Ti where the
  teacher lives (OOM).
- **No stale artifacts can be shipped.** The loop deletes `adapters/<tag>` before
  training, verifies the adapter was actually rewritten after training started
  (`fresh_adapter`), deletes the previous iteration's GGUFs before exporting, and
  refuses to serve if a quant artifact is missing. Each of those failure modes
  was silent otherwise: the loop would have graded the *previous* model.

## The judge

`scripts/review_students.py` asks the student (temperature 0), then asks the
teacher to grade **against the corpus reference** with a strict JSON rubric:

```json
{"score": 1-5, "pass": true|false, "missing": ["…"], "wrong": ["…"], "critique": "≤25 words"}
```

Three things learned the hard way, all encoded:

1. **Disable thinking for the judge.** Qwen3.5 is a thinking model: with a 160-token
   budget the entire answer was reasoning and the verdict never arrived. The judge
   call sets `chat_template_kwargs: {"enable_thinking": false}`.
2. **Budget the JSON.** At 700 tokens the verdict was truncated mid-critique on
   ~50 % of rows (measured), which quietly dropped rows from the gate. The rubric
   now demands terseness, the budget is 1200, and `extract_verdict()` falls back to
   a regex recovery of `score`/`pass` so a truncated verdict is *marked*, never
   silently lost (`judge_repaired` in the JSONL).
3. **Grade against the reference, not against vibes.** The rubric deducts for
   preamble, meta-commentary, restating the task, invented APIs and padding — the
   same properties the corpus teaches — so a tighter answer outranks a hedged one.

## Gate semantics

`gate.passed` requires **both**:

- `clears_floor` — mean pass rate across students ≥ `--min-pass` (default 0.60), and
- `no_regression` — mean pass rate ≥ the previous iteration's.

A failed gate is not a dead end: the same run writes
`datasets/<tag>-fix<iteration>.jsonl`, containing teacher-written answers for the
failed rows (grounded in the corpus reference + the critique). Feeding that into
the next iteration is what makes the loop *recursive* rather than a one-shot
fine-tune.

## Honest limits (measure before believing)

- **The teacher is a 9B grader.** Its scores are noisy; treat pass-rate *deltas*
  between iterations as the signal, and spot-read `review-<tag>.md` before acting.
- **Two students, one corpus.** Student A (quantized) and B (f16) see identical
  prompts at the same context; their scores differ by quantization error, not by
  curriculum. Report both, gate on the mean.
- **A 2.6B student has a capacity ceiling.** The corpus is style/format/grounded
  Q&A; the loop can move tone, structure and directness, not raw capability. The
  eval-only `openai-math` set exists to show where that ceiling is.
