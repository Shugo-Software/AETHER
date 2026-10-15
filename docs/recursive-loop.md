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
python scripts\recursive_train.py --iteration 1 --tag pro1 --corpus datasets\phase6_pro1.jsonl --epochs 3 --sample 40 --min-pass 0.6
```

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
