# Phase 7 — Student Training (LFM2.5-2.6B QLoRA on the 1060s)

> Status: pilot `pilot1` running 2026-10-14. Stack: torch **2.14.0+cu126** (pinned —
> last prebuilt Pascal wheels), transformers 5.19 + peft 0.21 + bitsandbytes 0.50 +
> accelerate 1.15 + trl 1.15, all in `venvs/ml/` (G:).

## Harness

`scripts/train_student.py` + launcher `scripts/run-train.cmd`:

- **Single-card default:** `CUDA_VISIBLE_DEVICES=1` masks everything but the 1060 6GB
  (torch then sees idx 0; `resolve_device` handles both masked and unmasked cases).
- **Pascal rules encoded:** fp16 compute (`bnb_4bit_compute_dtype=fp16`), `fp16=True /
  bf16=False`, `optim="paged_adamw_8bit"`, no flash-attn, LoRA r16/α32 on
  q/k/v/o + gate/up/down (≈1.18M trainable / 2.7B total, 0.044%).
- **Plain `transformers.Trainer`** with `DataCollatorForLanguageModeling(mlm=False)`,
  **not** `trl.SFTTrainer` (see Gotchas).
- **Dashboard hook:** writes `logs/train-<tag>.json` at each stage
  (`loading_tokenizer → loading_model → preparing_data → training → done` with loss).
- Launch: `scripts\run-train.cmd --tag <tag> [--epochs N] [--max-steps K] [--max-len L]`
  (defaults: dataset = newest `datasets/*.jsonl`, out = `adapters/lfm25-<tag>`).

## Why `run-train.cmd` exists (Windows quoting trap, 2026-10-14)

`Start-Process python.exe -u "G:\Program Prototype\...\train_student.py" ...` silently
executes **`G:\Program`** (the path split at the space) as a script — producing the
baffling error `File "G:\Program", line 2 / NVIDIA GeForce RTX 4070 Ti,... / SyntaxError`.
The `.cmd` wrapper with pre-quoted paths is the fix; **always launch via the wrapper**,
never via raw `Start-Process` with a spaced path + multiple args.

## Gotchas found during smoke tests (7 iterations)

1. `use_flash_attention_2` kwarg → rejected by transformers 5.19 `Lfm2ForCausalLM`
   (removed; default is already off).
2. `tok.chat_model` probe → `AttributeError` on transformers 5.x `TokenizersBackend`
   (now checks `getattr(tok, "chat_template", None)`).
3. `TrainingArguments.warmup_ratio` → renamed; use `warmup_steps=10`.
4. `SFTTrainer` with 3 visible GPUs → demands `CUDA_VISIBLE_DEVICES` masking
   (mask added to the launcher).
5. **TRL 1.15 fused-LM-head is broken on Pascal:** `add_fused_lm_head` patches
   `model.forward` to call `_ChunkedLogProbFunction.apply`, which is `None`
   (triton kernel never compiled) → `AttributeError: 'NoneType' has no attribute
   'apply'`. Fix = plain `Trainer`. (TRL stays installed for Phase 8 DPO, where its
   DPO path may work — verify then.)
6. `causal_conv1d` missing → transformers falls back to a slow reference impl
   (warning only; LFM2 arch uses conv layers, so expect slower-than-ideal steps).

## Train/inference mismatch on the `<think>` block — fixed 2026-10-15

**Root cause is upstream design, not a broken base model.** The LFM2.5 chat
template (`chat_template.jinja`) makes reasoning first-class:

- `add_generation_prompt=True` **always** ends the prompt with
  `<|im_start|>assistant\n<think>` — the template has no `enable_thinking` switch.
- An assistant turn may carry `thinking` / `reasoning` / `reasoning_content`,
  rendered as `<think>{reasoning}</think>{content}`.
- Replaying history it keeps only what follows `</think>`
  (`content.split("</think>")[-1]`), so the canonical finished-turn shape is
  `<think>…</think>answer` — the **close** tag is the reasoning/answer separator.

The old `format_record` rendered `[system, user, assistant]` *without*
`add_generation_prompt`, so every training turn was `…assistant\n{completion}`:
no `<think>` token anywhere. Inference then prefilled `<think>` — a token the
student had never seen — and the model stayed inside a block it could not close.

**Measured consequence** (`logs/eval-lfm25-jadis-starcoder.md`; 6 probes × 2
prompt conventions, greedy): with the template prefill the old adapter emitted
`</think>` on **1 of 6** probes (base: 2 of 6) and answered inside the block;
with a raw prompt (no prefill — its actual training shape) it answered normally.
The weights were fine; the prompt shape was wrong.

**Fix (implemented)** — build the head with the generation prompt so training
contains the real tokens, then teach the continuation inference needs:

```python
head = tok.apply_chat_template([system, user], tokenize=False,
                               add_generation_prompt=True)   # …assistant\n<think>
return head + (reason + "</think>\n" if reason else "</think>\n") + comp + "<|im_end|>\n"
```

A record with no reasoning therefore trains as
`<|im_start|>assistant\n<think></think>\n{completion}<|im_end|>` (close the block,
then answer), and a record carrying a `reasoning` field fills the block instead
(`<think>{reasoning}</think>…`) — the upstream-faithful shape — with no further
code change. Check any adapter with `scripts/eval_jadis.py --convention template`
(a correct build shows `close=y lead=y` on the tuned side); check any artifact
with `scripts/verify_export.py`.

### Outcome — the fix that mattered was the prompt, not the training signal

Jadis 1.0 (75 rows, 10 epochs, 50 steps, loss 1.6664) was trained with the
corrected `format_record`, and **the two-token training signal did not flip the
behaviour**: with the stock `<think>` prefill the tuned adapter still opened with
analysis prose (`logs/eval-jadis-1.0.md` — `close=33%`, identical to base, and the
first tokens are "The user wants me to explain…"). Two closing tokens among ~500
per record are simply too weak a signal at this scale.

What *did* work is shaping the prompt, measured on the same adapter
(`logs/eval-jadis-1.0-nothink.md`, block pre-closed as `…assistant` + `</think>`
+ newline):

| Probe | Base | Jadis 1.0 (same prompt, no thinking) |
|---|---|---|
| "Explain what a context manager is… in two sentences" | 109 words, lecture-style | **49 words, direct two sentences** |
| "Write a haiku about a GPU waiting for data" | haiku **plus a meta-note about its own syllable counting** | clean 3-line haiku, **no meta** |

So the export now ships a `chat_template.jinja` whose generation prompt
pre-closes the block (documented in the card), and the training-side change stays
because it is the correct target shape for any future dataset with real
reasoning. Generalisation: for a 2.6B QLoRA on a few dozen rows, prompt/template
shape dominates; do not expect a single structural token to be learned.

## The loss was never masked — fixed 2026-10-15 (changed every number before it)

`DataCollatorForLanguageModeling(tok, mlm=False)` masks **padding only**. So every
run through Jadis 1.0 and loop iteration 1 computed loss over the *whole* rendered
row: system prompt, user turn, code, and answer. Measured on `phase6_pro1.jsonl`
with `scripts/verify_dataset.py`, answers are only **35.8%** of the kept tokens —
i.e. ~64% of the gradient was spent learning to reproduce the input. That is
continued pretraining wearing an SFT costume.

It is worse than dilution. 67 of 500 rows had prompts longer than `--max-len`, so
their answer was truncated away entirely: **100% of their gradient was prompt**,
which trains the model to *continue StarCoder files* mid-document. The gradient
was not merely diluted, part of it pointed the wrong way.

What changed (`scripts/train_student.py`):

- `--prompt-loss {masked,full}`, **default `masked`**: labels are set to `-100`
  across the prompt head (`format_head`, now the single source of truth for the
  prompt/answer boundary), so only answer tokens carry gradient.
- `CompletionCollator` pads `input_ids`/`attention_mask` and pads labels with
  `-100` explicitly. Letting a generic collator pad labels with `pad_token_id`
  would train the model to emit padding.
- Rows left with **no** unmasked token are dropped and counted
  (`dropped_no_answer`). A masked row with zero labels has no valid loss at all
  and `CrossEntropyLoss` means over an empty set → **NaN**, which would silently
  destroy the run. `MetricsCallback` now also warns on any non-finite loss.
- Two new diagnostic lines print what the run will actually train on:
  `[train] labels: prompt_loss=masked rows_used=330/330 dropped_no_answer=0
  answer_cut=0 answer_token_share=0.3581`.
- **The rule is enforced in the trainer, in both loss modes** (2026-10-16). It used
  to be a promise `fit_corpus.py` kept and the trainer merely trusted, so any corpus
  that skipped fitting could be trained anyway: unmasked, a row whose answer was cut
  away is 100% prompt gradient — it teaches the student to reproduce the input.
  Measured on `datasets/starcoderdata-code.jsonl`: `answer_cut=3 rows=[1897, 2176,
  2210]` at `max_len=1536` (`phase6_pro1.jsonl`: 67 rows). Those rows are now dropped
  in `masked` *and* `full`, counted as `answer_cut` (per-row indices included), warned
  about with the exact `fit_corpus.py` command that fixes them, and written into the
  run's state — an unfitted corpus can no longer make a run quietly train on
  prompts. The measurement itself is shared: `record_pieces()` renders the row, and
  the fitter, the verifier and the trainer count the same string with the same
  tokenizer call (special tokens included), because a budget measured one way and
  enforced another is how three rows lost their answers while the fitter said they
  fitted.

**Loss values are not comparable across this change** (different token sets) and
not comparable across corpora. Before: 2.41 at step 2 of 96 on 500 rows. After:
1.29 at step 2 of 64 on the fitted 330-row corpus. The gate compares *behaviour*
via the teacher judge, not loss, for exactly this reason.

Masking makes a precondition mandatory: the corpus must be fitted to the budget
(`scripts/fit_corpus.py`), because a row whose prompt exceeds `--max-len` now has
nothing left to learn from. See `docs/dataset-generation.md`.



5 steps, max-len 256, on the 1060 6GB: **loss 3.864**, ~20 s/step, adapter saved
(`adapters/lfm25-smoke8/adapter_model.safetensors`, 4.6 MB). First step showed
`grad_norm: nan` (fp16 warmup transient — cleared by step 2).

## Pilot (tag `pilot1`)

Full `phase6_pilot1.jsonl` (75 rows), 1 epoch, max-len 1024, on the 1060 6GB.
~72 s/step at full length. Result recorded in `logs/train-pilot1.json` on completion.
