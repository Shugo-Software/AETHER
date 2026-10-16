#!/usr/bin/env python3
"""Phase 7 part 1/3: config + dataset prep for LFM2.5-2.6B QLoRA."""
from __future__ import annotations
# TAG-P7A
"""Phase 7 — LFM2.5-2.6B QLoRA on the 1060(s). Part A: config + data prep.

Pascal rules (verified constraints): fp16 only (no bf16), no flash-attn,
torch 2.14.0+cu126 pinned, single-6GB-card default with optional
two-card ZeRO via accelerate config. Teacher (CUDA0) untouched.
"""
import argparse
import glob
import json
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aether_env  # noqa: E402 - caches/temp stay on the repo drive (never C:)
aether_env.pin_caches()

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

import sysROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT = "LiquidAI/LFM2.5-2.6B"


def parse_args():
    ap = argparse.ArgumentParser(description="Phase 7 QLoRA trainer")
    ap.add_argument("--dataset", default=None,
                    help="phase6 jsonl (default: newest in datasets/)")
    ap.add_argument("--out", default=None,
                    help="adapter out dir (default: adapters/lfm25-<tag>)")
    ap.add_argument("--tag", default="p7a")
    ap.add_argument("--resume", default=None,
                    help="existing adapter dir to continue from (e.g. "
                         "adapters/lfm25-jadis); loads its weights instead of "
                         "attaching a fresh LoRA")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--prompt-loss", choices=["masked", "full"],
                    default="masked",
                    help="masked (default): labels are -100 over the prompt, so "
                         "only answer tokens carry gradient = instruction "
                         "tuning. full: legacy, the loss also covers the prompt "
                         "(continued-pretraining style, ~3x diluted signal)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1,
                    help="per-device batch (6GB VRAM: keep 1)")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--single-card", action="store_true", default=True,
                    help="pin to the 1060 6GB (default on)")
    ap.add_argument("--two-card", action="store_true",
                    help="joint 1060s via accelerate ZeRO (experimental)")
    return ap.parse_args()


def pick_dataset(explicit):
    if explicit:
        return explicit
    cands = sorted(glob.glob(os.path.join(ROOT, "datasets", "*.jsonl")),
                   key=os.path.getmtime)
    if not cands:
        raise SystemExit("no datasets/*.jsonl found — run Phase 6 first")
    return cands[-1]


def format_head(rec, tok):
    """The exact generation-prompt prefix of a rendered record.

    Single source of truth for the prompt/answer boundary: `format_record`
    builds the row on top of it, and completion-only label masking uses its
    token length to decide which tokens may contribute to the loss. Returns
    None when the chat template is unusable (caller falls back).
    """
    sys = rec.get("system") or "You are a helpful Python engineer."
    tmpl = getattr(tok, "chat_template", None)
    if tmpl:
        try:
            return tok.apply_chat_template(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": rec["prompt"]}],
                tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001 — caller uses the plain fallback
            return None
    return None


def format_record(rec, tok):
    """Render a record as the exact token stream the model meets at inference.

    The LFM2.5 template always prefills `<|im_start|>assistant\\n<think>` when
    add_generation_prompt=True, and it treats `</think>` in an assistant turn as
    the reasoning/answer split (replaying history, it trims everything up to
    `</think>`). So we build the head with the generation prompt — the real
    `<think>` tokens, not an approximation — and append the continuation the
    model must learn: `</think>\\n{completion}`, i.e. close the thinking block
    immediately and answer. Records that carry their own `reasoning` fill the
    block instead (`{reasoning}</think>\\n{completion}`), so a future
    reasoning-distilled dataset needs no code change.
    """
    sys = rec.get("system") or "You are a helpful Python engineer."
    prompt = rec["prompt"]
    comp = rec["completion"]
    reason = (rec.get("reasoning") or "").strip()
    head = format_head(rec, tok)
    if head is not None:
        return head + (reason + "</think>\n" if reason else "</think>\n") \
            + comp + "<|im_end|>\n"
    return f"{sys}\n\nUser: {prompt}\n\nAssistant: {comp}"


def load_rows(jsonl_path):
    rows = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mask_labels(ids, head_len):
    """Labels for one row: -100 over the prompt, real ids over the answer."""
    lab = list(ids)
    for i in range(min(head_len, len(lab))):
        lab[i] = -100
    return lab


def build_hf_dataset(jsonl_path, tok, max_len, mask_prompt=True):
    """Render + tokenize into {input_ids, attention_mask[, labels]}.

    With mask_prompt the prompt's labels are -100, so only answer tokens carry
    gradient — this is what makes the run instruction tuning rather than
    continued pretraining (unmasked, ~2/3 of the gradient goes to reproducing
    the input, and a row whose answer was truncated away trains nothing but the
    prompt). Such rows are dropped here: a row with no unmasked label has no
    valid loss and would produce a NaN step. Returns (dataset, stats).
    """
    rows = load_rows(jsonl_path)
    texts, heads = [], []
    for r in rows:
        texts.append(format_record(r, tok))
        h = format_head(r, tok)
        if h is None:
            sys = r.get("system") or "You are a helpful Python engineer."
            h = f"{sys}\n\nUser: {r['prompt']}\n\nAssistant: "
        heads.append(h)
    head_lens = [len(tok(h, add_special_tokens=False)["input_ids"])
                 for h in heads]
    ds = Dataset.from_dict({"text": texts, "head_len": head_lens})

    def tok_fn(batch):
        enc = tok(batch["text"], truncation=True, max_length=max_len)
        if mask_prompt:
            enc["labels"] = [_mask_labels(ids, hl) for ids, hl in
                             zip(enc["input_ids"], batch["head_len"])]
        return enc

    tokenized = ds.map(tok_fn, batched=True,
                       remove_columns=["text", "head_len"])
    stats = {"rows_in": len(rows), "mask_prompt": bool(mask_prompt),
             "dropped_no_answer": 0, "rows_used": len(tokenized)}
    if mask_prompt:
        keep = [i for i, lab in enumerate(tokenized["labels"])
                if any(x != -100 for x in lab)]
        stats["dropped_no_answer"] = len(tokenized) - len(keep)
        tokenized = tokenized.select(keep)
        stats["rows_used"] = len(tokenized)
        total = sum(len(x) for x in tokenized["input_ids"])
        answer = sum(sum(1 for t in lab if t != -100)
                     for lab in tokenized["labels"])
        stats["tokens_total"] = total
        stats["tokens_answer"] = answer
        stats["answer_share"] = round(1.0 * answer / total, 4) if total else None
    return tokenized, stats


class CompletionCollator:
    """Pad input_ids/attention_mask, and pad labels with -100 (not pad_token_id).

    Explicit on purpose: a generic collator padding labels with the pad token
    would train the model to emit padding, and precomputed labels are the whole
    point of completion-only masking.
    """

    def __init__(self, tok):
        self.pad_id = tok.pad_token_id

    def __call__(self, features):
        width = max(len(f["input_ids"]) for f in features)
        ids, mask, labs = [], [], []
        for f in features:
            n = width - len(f["input_ids"])
            ids.append(list(f["input_ids"]) + [self.pad_id] * n)
            am = f.get("attention_mask") or [1] * len(f["input_ids"])
            mask.append(list(am) + [0] * n)
            if "labels" in f:
                labs.append(list(f["labels"]) + [-100] * n)
        out = {"input_ids": torch.tensor(ids, dtype=torch.long),
               "attention_mask": torch.tensor(mask, dtype=torch.long)}
        if labs:
            out["labels"] = torch.tensor(labs, dtype=torch.long)
        return out


def resolve_device(args):
    if args.two_card:
        return None  # accelerate launcher owns placement
    # run-train.cmd masks to the 1060 6GB via CUDA_VISIBLE_DEVICES=1,
    # so torch sees exactly one GPU at idx 0 (principle #4: never hardcode).
    n = torch.cuda.device_count()
    if n == 1:
        return 0
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        if "1060" in name and "3GB" not in name:
            return i
    raise SystemExit("1060 6GB not visible to torch")

#!/usr/bin/env python3
"""Phase 7 part 2/3: model load + QLoRA attach on the pinned 1060 (fp16)."""
# TAG-P7B
"""Part B: fp16 4-bit model load + LoRA attach (Pascal-safe, no bf16)."""
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig



def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_qlora_model(lora_r, lora_alpha, single_dev, resume=None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Pascal: fp16, never bf16
        bnb_4bit_use_double_quant=True,
    )
    kw = {"quantization_config": bnb,
          "torch_dtype": torch.float16,
          "trust_remote_code": True}  # Pascal: no flash-attn (default off)
    if single_dev is not None:
        kw["device_map"] = {"": single_dev}
    else:
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(STUDENT, **kw)
    model = prepare_model_for_kbit_training(model)
    if resume:
        # Continue an existing adapter (e.g. jadis -> more data) instead of
        # stacking a fresh LoRA on top. is_trainable=True unfreezes its weights.
        from peft import PeftModel
        print("[train] resuming adapter from %s" % resume, flush=True)
        return PeftModel.from_pretrained(model, resume, is_trainable=True)
    # LFM2 arch: target the attention + MLP projections generically.
    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                     target_modules=["q_proj", "k_proj", "v_proj",
                                     "o_proj", "gate_proj", "up_proj",
                                     "down_proj"],
                     lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)



#!/usr/bin/env python3
"""Phase 7 part 3/3: SFTTrainer loop + dashboard state/log writes."""
# TAG-P7C
"""Part C: SFTTrainer loop + dashboard state/log writes."""
import json
import os
from datetime import datetime, timezone

from transformers import TrainerCallback


class MetricsCallback(TrainerCallback):
    """Append per-step telemetry to logs/train-<tag>.metrics.jsonl.

    Feeds the dashboard loss curve / model-health / step-time / ETA panels.
    Additive only — does not change training behaviour.
    """

    def __init__(self, metrics_path, phase=None, corpus=None, offset=0):
        self.metrics_path = metrics_path
        self.phase = phase
        self.corpus = corpus
        # Multi-corpus sweeps restart HF's step counter per phase. `offset` keeps
        # the logged step monotonic across the whole sweep, so the dashboard's loss
        # chart reads as one run instead of a sawtooth.
        self.offset = offset

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        import math
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "step": self.offset + state.global_step,
               "epoch": round(state.epoch, 4) if state.epoch is not None else None,
               "total": ((self.offset + state.max_steps)
                         if state.max_steps not in (None, -1) else None)}
        if self.phase is not None:
            rec["phase"] = self.phase
        if self.corpus:
            rec["corpus"] = self.corpus
        for k in ("loss", "grad_norm", "learning_rate"):
            v = logs.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            # NaN/inf is invalid JSON and breaks the dashboard parser -> null.
            rec[k] = round(fv, 6) if math.isfinite(fv) else None
        if rec.get("loss") is None and "loss" in logs:
            # A non-finite loss silently ruins a run (e.g. a row with no
            # unmasked label). Say so loudly instead of writing null and moving on.
            import sys as _sys
            print("[train] WARNING non-finite loss at step %s" % rec["step"],
                  file=_sys.stderr, flush=True)
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()


def build_trainer(args, tok, model, ds, dstats, out_dir, metrics_path, *,
                  max_steps=None, epochs=None, phase=None, corpus=None, offset=0):
    """TrainingArguments + Trainer for one corpus, plus the run's diagnostics.

    One corpus gets one Trainer, so it also gets a fresh optimizer and a fresh
    cosine schedule: momentum from one dataset's distribution never leaks into the
    next phase of a sweep.
    """
    from transformers import (DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=(args.epochs if epochs is None else epochs),
        max_steps=(args.max_steps if max_steps is None else max_steps),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        fp16=True,  # Pascal: fp16, never bf16
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )
    # Plain Trainer (not trl.SFTTrainer): trl 1.15's fused-LM-head patched
    # forward needs a triton-compiled chunked kernel that is None on Pascal.
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      processing_class=tok,
                      data_collator=(DataCollatorForLanguageModeling(tok, mlm=False)
                                     if args.prompt_loss == "full"
                                     else CompletionCollator(tok)),
                      callbacks=[MetricsCallback(metrics_path, phase=phase,
                                                 corpus=corpus, offset=offset)])
    # Diagnostic: HF derives the total step count from
    # len(train_dataloader) // grad_accum, and both the progress bar and the
    # dashboard's ETA read that total. Print what HF will actually do, so a
    # dataset/steps mismatch is visible in the log instead of implied.
    import math as _math
    _epochs = args.epochs if epochs is None else epochs
    _max_steps = args.max_steps if max_steps is None else max_steps
    try:
        _dl = trainer.get_train_dataloader()
        _spe = max(len(_dl) // max(args.grad_accum, 1), 1)
        print("[train] rows=%d dataset_rows=%d batch=%d grad_accum=%d "
              "dataloader_batches=%d steps_per_epoch=%d epochs=%s "
              "=> total_steps=%d (max_steps=%s)"
              % (dstats["rows_in"], len(ds), args.batch, args.grad_accum, len(_dl),
                 _spe, _epochs, _spe * int(_math.ceil(_epochs)), _max_steps),
              flush=True)
    except Exception as e:  # noqa: BLE001 — never let diagnostics block training
        print("[train] dataloader diagnostic failed: %s" % e, flush=True)
    print("[train] labels: prompt_loss=%s rows_used=%d/%d"
          " dropped_no_answer=%d answer_token_share=%s"
          % (args.prompt_loss, dstats["rows_used"], dstats["rows_in"],
             dstats["dropped_no_answer"], dstats.get("answer_share")), flush=True)
    if args.prompt_loss == "masked":
        print("[train] only answer tokens carry gradient (%d of %s kept tokens)"
              % (dstats.get("tokens_answer", 0), dstats.get("tokens_total")),
              flush=True)
    return trainer


def train_corpus(args, tok, model, ds_path, out_dir, state, *, phase=None,
                 max_steps=None, epochs=None, metrics_path=None,
                 metrics_fresh=False, offset=0, phase_dir=None, resume_dir=None):
    """One corpus: dataset -> trainer -> train -> save (+ a per-phase checkpoint).

    Both entry points go through here: `main()` once (a single corpus, exactly as
    before) and train_sweep.py per corpus — one base-model load, many corpora.
    Returns a summary dict for the caller to record.
    """
    state(corpus=ds_path, status="preparing_data", phase=phase)
    ds, dstats = build_hf_dataset(ds_path, tok, args.max_len,
                                  mask_prompt=(args.prompt_loss == "masked"))
    n_rows = dstats["rows_in"]
    if metrics_fresh and metrics_path:
        open(metrics_path, "w").close()   # fresh per run; appended across phases
    trainer = build_trainer(args, tok, model, ds, dstats, out_dir, metrics_path,
                            max_steps=max_steps, epochs=epochs, phase=phase,
                            corpus=os.path.basename(ds_path), offset=offset)
    _epochs = args.epochs if epochs is None else epochs
    _max_steps = args.max_steps if max_steps is None else max_steps
    state(corpus=ds_path, status="training", phase=phase, rows=n_rows,
          rows_used=dstats["rows_used"], epochs=_epochs, lr=args.lr,
          resumed_from=(os.path.relpath(resume_dir, ROOT) if resume_dir else None),
          config={"dataset": os.path.basename(ds_path), "student": STUDENT,
                  "epochs": _epochs, "lr": args.lr, "max_len": args.max_len,
                  "prompt_loss": args.prompt_loss,
                  "rows_used": dstats["rows_used"],
                  "dropped_no_answer": dstats["dropped_no_answer"],
                  "answer_token_share": dstats.get("answer_share"),
                  "max_steps": _max_steps, "batch": args.batch,
                  "grad_accum": args.grad_accum, "lora_r": args.lora_r,
                  "lora_alpha": args.lora_alpha, "optim": "paged_adamw_8bit",
                  "precision": "fp16", "phase": phase,
                  "metrics_file": os.path.basename(metrics_path or ""),
                  "resume": (os.path.relpath(resume_dir, ROOT)
                             if resume_dir else None)})
    result = trainer.train()
    state(corpus=ds_path, status="saving", phase=phase,
          train_loss=round(float(result.training_loss), 4))
    trainer.save_model(out_dir)
    if phase_dir:
        # The checkpoint this phase is graded on: a full PEFT adapter, so the
        # teacher's verdict is about the weights that existed at that moment.
        trainer.save_model(phase_dir)
    loss = round(float(result.training_loss), 4)
    print("[train] corpus done loss=%.4f steps=%d rows=%d -> %s"
          % (loss, int(result.global_step), n_rows, out_dir), flush=True)
    return {"train_loss": loss, "steps": int(result.global_step), "rows": n_rows,
            "rows_used": dstats["rows_used"], "corpus": ds_path, "phase": phase,
            "out": out_dir, "phase_dir": phase_dir}


def main():
    args = parse_args()
    ds_path = pick_dataset(args.dataset)
    tag = args.tag
    out_dir = args.out or os.path.join(ROOT, "adapters", "lfm25-" + tag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    state_path = os.path.join(ROOT, "logs", "train-%s.json" % tag)

    def state(corpus=None, **kw):
        # Merge with the existing file so once-written fields (e.g. config)
        # survive later status updates (saving/done) instead of being dropped.
        # `corpus` names the corpus being worked on *now* (a sweep changes it per
        # phase; a single-corpus run passes nothing and behaves as before).
        prev = {}
        try:
            with open(state_path, encoding="utf-8") as fh:
                prev = json.load(fh)
        except (OSError, ValueError):
            prev = {}
        prev.update(kw)
        prev.setdefault("tag", tag)
        prev["dataset"] = os.path.basename(corpus or ds_path)
        prev["out"] = os.path.relpath(out_dir, ROOT)
        # Every state names the process that wrote it, so a killed trainer is
        # visible as such instead of leaving status=training behind forever
        # (the dashboard reconciles the two; seen 2026-10-15).
        prev["pid"] = os.getpid()
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(prev, fh, indent=2)

    state(status="loading_tokenizer")
    tok = load_tokenizer()
    dev = resolve_device(args)
    state(status="loading_model", device=dev)
    resume_dir = None
    if args.resume:
        resume_dir = os.path.abspath(args.resume)
        if not os.path.isfile(os.path.join(resume_dir, "adapter_config.json")):
            raise SystemExit("--resume %r has no adapter_config.json" % args.resume)
    model = load_qlora_model(args.lora_r, args.lora_alpha, dev, resume=resume_dir)
    model.print_trainable_parameters()

    # The single-corpus path is exactly one sweep phase, so it runs through the same
    # function train_sweep.py loops over: one implementation of "train a corpus",
    # not two. Behaviour here is unchanged.
    summary = train_corpus(
        args, tok, model, ds_path, out_dir, state,
        metrics_path=os.path.join(ROOT, "logs", "train-%s.metrics.jsonl" % tag),
        metrics_fresh=True, resume_dir=resume_dir)
    state(status="done", train_loss=summary["train_loss"])
    print("[train] done loss=%.4f -> %s" % (summary["train_loss"], out_dir),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



