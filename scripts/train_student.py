#!/usr/bin/env python3
"""Phase 7 part 1/3: config + dataset prep for LFM2.5-2.6B QLoRA."""
from __future__ import annotations
# TAG-P7A
"""Phase 7 â€” LFM2.5-2.6B QLoRA on the 1060(s). Part A: config + data prep.

Pascal rules (verified constraints): fp16 only (no bf16), no flash-attn,
torch 2.14.0+cu126 pinned, single-6GB-card default with optional
two-card ZeRO via accelerate config. Teacher (CUDA0) untouched.
"""
import argparse
import glob
import json
import os

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        raise SystemExit("no datasets/*.jsonl found â€” run Phase 6 first")
    return cands[-1]


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
    tmpl = getattr(tok, "chat_template", None)
    if tmpl:
        try:
            head = tok.apply_chat_template(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
            return head + (reason + "</think>\n" if reason else "</think>\n") \
                + comp + "<|im_end|>\n"
        except Exception:  # noqa: BLE001 — fall through to plain format
            pass
    return f"{sys}\n\nUser: {prompt}\n\nAssistant: {comp}"


def build_hf_dataset(jsonl_path, tok, max_len):
    rows = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    texts = [format_record(r, tok) for r in rows]
    ds = Dataset.from_dict({"text": texts})

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=max_len)

    return ds.map(tok_fn, batched=True, remove_columns=["text"]), len(rows)


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

    def __init__(self, metrics_path):
        self.metrics_path = metrics_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        import math
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "step": state.global_step,
               "epoch": round(state.epoch, 4) if state.epoch is not None else None,
               "total": state.max_steps if state.max_steps not in (None, -1) else None}
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
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()


def main():
    args = parse_args()
    ds_path = pick_dataset(args.dataset)
    tag = args.tag
    out_dir = args.out or os.path.join(ROOT, "adapters", "lfm25-" + tag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    state_path = os.path.join(ROOT, "logs", "train-%s.json" % tag)

    def state(**kw):
        # Merge with the existing file so once-written fields (e.g. config)
        # survive later status updates (saving/done) instead of being dropped.
        prev = {}
        try:
            with open(state_path, encoding="utf-8") as fh:
                prev = json.load(fh)
        except (OSError, ValueError):
            prev = {}
        prev.update(kw)
        prev.setdefault("tag", tag)
        prev["dataset"] = os.path.basename(ds_path)
        prev["out"] = os.path.relpath(out_dir, ROOT)
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

    state(status="preparing_data")
    ds, n_rows = build_hf_dataset(ds_path, tok, args.max_len)

    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    targs = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
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
    metrics_path = os.path.join(ROOT, "logs", "train-%s.metrics.jsonl" % tag)
    open(metrics_path, "w").close()  # fresh per run
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      processing_class=tok,
                      data_collator=DataCollatorForLanguageModeling(
                          tok, mlm=False),
                      callbacks=[MetricsCallback(metrics_path)])
    state(status="training", rows=n_rows, epochs=args.epochs, lr=args.lr,
          resumed_from=(os.path.relpath(resume_dir, ROOT) if resume_dir else None),
          config={"dataset": os.path.basename(ds_path), "student": STUDENT,
                  "epochs": args.epochs, "lr": args.lr, "max_len": args.max_len,
                  "max_steps": args.max_steps, "batch": args.batch,
                  "grad_accum": args.grad_accum, "lora_r": args.lora_r,
                  "lora_alpha": args.lora_alpha, "optim": "paged_adamw_8bit",
                  "precision": "fp16", "metrics_file": os.path.basename(metrics_path),
                  "resume": (os.path.relpath(resume_dir, ROOT)
                             if resume_dir else None)})
    # Diagnostic: HF derives the total step count from
    # len(train_dataloader) // grad_accum, and both the progress bar and the
    # dashboard's ETA read that total. Print what HF will actually do, so a
    # dataset/steps mismatch is visible in the log instead of implied.
    import math as _math
    try:
        _dl = trainer.get_train_dataloader()
        _spe = max(len(_dl) // max(args.grad_accum, 1), 1)
        print("[train] rows=%d dataset_rows=%d batch=%d grad_accum=%d "
              "dataloader_batches=%d steps_per_epoch=%d epochs=%s "
              "=> total_steps=%d"
              % (n_rows, len(ds), args.batch, args.grad_accum, len(_dl), _spe,
                 args.epochs, _spe * int(_math.ceil(args.epochs))), flush=True)
    except Exception as e:  # noqa: BLE001 — never let diagnostics block training
        print("[train] dataloader diagnostic failed: %s" % e, flush=True)
    result = trainer.train()
    state(status="saving", train_loss=round(float(result.training_loss), 4))
    trainer.save_model(out_dir)
    state(status="done", train_loss=round(float(result.training_loss), 4))
    print("[train] done loss=%.4f -> %s"
          % (result.training_loss, out_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



