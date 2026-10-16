#!/usr/bin/env python3
"""Phase 7 — generate sample outputs from a trained student adapter.

Loads the LFM2.5-2.6B base + a QLoRA adapter, runs a small fixed prompt set,
and writes logs/train-<tag>.samples.json for the dashboard "Live generations"
panel. Mirrors train_student.py's Pascal-safe load (fp16 4-bit, no flash-attn).

Usage:
  python scripts/sample_student.py --adapter adapters/lfm25-pilot1 --tag pilot1
  python scripts/sample_student.py --adapter adapters/lfm25-pilot1 --tag pilot1 \
      --prompts-file myprompts.json

A prompt entry is either a bare string or {"recipe": "...", "prompt": "..."}.
Falls back to a built-in Python-engineering probe set when no file is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aether_env  # noqa: E402 - caches/temp stay on the repo drive (never C:)
aether_env.pin_caches()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT = "LiquidAI/LFM2.5-2.6B"

DEFAULT_PROMPTS = [
    {"recipe": "explain_code",
     "prompt": "Explain this Python code for a mid-level developer: what does it do, "
               "and what is one likely pitfall when reusing it?\n\n<code>\n"
               "def chunked(it, n):\n    it = iter(it)\n    while True:\n"
               "        c = list(itertools.islice(it, n))\n"
               "        if not c:\n            return\n        yield c\n</code>"},
    {"recipe": "qa",
     "prompt": "What is the difference between Python's threading and asyncio, "
               "and when is each the right choice?"},
    {"recipe": "explain",
     "prompt": "Why is paging-through-a-large-file line-by-line usually better "
               "than reading it all into memory? Give one concrete consequence."},
]

SYSTEM = "You are a precise senior Python engineer. Be concrete and correct."


def parse_args():
    ap = argparse.ArgumentParser(description="Sample a trained student adapter")
    ap.add_argument("--adapter", required=True,
                    help="path to adapter dir, e.g. adapters/lfm25-pilot1")
    ap.add_argument("--tag", default=None,
                    help="tag for logs/train-<tag>.samples.json (default: adapter dir name)")
    ap.add_argument("--prompts-file", default=None,
                    help="JSON list of strings or {recipe,prompt} objects")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--device", type=int, default=0,
                    help="single-device index (CUDA_VISIBLE_DEVICES masks upstream)")
    return ap.parse_args()


def load_prompts(path):
    if not path:
        return list(DEFAULT_PROMPTS)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    out = []
    for item in raw:
        if isinstance(item, str):
            out.append({"recipe": "gen", "prompt": item})
        else:
            out.append({"recipe": item.get("recipe", "gen"),
                        "prompt": item["prompt"]})
    return out


def build_model(adapter, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT, quantization_config=bnb, torch_dtype=torch.float16,
        trust_remote_code=True, device_map={"": device})
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, tok


def generate_one(model, tok, prompt, args):
    import torch
    tmpl = getattr(tok, "chat_template", None)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}]
    if tmpl:
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "%s\n\nUser: %s\n\nAssistant:" % (SYSTEM, prompt)
    else:
        text = "%s\n\nUser: %s\n\nAssistant:" % (SYSTEM, prompt)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             do_sample=True, temperature=args.temperature,
                             top_p=0.9, pad_token_id=tok.pad_token_id)
    completion = tok.decode(out[0][enc["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return completion.strip()


def main():
    args = parse_args()
    tag = args.tag or os.path.basename(args.adapter.rstrip("\\/"))
    prompts = load_prompts(args.prompts_file)
    print("[sample] loading base + adapter %s" % args.adapter, flush=True)
    model, tok = build_model(args.adapter, args.device)

    samples = []
    for i, p in enumerate(prompts, 1):
        print("[sample] %d/%d %s" % (i, len(prompts), p["recipe"]), flush=True)
        comp = generate_one(model, tok, p["prompt"], args)
        rel = args.adapter if not os.path.isabs(args.adapter) else os.path.relpath(args.adapter, ROOT)
        samples.append({"recipe": p["recipe"], "prompt": p["prompt"],
                        "completion": comp, "model": STUDENT, "adapter": rel})

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    out_path = os.path.join(ROOT, "logs", "train-%s.samples.json" % tag)
    payload = {"tag": tag, "adapter": args.adapter, "student": STUDENT,
               "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "params": {"max_new_tokens": args.max_new_tokens,
                          "temperature": args.temperature},
               "samples": samples}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print("[sample] wrote %d samples -> %s" % (len(samples), out_path), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

