#!/usr/bin/env python3
"""Release check: does the fine-tune actually behave like Jadis?

Side-by-side base vs base+adapter on the same prompts, with automatic metrics so
the verdict is evidence instead of vibes:

  * think-close rate — did the model leave the prefilled <think> block?
  * preamble rate    — the trained system prompt forbids preambles/"here's how"
  * format markers   — per-recipe structure the corpus taught (Q1:/A1:, ...)
  * memory overlap   — longest run of words shared with the training completion
                       (in-domain prompts) => memorized vs generalized
  * length + seconds per generation

Two prompt conventions are supported because they are NOT equivalent:
  template — apply_chat_template(add_generation_prompt=True), i.e. ends with
             "<|im_start|>assistant\\n<think>" (what llama.cpp --jinja does)
  raw      — the same head WITHOUT the <think> prefill. train_student.py used to
             render assistant turns with no generation prompt, so the older
             adapters were trained on this shape and are off-distribution with
             the template prefill.

Usage:
  python scripts/eval_jadis.py --adapter adapters/lfm25-jadis-starcoder \\
      --tag lfm25-jadis-starcoder --convention both
  python scripts/eval_jadis.py --adapter adapters/jadis-1.0 --tag jadis-1.0
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aether_env  # noqa: E402 - caches/temp stay on the repo drive (never C:)
aether_env.pin_caches()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import aether_models as am  # noqa: E402 - one place decides which base is trained
STUDENT = am.DEFAULT
DATASET = os.path.join(ROOT, "datasets", "phase6_pilot1.jsonl")
LOGS = os.path.join(ROOT, "logs")

# Out-of-domain probes: nothing here is in the pilot corpus, so what comes back
# is generalization (voice/format transfer), not recall.
OOD = [
    {"recipe": "ood_explain",
     "prompt": "Explain what a context manager is in Python and when you would "
               "write one, in two sentences."},
    {"recipe": "ood_identity",
     "prompt": "Who are you, and what kind of help are you good at?"},
    {"recipe": "ood_creative",
     "prompt": "Write a haiku about a GPU that is waiting for data."},
]

PREAMBLE_RE = re.compile(
    r"^\W*(sure|certainly|of course|absolutely|great question|good question|"
    r"here'?s|here is|i'?d be happy|happy to|let'?s)", re.IGNORECASE)


def parse_args():
    ap = argparse.ArgumentParser(description="Base-vs-adapter behavioural eval")
    ap.add_argument("--adapter", required=True,
                    help="adapter dir, e.g. adapters/jadis-1.0")
    ap.add_argument("--tag", default=None, help="report name (default: dir name)")
    ap.add_argument("--model", default=None,
                    help="base model the adapter was trained on (default: %s) — the "
                         "base-vs-tuned comparison is meaningless against another one"
                         % am.DEFAULT)
    ap.add_argument("--dataset", default=DATASET,
                    help="corpus for in-domain prompts + memory overlap")
    ap.add_argument("--in-domain", type=int, default=3,
                    help="how many dataset rows to probe (one per recipe)")
    ap.add_argument("--convention", default="both",
                    choices=["both", "template", "raw", "nothink"])
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048,
                    help="truncate long prompts (code-heavy rows)")
    ap.add_argument("--device", type=int, default=0,
                    help="CUDA index as torch sees it (CUDA_VISIBLE_DEVICES masks upstream)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", action="store_true",
                    help="also write logs/train-<tag>.samples.json for the dashboard")
    return ap.parse_args()


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_in_domain(rows, n):
    """One row per recipe (deterministic: the first of each), for variety."""
    seen, out = set(), []
    for r in rows:
        if r.get("recipe") in seen:
            continue
        seen.add(r.get("recipe"))
        out.append({"recipe": r.get("recipe"), "prompt": r.get("prompt"),
                    "reference": r.get("completion"), "system": r.get("system")})
        if len(out) >= n:
            break
    return out


def _words(text):
    return re.findall(r"\w+", (text or "").lower())


def memory_overlap(completion, reference):
    """Longest run of words shared with the training completion + word overlap."""
    a, b = _words(completion), _words(reference)
    if not a or not b:
        return {"max_word_run": 0, "common_word_ratio": 0.0}
    run = difflib.SequenceMatcher(None, a, b).find_longest_match(
        0, len(a), 0, len(b)).size
    common = len(set(a) & set(b))
    return {"max_word_run": run,
            "common_word_ratio": round(common / max(len(set(a)), 1), 3)}


# Per-recipe structure the corpus taught (heuristic markers, reported as-is).
FORMAT_CHECKS = {
    "qa_pairs": [("Q/A list", lambda t: len(re.findall(r"\bQ\d\s*:", t)) >= 3)],
    "explain_code": [("purpose", lambda t: "purpose" in t.lower()),
                     ("pitfall", lambda t: "pitfall" in t.lower())],
    "code_review": [("issue/risk", lambda t: bool(re.search(r"issue|risk|bug", t, re.I)))],
    "seed_gen": [("substantive", lambda t: len(t.split()) >= 60)],
}


def score(completion, probe):
    text = completion or ""
    out = {"chars": len(text), "words": len(text.split()),
           "think_closed": "</think>" in text,
           "leading_think_close": text.lstrip().startswith("</think>"),
           "preamble": bool(PREAMBLE_RE.match(text))}
    out["format"] = {name: bool(fn(text))
                     for name, fn in FORMAT_CHECKS.get(probe.get("recipe"), [])}
    if probe.get("reference"):
        out["memory"] = memory_overlap(text, probe["reference"])
    return out


def build_base(device):
    """4-bit base on one card (Pascal-safe: fp16 compute, no flash-attn)."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, quantization_config=bnb, torch_dtype=torch.float16,
        trust_remote_code=True, device_map={"": device})
    model.eval()
    return model, tok


def render(tok, system, prompt, convention):
    """Three prompt conventions, all of which this model can meet:

    template — the stock generation prompt, ends "...<|im_start|>assistant\\n<think>"
               (what llama.cpp --jinja does: the model must close the block)
    raw      — same head without any think tag (what the pre-fix trainer used)
    nothink  — same head with the block already closed ("...assistant\\n</think>\\n"),
               i.e. the model is asked to answer directly. This is the reliable
               convention for this adapter generation.
    """
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    if convention in ("raw", "nothink"):
        if not text.endswith("<think>"):
            raise SystemExit("template no longer ends with <think> — update render()")
        text = text[:-len("<think>")]
        if convention == "nothink":
            text += "</think>\n"
    return text


def generate(model, tok, probe, system, args, convention):
    import torch
    text = render(tok, system, probe["prompt"], convention)
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=args.max_prompt_tokens).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    dt = time.time() - t0
    comp = tok.decode(out[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()
    return comp, dt


def _memory_cell(s):
    m = s.get("memory")
    return "-" if not m else "%d / %s" % (m["max_word_run"], m["common_word_ratio"])


def _index(results):
    return {(r["convention"], r["model"], r["idx"]): r for r in results}


def build_report(args, probes, results, conventions, tag):
    idx = _index(results)
    lines = [
        "# Jadis behaviour eval — `%s`" % tag, "",
        "- adapter: `%s`" % args.adapter,
        "- base: `%s`, 4-bit NF4, greedy, max %d new tokens"
        % (STUDENT, args.max_new_tokens),
        "- corpus: `%s` — %d in-domain rows + %d out-of-domain probes"
        % (os.path.relpath(args.dataset, ROOT), args.in_domain, len(OOD)),
        "- generated: %s" % datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "",
        "`close` = completion contains `</think>` · `lead` = completion starts with "
        "`</think>` · `pre` = preamble/hedge detected (the trained system prompt "
        "forbids it) · `mem` = longest word run shared with the training completion "
        "(in-domain rows only; high = recall, low = generalization) · `fmt` = "
        "per-recipe format markers from the corpus.", ""]
    for convention in conventions:
        lines += ["## Convention `%s`" % convention, "",
                  "| probe | model | words | close | lead | pre | mem | fmt | s |",
                  "|---|---|---:|:--:|:--:|:--:|---:|:--:|---:|"]
        for p in probes:
            for model_name in ("base", "tuned"):
                rec = idx.get((convention, model_name, p["idx"]))
                if not rec:
                    continue
                s = rec["score"]
                fmt = ",".join("%s=%s" % (k, "y" if v else "n")
                               for k, v in (s.get("format") or {}).items()) or "-"
                lines.append("| %s | %s | %d | %s | %s | %s | %s | %s | %.1f |"
                             % (p["recipe"], model_name, s["words"],
                                "y" if s["think_closed"] else "n",
                                "y" if s["leading_think_close"] else "n",
                                "y" if s["preamble"] else "n",
                                _memory_cell(s), fmt, rec["seconds"]))
        lines.append("")
        for p in probes:
            lines += ["### `%s` — convention `%s`" % (p["recipe"], convention), "",
                      "**prompt**", "", "```", p["prompt"][:1200], "```", ""]
            for model_name in ("base", "tuned"):
                rec = idx.get((convention, model_name, p["idx"]))
                if rec:
                    lines += ["**%s**" % model_name, "", "```",
                              rec["completion"][:2500], "```", ""]
    return "\n".join(lines) + "\n"


def main():
    global STUDENT
    args = parse_args()
    # The base is a choice, and every report below must name the one actually used.
    STUDENT = am.resolve(args.model)
    print("[eval] base model: %s" % am.describe(STUDENT), flush=True)
    tag = args.tag or os.path.basename(args.adapter.rstrip("/\\"))
    rows = load_rows(args.dataset)
    probes = pick_in_domain(rows, args.in_domain) + [dict(p) for p in OOD]
    for i, p in enumerate(probes):
        p["idx"] = i
    conventions = ["template", "raw"] if args.convention == "both" \
        else [args.convention]
    system = rows[0].get("system") or "You are a helpful assistant."

    import torch
    torch.manual_seed(args.seed)
    print("[eval] adapter=%s tag=%s probes=%d conventions=%s"
          % (args.adapter, tag, len(probes), conventions), flush=True)
    model, tok = build_base(args.device)
    print("[eval] base loaded on device %d" % args.device, flush=True)
    results = []

    def run_all(model_name):
        for convention in conventions:
            for p in probes:
                comp, dt = generate(model, tok, p, p.get("system") or system,
                                    args, convention)
                results.append({"convention": convention, "model": model_name,
                                "idx": p["idx"], "recipe": p["recipe"],
                                "prompt": p["prompt"], "completion": comp,
                                "seconds": round(dt, 1),
                                "score": score(comp, p)})
                print("[eval] %-5s %-8s %-10s %5.1fs %5d words"
                      % (model_name, convention, p["recipe"], dt,
                         results[-1]["score"]["words"]), flush=True)

    run_all("base")
    from peft import PeftModel
    print("[eval] attaching adapter %s" % args.adapter, flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    run_all("tuned")

    os.makedirs(LOGS, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md_path = os.path.join(LOGS, "eval-%s.md" % tag)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(build_report(args, probes, results, conventions, tag))
    with open(os.path.join(LOGS, "eval-%s.json" % tag), "w",
              encoding="utf-8") as fh:
        json.dump({"tag": tag, "adapter": args.adapter, "student": STUDENT,
                   "generated_utc": now, "conventions": conventions,
                   "params": {"max_new_tokens": args.max_new_tokens,
                              "max_prompt_tokens": args.max_prompt_tokens,
                              "seed": args.seed, "greedy": True},
                   "probes": probes, "results": results}, fh, indent=2)
    print("[eval] report -> %s" % md_path, flush=True)

    for convention in conventions:
        for model_name in ("base", "tuned"):
            rs = [r for r in results if r["convention"] == convention
                  and r["model"] == model_name]
            n = max(len(rs), 1)
            close = sum(1 for r in rs if r["score"]["think_closed"]) / n
            lead = sum(1 for r in rs if r["score"]["leading_think_close"]) / n
            pre = sum(1 for r in rs if r["score"]["preamble"]) / n
            words = sum(r["score"]["words"] for r in rs) / n
            print("[eval] SUMMARY %-8s %-5s close=%3.0f%% lead=%3.0f%% "
                  "preamble=%3.0f%% avg_words=%4.0f"
                  % (convention, model_name, 100 * close, 100 * lead,
                     100 * pre, words), flush=True)

    if args.samples:
        samples = [{"recipe": r["recipe"], "prompt": r["prompt"],
                    "completion": r["completion"], "model": STUDENT,
                    "adapter": args.adapter}
                   for r in results
                   if r["model"] == "tuned" and r["convention"] == conventions[0]]
        sp = os.path.join(LOGS, "train-%s.samples.json" % tag)
        with open(sp, "w", encoding="utf-8") as fh:
            json.dump({"tag": tag, "adapter": args.adapter, "student": STUDENT,
                       "generated_utc": now,
                       "params": {"max_new_tokens": args.max_new_tokens,
                                  "temperature": 0.0},
                       "samples": samples}, fh, indent=2)
        print("[eval] dashboard samples -> %s" % sp, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
