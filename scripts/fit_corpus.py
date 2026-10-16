#!/usr/bin/env python3
"""Fit a corpus to the training budget. Read-only on the input.

`verify_dataset.py` measures a corpus; this rewrites it so every row is actually
learnable inside `--max-len`, which completion-only label masking makes
non-negotiable: a row whose prompt alone exceeds the budget has no answer tokens
to train on, and in a masked run such a row has no valid loss at all.

Policies applied, in order:
  1. holdout   — drop prompts that also appear in `--holdout` (the anchor set
                 must stay unseen or every comparison against it is void)
  2. fit       — rows that already fit pass through untouched; rows that do not
                 get their answer trimmed back to the last natural boundary
                 (paragraph, then line, then sentence, then word) that still
                 leaves room for the terminator; if fewer than
                 `--min-answer-tokens` survive, the row is dropped
  3. dedupe    — cap rows per identical prompt (`--max-per-prompt`), keeping the
                 rows that retain the most answer tokens

Writes the corpus plus `<out>.filter.json` provenance (input hash + policy +
counts), so the transformation is auditable and reproducible.

Usage:
  python scripts/fit_corpus.py --in datasets/phase6_pro1.jsonl \\
      --out datasets/phase6_pro1-fit.jsonl --max-len 1536 \\
      --holdout datasets/phase6_pilot1.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aether_env  # noqa: E402 - caches/temp stay on the repo drive (never C:)
aether_env.pin_caches()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
STUDENT = "LiquidAI/LFM2.5-2.6B"
END = "<|im_end|>\n"


def parse_args():
    ap = argparse.ArgumentParser(description="Fit a corpus to max_len")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--holdout", default=None,
                    help="JSONL whose prompts must not appear in the output")
    ap.add_argument("--max-per-prompt", type=int, default=4)
    ap.add_argument("--min-answer-tokens", type=int, default=24)
    ap.add_argument("--tokenizer", default=STUDENT)
    return ap.parse_args()


def abspath(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_key(rec):
    """Same key the verifier uses — one definition, no drift."""
    return hashlib.sha1((rec.get("prompt") or "").encode("utf-8")).hexdigest()


def boundaries(text):
    """Natural break points, ascending: paragraph > line > sentence > word."""
    pos = set()
    for sep in ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "):
        start = text.find(sep)
        while start != -1:
            pos.add(start + len(sep))
            start = text.find(sep, start + 1)
    return sorted(p for p in pos if p > 0)


def main():
    a = parse_args()
    src, out = abspath(a.src), abspath(a.out)
    if not os.path.isfile(src):
        raise SystemExit("no such corpus: %s" % src)
    import train_student as ts
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)

    def n_tokens(text):
        # The same call the trainer truncates with (special tokens included). A
        # budget that is measured one way and enforced another is exactly how rows
        # lose their answers while the fitter believes they fit.
        return len(tok(text)["input_ids"])

    end = ts.END_MARK
    rows = load_rows(src)
    holdout = set()
    if a.holdout:
        hp = abspath(a.holdout)
        holdout = {prompt_key(r) for r in load_rows(hp)}
        print("[fit] holdout %s: %d prompts" % (os.path.relpath(hp, ROOT),
                                                len(holdout)), flush=True)

    def answer_tokens(head, pre, comp):
        """Tokens that completion-only masking would actually train on.

        `head`/`pre`/`end` come from `train_student.record_pieces`, so the fitter and
        the trainer count the same string — one definition, no drift.
        """
        return n_tokens(head + pre + comp + end) - n_tokens(head + pre)

    kept, dropped = [], {"prompt_too_long": 0, "anchor_holdout": 0,
                         "answer_too_short_after_trim": 0, "duplicate_cap": 0}
    trimmed = 0
    tok_in = tok_out = 0
    for rec in rows:
        head, pre, _ = ts.record_pieces(rec, tok)
        if head is None:
            head = ("%s\n\nUser: %s\n\nAssistant: "
                    % (rec.get("system") or "You are a helpful Python engineer.",
                       rec["prompt"]))
        base = n_tokens(head + pre)
        if prompt_key(rec) in holdout:
            dropped["anchor_holdout"] += 1
            continue
        if base >= a.max_len:
            # The prompt alone eats the whole budget: the answer can never be
            # trained on, so the row is misleading rather than merely long.
            dropped["prompt_too_long"] += 1
            continue
        comp = rec["completion"]
        if n_tokens(head + pre + comp + end) <= a.max_len:
            new, atk = dict(rec), answer_tokens(head, pre, comp)
        else:
            # Binary search (token count is monotonic in cut position) for the
            # largest natural boundary that still fits.
            cands = boundaries(comp)
            lo, hi, best = 0, len(cands) - 1, None
            while lo <= hi:
                mid = (lo + hi) // 2
                if n_tokens(head + pre + comp[:cands[mid]] + end) <= a.max_len:
                    best, lo = cands[mid], mid + 1
                else:
                    hi = mid - 1
            cut = comp[:best].rstrip() if best is not None else None
            atk = answer_tokens(head, pre, cut) if cut else 0
            if not cut or atk < a.min_answer_tokens:
                dropped["answer_too_short_after_trim"] += 1
                continue
            new = dict(rec)
            new["completion"] = cut
            meta = dict(new.get("meta") or {})
            meta["fit"] = {"trimmed": True, "max_len": a.max_len,
                           "chars_before": len(comp), "chars_after": len(cut)}
            new["meta"] = meta
            trimmed += 1
        tok_in += n_tokens(head + pre + comp + END)
        tok_out += n_tokens(head + pre + new["completion"] + END)
        kept.append((new, atk))

    # 3. cap rows per identical prompt, keeping the richest answers first.
    groups = {}
    for i, (rec, atk) in enumerate(kept):
        groups.setdefault(prompt_key(rec), []).append((i, rec, atk))
    cap = []
    for items in groups.values():
        if len(items) > a.max_per_prompt:
            items.sort(key=lambda t: (-t[2], t[0]))
            dropped["duplicate_cap"] += len(items) - a.max_per_prompt
            items = items[:a.max_per_prompt]
        cap.extend(items)
    cap.sort(key=lambda t: t[0])  # keep original corpus order
    final = [rec for _, rec, _ in cap]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for rec in final:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    prov = {
        "tool": "scripts/fit_corpus.py",
        "input": {"path": os.path.relpath(src, ROOT), "sha256": sha256(src),
                  "rows": len(rows)},
        "output": {"path": os.path.relpath(out, ROOT), "rows": len(final)},
        "policy": {"max_len": a.max_len, "max_per_prompt": a.max_per_prompt,
                   "min_answer_tokens": a.min_answer_tokens,
                   "holdout": (os.path.relpath(abspath(a.holdout), ROOT)
                               if a.holdout else None),
                   "holdout_prompts": len(holdout)},
        "dropped": dropped,
        "trimmed_rows": trimmed,
        "estimated_full_row_tokens": {"in": tok_in, "out": tok_out},
    }
    with open(out + ".filter.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=1)
    print("[fit] %d rows -> %d kept (%d trimmed)"
          % (len(rows), len(final), trimmed))
    print("[fit] dropped: %s"
          % ", ".join("%s=%d" % kv for kv in dropped.items() if kv[1]))
    print("[fit] full-row tokens %d -> %d" % (tok_in, tok_out))
    print("[fit] wrote %s (+ .filter.json)" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())