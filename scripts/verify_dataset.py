#!/usr/bin/env python3
"""Verify a dataset: composition, rendering, truncation and contamination.

Answers "what is the model actually learning this round?" — not what the corpus
looks like on paper. It applies the SAME renderer the trainer uses
(`train_student.format_record`) and the same tokenizer, then measures per row:

  * composition (sources, recipes, prompt/completion char stats)
  * tokens at --max-len, split into prompt tokens and completion tokens
  * truncation class: "complete" (fits), "partial" (answer partly cut),
    "answer_cut" (the answer never appears — the row only teaches prompt tokens),
    and whether `<|im_end|>` survives (does the model learn to stop?)
  * duplicate prompts inside the file, and prompt overlap with another file
    (leakage: is the anchor/eval set inside the training corpus?)

Usage:
  python scripts/verify_dataset.py --jsonl datasets/phase6_pro1.jsonl --max-len 1536
  python scripts/verify_dataset.py --jsonl datasets/phase6_pro1.jsonl \\
      --overlap datasets/phase6_pilot1.jsonl --json logs/verify-dataset-pro1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aether_env  # noqa: E402 - caches/temp stay on the repo drive (never C:)
aether_env.pin_caches()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
STUDENT = "LiquidAI/LFM2.5-2.6B"
END_MARK = "<|im_end|>"


def parse_args():
    ap = argparse.ArgumentParser(description="Dataset verification")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--max-len", type=int, default=1536,
                    help="the training max_len to audit against")
    ap.add_argument("--overlap", default=None,
                    help="another JSONL to check prompt leakage against")
    ap.add_argument("--samples", type=int, default=3,
                    help="how many rendered rows to show")
    ap.add_argument("--json", default=None, help="write the report as JSON here")
    ap.add_argument("--tokenizer", default=STUDENT)
    return ap.parse_args()


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(n, d):
    return "%.0f%%" % (100.0 * n / d) if d else "n/a"


def _has_subseq(ids, sub, window=4):
    """True if `sub` occurs within the last `window` tokens of `ids`.

    The rendered row ends with `...answer<|im_end|>\\n`, so the terminator is
    the second-to-last token, not the last — comparing only the final token
    reports 0% even for rows that end perfectly.
    """
    if not sub or len(ids) < len(sub):
        return False
    start = max(0, len(ids) - window - len(sub) + 1)
    for i in range(start, len(ids) - len(sub) + 1):
        if ids[i:i + len(sub)] == sub:
            return True
    return False


def prompt_key(row):
    return hashlib.sha1((row.get("prompt") or "").encode("utf-8")).hexdigest()


def head_text(rec, tok):
    """The generation-prompt prefix the trainer puts before the answer."""
    sys_p = rec.get("system") or "You are a helpful Python engineer."
    tmpl = getattr(tok, "chat_template", None)
    if tmpl:
        try:
            return tok.apply_chat_template(
                [{"role": "system", "content": sys_p},
                 {"role": "user", "content": rec["prompt"]}],
                tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001 — mirror format_record's fallback
            pass
    return f"{sys_p}\n\nUser: {rec['prompt']}\n\nAssistant: "


def audit(rows, tok, max_len, fmt):
    """Token-level audit: what survives truncation, per row."""
    end_ids = tok.encode(END_MARK, add_special_tokens=False)
    counts = {"complete": 0, "partial": 0, "answer_cut": 0}
    prompt_tok = comp_tok = comp_tok_full = kept_total = 0
    lens, heads, kept_comp = [], [], []
    term = 0
    detail = []
    for i, rec in enumerate(rows):
        full = fmt(rec, tok)
        ids = tok(full)["input_ids"]
        n_full = len(ids)
        h_ids = tok(head_text(rec, tok))["input_ids"]
        n_head = len(h_ids)
        n_comp = max(n_full - n_head, 0)
        kept = ids[:max_len]
        survives = min(n_full, max_len)
        comp_kept = max(survives - n_head, 0)
        if n_full <= max_len:
            cls = "complete"
        elif n_head >= max_len:
            cls = "answer_cut"
        else:
            cls = "partial"
        counts[cls] += 1
        prompt_tok += min(n_head, max_len)
        kept_total += survives
        comp_tok += comp_kept
        comp_tok_full += n_comp
        lens.append(n_full)
        heads.append(n_head)
        if cls != "complete":
            kept_comp.append(round(comp_kept / n_comp, 3) if n_comp else 0.0)
        if end_ids and _has_subseq(kept, end_ids, window=4):
            term += 1
        detail.append({"i": i, "source": rec.get("source", "?"),
                       "recipe": rec.get("recipe", "?"),
                       "cls": cls, "tokens": n_full, "head": n_head,
                       "comp_full": n_comp, "comp_kept": comp_kept,
                       "kept_pct": (round(100.0 * comp_kept / n_comp)
                                    if n_comp else None),
                       "tail": tok.decode(kept[-40:], skip_special_tokens=False)
                       .replace("\n", "\\n")})
    n = len(rows)
    med = statistics.median
    summary = {
        "rows": n,
        "max_len": max_len,
        "classes": counts,
        "class_pct": {k: round(100.0 * v / n, 1) for k, v in counts.items()},
        "answer_fully_lost": counts["answer_cut"],
        "terminator_survives": term,
        "terminator_pct": round(100.0 * term / n, 1),
        "tokens": {
            "prompt_total": prompt_tok,
            "completion_kept": comp_tok,
            "completion_available": comp_tok_full,
            "completion_kept_pct": (round(100.0 * comp_tok / comp_tok_full, 1)
                                    if comp_tok_full else None),
            "signal_ratio": (round(1.0 * comp_tok / prompt_tok, 3)
                             if prompt_tok else None),
            "kept_total": kept_total,
            "answer_loss_share": (round(1.0 * comp_tok / kept_total, 4)
                                  if kept_total else None),
        },
        "len_tokens": {"median": med(lens), "p90": sorted(lens)[int(0.9 * n) - 1],
                       "max": max(lens)},
        "head_tokens": {"median": med(heads),
                        "p90": sorted(heads)[int(0.9 * n) - 1], "max": max(heads)},
    }
    if kept_comp:
        summary["kept_fraction_when_truncated"] = {
            "median": med(kept_comp), "min": min(kept_comp)}
    return summary, detail


def composition(rows):
    src, rec_, pchars, cchars = {}, {}, [], []
    for r in rows:
        src[r.get("source", "?")] = src.get(r.get("source", "?"), 0) + 1
        rec_[r.get("recipe", "?")] = rec_.get(r.get("recipe", "?"), 0) + 1
        pchars.append(len(r.get("prompt") or ""))
        cchars.append(len(r.get("completion") or ""))
    med = statistics.median
    return {"by_source": dict(sorted(src.items(), key=lambda kv: -kv[1])),
            "by_recipe": dict(sorted(rec_.items(), key=lambda kv: -kv[1])),
            "prompt_chars": {"median": med(pchars), "max": max(pchars)},
            "completion_chars": {"median": med(cchars), "max": max(cchars)},
            "fields": sorted({k for r in rows for k in r})}


def duplicates(rows):
    seen = {}
    for r in rows:
        k = prompt_key(r)
        seen.setdefault(k, {"n": 0, "sample": (r.get("prompt") or "")[:70]})
        seen[k]["n"] += 1
    dupes = sorted([v for v in seen.values() if v["n"] > 1],
                   key=lambda v: -v["n"])
    return {"unique_prompts": len(seen), "duplicate_groups": len(dupes),
            "rows_in_duplicate_groups": sum(v["n"] for v in dupes),
            "top": dupes[:5]}


def print_report(path, rows, comp, summ, det, dup, overlap, max_len, n_samples):
    print("=" * 78)
    print("DATASET %s" % path)
    print("=" * 78)
    print("rows %d | size %.1f MB | fields %s"
          % (len(rows), os.path.getsize(path) / 1e6, ",".join(comp["fields"])))
    print("\ncomposition")
    print("  by source   : %s" % comp["by_source"])
    print("  by recipe   : %s" % comp["by_recipe"])
    print("  prompt chars: median %s / max %s"
          % (comp["prompt_chars"]["median"], comp["prompt_chars"]["max"]))
    print("  answer chars: median %s / max %s"
          % (comp["completion_chars"]["median"], comp["completion_chars"]["max"]))

    print("\ntokens at max_len=%d (rendered exactly as the trainer renders it)"
          % max_len)
    print("  full-row tokens : median %s | p90 %s | max %s"
          % (summ["len_tokens"]["median"], summ["len_tokens"]["p90"],
             summ["len_tokens"]["max"]))
    print("  prompt tokens   : median %s | p90 %s | max %s"
          % (summ["head_tokens"]["median"], summ["head_tokens"]["p90"],
             summ["head_tokens"]["max"]))
    c = summ["classes"]
    print("  complete %d (%s) | partial %d (%s) | ANSWER CUT %d (%s)"
          % (c["complete"], pct(c["complete"], len(rows)), c["partial"],
             pct(c["partial"], len(rows)), c["answer_cut"],
             pct(c["answer_cut"], len(rows))))
    print("  terminator <|im_end|> survives in %d rows (%s)"
          % (summ["terminator_survives"],
             "%.1f%%" % (100.0 * summ["terminator_survives"] / len(rows))))
    t = summ["tokens"]
    print("  answer tokens kept %d of %d (%s%%)  signal/prompt ratio %.3f"
          % (t["completion_kept"], t["completion_available"],
             t["completion_kept_pct"], t["signal_ratio"]))
    print("  loss-relevant tokens kept %d | answers are %s%% of them"
          % (t["kept_total"], round(100 * t["answer_loss_share"], 1)))
    print("  -> with prompt loss MASKED the model learns %d answer tokens;"
          " with prompt loss UNMASKED only %s%% of the gradient is on answers"
          % (t["completion_kept"], round(100 * t["answer_loss_share"], 1)))
    if "kept_fraction_when_truncated" in summ:
        k = summ["kept_fraction_when_truncated"]
        print("  when truncated: median %s%% of the answer survives (worst %s%%)"
              % (round(100 * k["median"]), round(100 * k["min"])))

    print("\nduplicates")
    print("  %d unique prompts / %d rows | %d duplicate groups covering %d rows"
          % (dup["unique_prompts"], len(rows), dup["duplicate_groups"],
             dup["rows_in_duplicate_groups"]))
    for v in dup["top"]:
        print("    x%-3d %s" % (v["n"], v["sample"].replace("\n", " ")))

    if overlap is not None:
        print("\noverlap with %s" % overlap["path"])
        print("  %d of %d prompts (%.1f%%) also appear there"
              % (overlap["shared"], len(rows), 100.0 * overlap["shared"] / len(rows)))
        for s in overlap["samples"]:
            print("    %s" % s.replace("\n", " "))

    print("\nsamples (rendered as the model sees them)")
    by_cut = sorted(det, key=lambda d: -(d["comp_full"] - d["comp_kept"]))
    picks = [d for d in by_cut if d["cls"] != "complete"][:n_samples]
    whole = [d for d in det if d["cls"] == "complete"]
    if whole:
        picks.append(whole[0])
    for d in picks:
        r = rows[d["i"]]
        print("\n  [row %d] source=%s recipe=%s -> %s"
              % (d["i"], d["source"], d["recipe"], d["cls"].upper()))
        print("    tokens %d (prompt %d + answer %d) | answer kept %d"
              " (%s%%)" % (d["tokens"], d["head"], d["comp_full"],
                           d["comp_kept"], d["kept_pct"]))
        print("    PROMPT: %s" % (r["prompt"][:170].replace("\n", " ") + "..."))
        print("    ANSWER: %s"
              % (r["completion"][:170].replace("\n", " ") + "..."))
        print("    trained text ends with: \"%s\"" % d["tail"][-150:])


def main():
    a = parse_args()
    jsonl = a.jsonl if os.path.isabs(a.jsonl) else os.path.join(ROOT, a.jsonl)
    if not os.path.isfile(jsonl):
        raise SystemExit("no such file: %s" % jsonl)
    import train_student  # the trainer's own renderer (single source of truth)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    rows = load_rows(jsonl)
    print("[verify] tokenizer=%s rows=%d" % (a.tokenizer, len(rows)), flush=True)
    summ, det = audit(rows, tok, a.max_len, train_student.format_record)
    comp = composition(rows)
    dup = duplicates(rows)
    overlap = None
    if a.overlap:
        op = a.overlap if os.path.isabs(a.overlap) else os.path.join(ROOT, a.overlap)
        orows = load_rows(op)
        mine = {prompt_key(r) for r in rows}
        theirs = {}
        for r in orows:
            theirs.setdefault(prompt_key(r), (r.get("prompt") or "")[:120])
        shared = sorted(set(theirs) & mine)
        overlap = {"path": os.path.relpath(op, ROOT),
                   "theirs_rows": len(orows), "shared": len(shared),
                   "samples": [theirs[k] for k in shared[:5]]}
    print_report(jsonl, rows, comp, summ, det, dup, overlap, a.max_len,
                 a.samples)
    if a.json:
        out = a.json if os.path.isabs(a.json) else os.path.join(ROOT, a.json)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"file": os.path.relpath(jsonl, ROOT),
                       "composition": comp, "tokens": summ,
                       "counts": summ.get("tokens") or {},
                       "duplicates": dup, "overlap": overlap,
                       "truncated_rows": [d for d in det
                                          if d["cls"] != "complete"]},
                      fh, indent=1)
        print("\n[verify] report -> %s" % os.path.relpath(out, ROOT))
    bad = summ["classes"]["answer_cut"]
    print("[verify] %s: %d/%d rows lose their answer entirely at max_len=%d"
          % (os.path.basename(jsonl), bad, len(rows), a.max_len))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
