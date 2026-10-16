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
    ap.add_argument("--in", dest="src", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--holdout", default=None,
                    help="JSONL whose prompts must not appear in the output")
    ap.add_argument("--max-per-prompt", type=int, default=4)
    ap.add_argument("--min-answer-tokens", type=int, default=24)
    ap.add_argument("--prompt-trim", choices=["tail", "head", "none"], default="tail",
                    help="when the prompt alone fills the window: keep the end of it "
                         "(tail — the recent context is what the answer responds to), "
                         "keep its start (head), or drop the row (none = old behaviour)")
    ap.add_argument("--trim-marker", default=None,
                    help="text inserted where the prompt was cut; '{n}' is replaced "
                         "with the number of omitted characters (default: an "
                         "'[... N earlier/later characters omitted ...]' line)")
    ap.add_argument("--min-kept-prompt-chars", type=int, default=200,
                    help="do not trim a prompt below this many characters; a row that "
                         "cannot be trimmed this far is dropped instead")
    ap.add_argument("--tokenizer", default=STUDENT)
    ap.add_argument("--selftest", action="store_true",
                    help="offline checks (trim boundaries, markers, the fit guarantee)")
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


# Longest first, so the *kind* of a cut point is named by the strongest boundary it
# sits on. Used for the provenance of a trimmed row.
SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def boundary_kind(text, pos):
    """Which separator the cut at `pos` lands on (for attribution)."""
    for sep in SEPARATORS:
        if pos >= len(sep) and text[pos - len(sep):pos] == sep:
            return {"\n\n": "paragraph", "\n": "line", ". ": "sentence",
                    "? ": "sentence", "! ": "sentence", "; ": "clause",
                    ", ": "clause", " ": "word"}[sep]
    return "char"


def trim_marker(omitted, at, template=None):
    """The line that marks where a prompt was cut, in the direction it was cut.

    The marker is part of the *prompt*, never of the answer: a marker inside a
    completion teaches the model to emit markers.
    """
    if template is not None:
        return template.format(n=omitted)
    side = "earlier" if at == "tail" else "later"
    return "[... %d %s characters omitted ...]" % (omitted, side)


def trim_prompt(text, keep_chars, at="tail", template=None, min_kept=200):
    """Keep one end of a too-long prompt, cut at a natural boundary.

    Returns `(kept_text, info)` where info records exactly what was dropped — the
    character the cut was made at, how much went, and which boundary it landed on —
    so a trimmed row can be attributed later instead of being an unexplained long
    prompt. `(None, info)` means the budget cannot be met without destroying the
    prompt (nothing to keep, or less than `min_kept` characters survive), in which
    case the caller drops the row and says so.

    'tail' keeps the end (the recent context an answer responds to); 'head' keeps the
    beginning (question-first corpora).
    """
    info = {"at": at, "chars_before": len(text), "kept_chars": 0,
            "dropped_chars": 0, "cut_at_char": None, "boundary": None,
            "marker": "", "marker_chars": 0}
    if not text:
        return None, info
    cands = boundaries(text)
    if at == "tail":
        # Smallest boundary at or after (len - keep): shortest possible cut that
        # leaves at most `keep_chars`.
        target = max(len(text) - keep_chars, 0)
        pos = next((p for p in cands if p >= target), None)
        if pos is None or len(text) - pos < min_kept:
            return None, info
        kept, dropped = text[pos:], pos
    else:
        pos = next((p for p in reversed(cands) if p <= keep_chars), None)
        if pos is None or pos < min_kept:
            return None, info
        kept, dropped = text[:pos], len(text) - pos
    marker = trim_marker(dropped, at, template)
    kept = (marker + "\n" + kept.lstrip("\n")) if at == "tail" \
        else (kept.rstrip() + "\n" + marker)
    info.update({"kept_chars": len(kept), "dropped_chars": dropped,
                 "cut_at_char": pos, "boundary": boundary_kind(text, pos),
                 "marker": marker, "marker_chars": len(marker)})
    return kept, info


def main():
    a = parse_args()
    if a.selftest:
        return selftest()
    if not a.src or not a.out:
        raise SystemExit("--in and --out are required (or use --selftest)")
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

    kept, dropped = [], {"prompt_too_long": 0, "prompt_too_long_after_trim": 0,
                         "anchor_holdout": 0, "answer_too_short_after_trim": 0,
                         "duplicate_cap": 0}
    trimmed = 0            # rows whose answer was trimmed
    prompt_trimmed = 0     # rows whose prompt was trimmed instead of dropped
    rewrites = []          # attribution: one entry per rewritten row
    tok_in = tok_out = 0
    for idx, rec in enumerate(rows):
        head, pre, _ = ts.record_pieces(rec, tok)
        if head is None:
            head = ("%s\n\nUser: %s\n\nAssistant: "
                    % (rec.get("system") or "You are a helpful Python engineer.",
                       rec["prompt"]))
        base = n_tokens(head + pre)
        if prompt_key(rec) in holdout:
            dropped["anchor_holdout"] += 1
            continue
        new = dict(rec)
        prompt_trim = None
        if base >= a.max_len:
            # The prompt alone fills the window. Dropping the row throws away the
            # hardest example in the corpus; training it unmasked teaches nothing but
            # the prompt. So trim the prompt instead — cut at a natural boundary,
            # leave a marker where the cut happened, and record exactly what went
            # missing so the row can be attributed later (2026-10-16).
            if a.prompt_trim == "none":
                dropped["prompt_too_long"] += 1
                continue
            hint = rec["prompt"]
            bounds = boundaries(hint)
            ks = sorted((len(hint) - p for p in bounds) if a.prompt_trim == "tail"
                        else bounds)

            def _render(prompt_text):
                r2 = dict(rec, prompt=prompt_text)
                h2, p2, _ = ts.record_pieces(r2, tok)
                if h2 is None:
                    h2 = ("%s\n\nUser: %s\n\nAssistant: "
                          % (r2.get("system") or "You are a helpful Python engineer.",
                             prompt_text))
                return h2, p2

            def _search(strict):
                """Largest kept prompt that leaves room (binary search: monotone)."""
                lo, hi, found = 0, len(ks) - 1, None
                while lo <= hi:
                    mid = (lo + hi) // 2
                    kept_text, info = trim_prompt(hint, ks[mid], at=a.prompt_trim,
                                                  template=a.trim_marker,
                                                  min_kept=a.min_kept_prompt_chars)
                    if kept_text is not None:
                        h2, p2 = _render(kept_text)
                        if strict:
                            fits = n_tokens(h2 + p2 + rec["completion"] + end) <= a.max_len
                        else:
                            # The answer can be trimmed too, and that path records its
                            # own cut — so a prompt trim is also acceptable when only
                            # *some* answer of at least the minimum size fits. Keeping
                            # the row (with both cuts written down) beats dropping it.
                            fits = (n_tokens(h2 + p2 + end) + a.min_answer_tokens
                                    <= a.max_len)
                        if fits:
                            found, lo = (kept_text, info, h2, p2), mid + 1
                            continue
                    hi = mid - 1
                return found

            # Prefer leaving the answer untouched; if that is impossible, take the
            # prompt trim that makes room for the smallest acceptable answer.
            best = _search(True) or _search(False)
            if best is None:
                # Trimming cannot rescue it (the prompt is one unbreakable word, or
                # even a minimal answer does not fit beside the marker).
                dropped["prompt_too_long_after_trim"] += 1
                continue
            kept_text, prompt_trim, head, pre = best
            base = n_tokens(head + pre)
            new["prompt"] = kept_text
            meta = dict(new.get("meta") or {})
            fit = dict(meta.get("fit") or {})
            fit.update({"trimmed": True, "max_len": a.max_len,
                        "policy": "prompt-%s@%s" % (prompt_trim["at"],
                                                    prompt_trim["boundary"]),
                        "prompt": dict(prompt_trim, trimmed=True)})
            meta["fit"] = fit
            new["meta"] = meta
            prompt_trimmed += 1
        comp = rec["completion"]
        if n_tokens(head + pre + comp + end) <= a.max_len:
            atk = answer_tokens(head, pre, comp)
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
            new["completion"] = cut
            meta = dict(new.get("meta") or {})
            fit = dict(meta.get("fit") or {})
            # The answer gets no marker: a marker inside the completion would teach
            # the model to emit markers. The cut is recorded instead, so a behaviour
            # change can still be attributed to a specific row's truncation.
            fit.update({"trimmed": True, "max_len": a.max_len,
                        "chars_before": len(comp), "chars_after": len(cut),
                        "answer": {"trimmed": True, "chars_before": len(comp),
                                   "chars_after": len(cut), "cut_at_char": best,
                                   "boundary": boundary_kind(comp, best),
                                   "answer_tokens": atk, "marker": None,
                                   "note": "no marker in the answer — one there "
                                           "would be trained on"}})
            meta["fit"] = fit
            new["meta"] = meta
            trimmed += 1
        tok_in += n_tokens(head + pre + comp + end)
        tok_out += n_tokens(head + pre + new["completion"] + end)
        new_fit = ((new.get("meta") or {}).get("fit") or {})
        if new_fit.get("trimmed"):
            rewrites.append({"index": idx, "id": rec.get("id"),
                             "rows": len(rows),
                             "prompt": new_fit.get("prompt"),
                             "answer": new_fit.get("answer")})
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
                   "prompt_trim": a.prompt_trim,
                   "trim_marker": (a.trim_marker
                                   or "(auto: [... N earlier/later characters omitted ...])"),
                   "min_kept_prompt_chars": a.min_kept_prompt_chars,
                   "holdout": (os.path.relpath(abspath(a.holdout), ROOT)
                               if a.holdout else None),
                   "holdout_prompts": len(holdout)},
        "dropped": dropped,
        "trimmed_rows": trimmed,                 # answers trimmed to the window
        "prompt_trimmed_rows": prompt_trimmed,   # rows rescued by trimming the prompt
        # Attribution: exactly which row was cut, where, and how much went missing,
        # so a later behaviour change can be traced to a specific truncation instead
        # of to "the corpus".
        "rewritten_rows": rewrites,
        "estimated_full_row_tokens": {"in": tok_in, "out": tok_out},
    }
    with open(out + ".filter.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=1)
    print("[fit] %d rows -> %d kept (%d prompt-trimmed, %d answer-trimmed)"
          % (len(rows), len(final), prompt_trimmed, trimmed))
    print("[fit] dropped: %s"
          % ", ".join("%s=%d" % kv for kv in dropped.items() if kv[1]))
    if rewrites:
        def _where(w):
            src_ = w.get("prompt") or w.get("answer") or {}
            return "%s:%s" % (w.get("id"), src_.get("boundary") or "?")
        print("[fit] cut where: %s%s"
              % (", ".join(_where(w) for w in rewrites[:6]),
                 " ..." if len(rewrites) > 6 else ""))
    print("[fit] full-row tokens %d -> %d" % (tok_in, tok_out))
    print("[fit] wrote %s (+ .filter.json)" % os.path.relpath(out, ROOT))
    return 0


def selftest():
    """Offline checks: trim boundaries, markers, and the fit guarantee.

    The tokenizer is real (the guarantee is about tokens), the corpus is synthetic,
    and everything lands in cache/fit-selftest which is removed afterwards.
    """
    import subprocess
    import train_student as ts
    from transformers import AutoTokenizer

    ok = True

    def check(good, label):
        nonlocal ok
        ok = ok and bool(good)
        print("[selftest] %s %s" % ("ok  " if good else "FAIL", label))

    text = "first sentence here.\nsecond line.\n\nthird paragraph ends it."
    kept, info = trim_prompt(text, 20, at="tail", min_kept=5)
    check(kept and info["kept_chars"] <= len(text) + info["marker_chars"],
          "a tail trim reports what it kept")
    check(kept.startswith("[...") and info["dropped_chars"] > 0
          and info["boundary"] in ("paragraph", "line", "sentence", "word", "char"),
          "the cut is marked and named by its boundary (%s after %d chars)"
          % (info["boundary"], info["dropped_chars"]))
    head, hinfo = trim_prompt(text, 22, at="head", min_kept=5)
    check(head.startswith("first") and head.rstrip().endswith("...]"),
          "a head trim keeps the start and marks the break at the end")
    check(trim_prompt("short", 100, at="tail", min_kept=200)[0] is None,
          "a prompt that cannot be trimmed far enough is refused, not mangled")
    check("earlier" in trim_marker(1234, "tail") and "later" in trim_marker(9, "head"),
          "the marker says which side went missing")

    tmp = os.path.join(ROOT, "cache", "fit-selftest")
    os.makedirs(tmp, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    src = os.path.join(tmp, "probe.jsonl")
    probe = [
        {"id": "s-1", "system": "You are a helpful Python engineer.",
         "prompt": "What is 2+2?", "completion": "4."},
        {"id": "s-2", "system": "You are a helpful Python engineer.",
         "prompt": "x " * 4000, "completion": "42."},
        {"id": "s-3", "system": "You are a helpful Python engineer.",
         "prompt": "Say hello.", "completion": "hello " * 3000},
    ]
    with open(src, "w", encoding="utf-8") as fh:
        for r in probe:
            fh.write(json.dumps(r) + "\n")

    def run(extra):
        return subprocess.run([sys.executable, os.path.abspath(__file__)] + extra,
                              capture_output=True, text=True, cwd=ROOT)

    fit_out = os.path.join(tmp, "probe-fit.jsonl")
    r = run(["--in", src, "--out", fit_out, "--max-len", "512",
             "--prompt-trim", "tail"])
    kept_rows = load_rows(fit_out) if os.path.isfile(fit_out) else []
    check(r.returncode == 0 and len(kept_rows) == 3,
          "prompt trimming keeps all three rows (dropping kept two)")
    trimmed = [x for x in kept_rows
               if (((x.get("meta") or {}).get("fit") or {})
                   .get("prompt") or {}).get("trimmed")]
    check([t["id"] for t in trimmed] == ["s-2"],
          "the long-prompt row is the one marked prompt-trimmed")
    check(trimmed and "omitted" in ((trimmed[0]["meta"]["fit"]["prompt"]["marker"] or "")),
          "and it carries a marker where the cut happened")
    check(trimmed and trimmed[0]["meta"]["fit"]["prompt"]["cut_at_char"] > 0,
          "with the character the cut was made at recorded")
    no_marker = [x for x in kept_rows
                 if "omitted" in (x.get("completion") or "")]
    check(not no_marker, "no marker is ever put inside an answer")
    over = [x["id"] for x in kept_rows
            if len(tok(ts.format_record(x, tok))["input_ids"]) > 512]
    check(not over, "every kept row renders inside max_len (over: %s)" % over)

    none_out = os.path.join(tmp, "probe-none.jsonl")
    run(["--in", src, "--out", none_out, "--max-len", "512", "--prompt-trim", "none"])
    prov = {}
    if os.path.isfile(none_out + ".filter.json"):
        with open(none_out + ".filter.json", encoding="utf-8") as fh:
            prov = json.load(fh)
    check((len(load_rows(none_out)) if os.path.isfile(none_out) else 0) == 2
          and prov.get("dropped", {}).get("prompt_too_long") == 1,
          "--prompt-trim none still drops it, counted as prompt_too_long")

    again = os.path.join(tmp, "probe-fit2.jsonl")
    run(["--in", fit_out, "--out", again, "--max-len", "512"])
    same = (os.path.isfile(again)
            and open(again, encoding="utf-8").read()
            == open(fit_out, encoding="utf-8").read())
    check(same, "fitting an already-fitted corpus is a byte-identical no-op")
    prov_fit = {}
    if os.path.isfile(fit_out + ".filter.json"):
        with open(fit_out + ".filter.json", encoding="utf-8") as fh:
            prov_fit = json.load(fh)
    check(prov_fit.get("prompt_trimmed_rows") == 1
          and prov_fit.get("rewritten_rows")
          and prov_fit["rewritten_rows"][0]["id"] == "s-2",
          "the provenance names the row it cut and where")

    for path in (src, fit_out, fit_out + ".filter.json", none_out,
                 none_out + ".filter.json", again, again + ".filter.json"):
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    print("[selftest] %s" % ("ALL OK" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())