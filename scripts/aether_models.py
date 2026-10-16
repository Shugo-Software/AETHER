#!/usr/bin/env python3
"""Which base models AETHER trains, exports and verifies — in one place.

The student id used to be a constant in six scripts (`train_student.py`,
`export_model.py`, `sample_student.py`, `eval_jadis.py`, `verify_dataset.py`,
`fit_corpus.py`). That made "select the student's model" a code change instead of a
flag, and it made one specific mistake easy: merging a LoRA onto a base that never
produced it. A wrong base merges happily and produces nonsense, with no error to
notice until a human reads a generation.

So: one registry, one lookup, and every entry carries what a release has to state —
the label the UI shows, the licence that applies to derivative work, the context
window a harness may rely on, and a note for the model card. `--model` accepts an id
from here or any hub id, but *exporting* an unregistered model is refused: the card
would otherwise claim a licence and an upstream link that nobody checked.

    from aether_models import DEFAULT, adapter_dir, info, resolve
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT = "LiquidAI/LFM2.5-2.6B"

MODELS = {
    "LiquidAI/LFM2.5-2.6B": {
        "label": "LFM2.5 2.6B (base)",
        "family": "lfm2",
        "license": "LFM Open License v1.0",
        "license_url": ("https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/LICENSE"),
        "license_file": "LICENSE-UPSTREAM-LFM.md",
        "context": 32768,
        "chat_template": True,
        # What a *baseline* export of this model is called (see export_model.py
        # --base-only): the unmodified base, converted so it can be served the same
        # way as a fine-tune and compared against one.
        "export_name": "lfm25-base",
        "notes": ("The student base for every run so far. QLoRA 4-bit with fp16 "
                  "compute (Pascal 1060: no bf16, no flash-attn). Its chat template "
                  "prefills <think> in assistant turns, which is what the corpus "
                  "renderer builds on."),
    },
}

# Adapter dirs written before the model was selectable carry this prefix, so an
# existing tag still resolves to its adapter without renaming anything on disk.
LEGACY_PREFIX = "lfm25-"


def resolve(model=None):
    """The model id to use — the default when nothing was selected."""
    return (model or DEFAULT).strip()


def info(model=None):
    """What is known about a model: label, licence, context, card note."""
    m = resolve(model)
    entry = MODELS.get(m)
    if entry:
        return dict(entry, id=m, registered=True)
    return {"id": m, "label": m, "family": None, "license": None,
            "license_url": None, "license_file": None, "context": None,
            "chat_template": None, "registered": False,
            "notes": ("not in aether_models.MODELS — training and serving work, "
                      "exporting is refused until the base licence is stated.")}


def known(model=None):
    return resolve(model) in MODELS


def require_known(model, what="this"):
    """Raise unless the model is registered — used before anything is published."""
    if not known(model):
        raise SystemExit(
            "%s: %s is not in aether_models.MODELS. Add it (label, licence, "
            "licence_url, context) first — exporting without that would publish a "
            "model card that states a base licence nobody verified."
            % (what, resolve(model)))
    return resolve(model)


def adapter_dir(tag, root=None):
    """Where a run's adapter lives: `adapters/<tag>`, or the legacy `lfm25-<tag>`.

    Old runs wrote `adapters/lfm25-<tag>` and those directories are named in existing
    state files and exports, so an existing legacy directory wins. New runs use the
    plain tag — the base is recorded in the run state and in `adapter_config.json`,
    which is where a fact about a run belongs, not in its directory name.
    """
    root = root or ROOT
    legacy = os.path.join(root, "adapters", LEGACY_PREFIX + str(tag))
    if os.path.isdir(legacy):
        return legacy
    return os.path.join(root, "adapters", str(tag))


def describe(model=None):
    """One log line: id, label, licence."""
    e = info(model)
    return "%s [%s, %s]" % (e["id"], e["label"],
                            e.get("license") or "licence not registered")


def selftest():
    """Offline checks: the registry answers, and refuses what it cannot vouch for."""
    ok = True

    def check(good, label):
        nonlocal ok
        ok = ok and bool(good)
        print("[selftest] %s %s" % ("ok  " if good else "FAIL", label))

    check(resolve(None) == DEFAULT and resolve("") == DEFAULT,
          "an unset selection resolves to the default")
    check(resolve("  org/other  ") == "org/other",
          "an explicit selection is trimmed and kept")
    check(known(DEFAULT) and known(""),
          "the default is registered")
    check(not known("someone/unknown-model"),
          "an unregistered model is not silently treated as known")
    e = info("someone/unknown-model")
    check(e["label"] and not e["registered"],
          "an unregistered model still describes itself (label = id)")
    check(info(DEFAULT)["license"] and info(DEFAULT)["license_url"]
          and info(DEFAULT)["context"],
          "the default carries licence, upstream url and context")
    try:
        require_known("someone/unknown-model", "export")
        refused = False
    except SystemExit:
        refused = True
    check(refused, "publishing an unregistered base is refused")
    legacy = os.path.join(ROOT, "adapters", LEGACY_PREFIX + "jadis")
    check(adapter_dir("jadis") == legacy,
          "an existing legacy adapter dir wins (nothing is renamed on disk)")
    check(adapter_dir("brand-new-run").endswith(os.path.join("adapters",
                                                            "brand-new-run")),
          "a new run gets adapters/<tag>")
    check(describe(DEFAULT).startswith(DEFAULT) and "LFM" in describe(DEFAULT),
          "describe() names the model and its licence")
    print("[selftest] %s" % ("ALL OK" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print("default : %s" % DEFAULT)
    for mid in sorted(MODELS):
        e = MODELS[mid]
        print("  %-26s %-24s %s" % (mid, e["label"], e.get("license") or "-"))
