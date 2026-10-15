#!/usr/bin/env python3
"""Export a trained LoRA adapter as a standalone, shareable model.

Stages (each opt-in):
  1. merge  — merge the adapter into the base LFM2.5-2.6B fp16 weights and save
              a plain transformers model under exports/<adapter>/ (no PEFT
              needed to load it; works with any transformers install).
  2. gguf   — convert the merged model with llama.cpp convert_hf_to_gguf.py
              (--outtype f16) and quantize each requested type with the CUDA
              build's llama-quantize (`--gguf f16,q4_k_m,q3_k_m`, …); iq* types
              also need `--imatrix` (llama-imatrix output). llama.cpp source is
              the shared pin (read-only).
  3. hub    — push the merged folder to huggingface.co, PRIVATE by default.
              Auth: HF_TOKEN env var or a prior `huggingface-cli login`
              (never taken from the dashboard/web).

Status is written to logs/export-<adapter>.json and mirrored by
export_control.py in logs/export-active.json for the dashboard.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORTS = os.path.join(ROOT, "exports")
ADAPTERS = os.path.join(ROOT, "adapters")
LOGS = os.path.join(ROOT, "logs")
STUDENT = "LiquidAI/LFM2.5-2.6B"
# llama.cpp checkout used for HF -> GGUF conversion (read-only: this pipeline
# never writes into it). Point AETHER_LLAMA_CPP_SRC at your clone.
LLAMA_SRC = os.environ.get("AETHER_LLAMA_CPP_SRC",
                           os.path.join(ROOT, "vendor", "llama.cpp"))
CONVERT = os.path.join(LLAMA_SRC, "convert_hf_to_gguf.py")
QUANTIZE = os.path.join(ROOT, "builds", "cuda", "bin", "llama-quantize.exe")
# GGUF file types llama-quantize understands, mapped from our short names.
# "f16" = no quantize (the conversion output itself). The small K-quants are
# what matters when the target is a 3 GB card (see docs/export.md).
QUANT_TYPES = {
    "f16": None,
    "q8_0": "Q8_0",
    "q6_k": "Q6_K",
    "q5_k_m": "Q5_K_M",
    "q5_k_s": "Q5_K_S",
    "q4_k_m": "Q4_K_M",
    "q4_k_s": "Q4_K_S",
    "q3_k_m": "Q3_K_M",
    "q3_k_s": "Q3_K_S",
    "iq4_xs": "IQ4_XS",   # needs an importance matrix: see IMATRIX_QUANTS
}
# Types that want an importance matrix (llama-imatrix over a calibration text);
# quantizing these without one gives a visibly worse model, so require --imatrix.
IMATRIX_QUANTS = {"iq4_xs"}

BASE_LICENSE_NAME = "LFM Open License v1.0"
# Front-matter needs a slug (HF validates /^[a-z0-9-.]+$/); the prose uses the
# human-readable BASE_LICENSE_NAME.
BASE_LICENSE_SLUG = "lfm-open-license-v1.0"
BASE_LICENSE_URL = "https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/LICENSE"
IMATRIX = os.path.join(ROOT, "builds", "cuda", "bin", "llama-imatrix.exe")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hf_env():
    """All HF/temp caches on G: (C: is small — plan.md)."""
    env = dict(os.environ)
    hf = os.path.join(ROOT, "cache", "hf")
    env.update({"HF_HOME": hf, "HF_HUB_CACHE": hf, "HF_DATASETS_CACHE": hf,
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TEMP": os.path.join(ROOT, "cache", "temp"),
                "TMP": os.path.join(ROOT, "cache", "temp")})
    return env


class State:
    """logs/export-<adapter>.json status writer (dashboard reads this)."""

    def __init__(self, adapter, out_dir):
        self.path = os.path.join(LOGS, "export-%s.json" % adapter)
        self.base = {"adapter": adapter, "out": os.path.relpath(out_dir, ROOT),
                     "started_utc": _now()}
        self.write("starting")

    def write(self, status, **kw):
        rec = dict(self.base)
        rec.update(kw)
        rec["status"] = status
        rec["updated_utc"] = _now()
        if status in ("done", "failed"):
            rec["finished_utc"] = _now()
        os.makedirs(LOGS, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)


def _snapshot_dir():
    """The cached HF snapshot of the base model (for custom .py code files)."""
    hits = glob.glob(os.path.join(ROOT, "cache", "hf",
                                  "models--LiquidAI--LFM2.5-2.6B",
                                  "snapshots", "*"))
    return hits[0] if hits else None


def _copy_custom_code(out_dir):
    """Copy the base repo's custom modeling/tokenizer code into the export.

    LFM2.5 ships trust_remote_code files (auto_map in config.json points at
    them); without these the export only loads with network access.
    """
    snap = _snapshot_dir()
    if not snap:
        return []
    copied = []
    for name in os.listdir(snap):
        src = os.path.join(snap, name)
        dst = os.path.join(out_dir, name)
        if not os.path.isfile(src) or os.path.exists(dst):
            continue  # never overwrite what save_pretrained just wrote
        if name.endswith((".py", ".tiktoken", ".model", ".txt")) \
                or name in ("tokenizer.json",):
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


_MODEL_CARD = """\
---
license: other
license_name: {license_slug}
license_link: {license_url}
base_model: {base}
pipeline_tag: text-generation
library_name: transformers
language:
- en
tags:
- lfm2
- lfm
- jadis
- samurai
- distillation
- qlora
---

# {title}

**{title}** is a compact instruction model in the **Jadis** line by **SamurAI**,
distilled and trained locally by the AETHER pipeline (local teacher → student
LoRA → merged checkpoint). It is a **full fine-tune**: the adapter has been merged
into the base weights, so the folder loads with plain `transformers` — no PEFT,
no adapter bookkeeping.

## Lineage

`{title}` is a derivative of [`{base}`](https://huggingface.co/{base}) by Liquid
AI. Base architecture, tokenizer and chat template are unchanged; the upstream
license (**{license_name}**) governs this model too and ships here as `LICENSE`.
The distillation teacher generated training text only — **none of its weights are
present in this model**.

## Method

| | |
|---|---|
| Student | `{base}` — 2.7 B params, `lfm2` architecture, 30 layers |
| Adapter | `{adapter}` — QLoRA: 4-bit NF4 base, fp16 compute, r=16, α=32, merged with `merge_and_unload()` |
| Data | {data_note} |
| Turn format | native LFM2.5 turns (see *Thinking* below) |
| Verification | merged safetensors header asserted all-`F16`, plus a real llama.cpp generation per GGUF (`scripts/verify_export.py`) |

> **Scope — measured, not marketing.** This corpus teaches *voice and format*:
> direct, preamble-free, structured answers with no meta-commentary. It carries no
> identity/persona data, so the model still describes itself with the base model's
> words when asked "who are you?". Treat it as a behaviour/style fine-tune of a
> small model, not a knowledgeable assistant.

## Thinking

The LFM2.5 template always prefills the assistant turn with `<|im_start|>assistant\\n<think>`, and it treats `</think>` as the reasoning/answer split (when replaying
history it keeps only the text after `</think>`). This adapter was trained with
that block **pre-closed** — the training turn is `…assistant`, then
`<think></think>`, then the answer — but measurement showed the model still opens
with analysis when the stock template hands it an *open* `<think>`: a single
closing token is a weak signal at this training scale. So **the template shipped
here pre-closes the block** (`…assistant`, then `</think>`, then a newline), the
shape this model answers best from: direct, preamble-free replies with no hidden
reasoning budget. To get upstream LFM2.5 thinking back, restore the upstream
prefill (`<think>` instead of `</think>`) in `chat_template.jinja` and rebuild the
GGUFs.

## Run it

`transformers` — fp16, ~5.4 GB RAM/VRAM:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained("{repo}", torch_dtype="float16")

msgs = [{{"role": "user", "content": "Explain what a context manager is, in two sentences."}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
out = model.generate(ids, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

llama.cpp (GGUF) — `lfm2` support, build `f9a95ae` or newer{gguf_repo_note}:

```
llama-cli -m {quant_example} -c 4096 -cnv --jinja
```

{gguf_table}
## License

{license_name} — see the [upstream license]({license_url}) and the copy shipped
in this repo. Derivatives are permitted; commercial use is subject to the revenue
thresholds in the upstream license. This model's weights derive from the base
model and remain under that license.
"""


def _read_train_log(adapter_name):
    """The run record (dataset/rows/epochs/lr) written by train_student.py."""
    try:
        with open(os.path.join(LOGS, "train-%s.json" % adapter_name),
                  encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _data_note(adapter_name):
    """Honest, automatic description of what the student actually saw."""
    rec = _read_train_log(adapter_name)
    if not rec:
        return "a small curated distillation corpus (see the training run record)"
    bits = []
    if rec.get("rows"):
        bits.append("%s curated distillation examples" % rec["rows"])
    if rec.get("epochs"):
        bits.append("%s epochs" % rec["epochs"])
    if rec.get("lr"):
        bits.append("lr %s" % rec["lr"])
    note = ", ".join(bits) if bits else "a small curated distillation corpus"
    if rec.get("resumed_from"):
        note += " (continued from `%s`)" % os.path.basename(rec["resumed_from"])
    return note + " — a focused pilot corpus, so expect strong voice and format " \
                  "adherence rather than broad knowledge"


def _gguf_table(gguf_files):
    """Markdown table of the GGUF artifacts with their real sizes."""
    if not gguf_files:
        return ""
    rows = ["| File | Size | Notes |", "|---|---:|---|"]
    for path in gguf_files:
        name = os.path.basename(path)
        try:
            mib = os.path.getsize(path) / 1024 / 1024
        except OSError:
            continue
        size = ("%.2f GiB" % (mib / 1024)) if mib >= 1024 else ("%d MiB" % mib)
        note = "f16 (reference)" if name.endswith("-f16.gguf") else "quantized"
        if any(t in name for t in ("-q4_k_m", "-q4_k_s", "-q3_", "-iq4_xs")):
            note += " — fits a 3 GB GPU"
        rows.append("| `%s` | %s | %s |" % (name, size, note))
    return "\n".join(rows) + "\n\n"


def _write_card(out_dir, name, title, gguf_files, repo=None, gguf_repo=None):
    """Write README.md (Jadis-branded card) for the export."""
    if gguf_files:
        pref = [f for f in gguf_files if f.endswith("-q4_k_m.gguf")]
        quant_example = os.path.basename(pref[0] if pref else gguf_files[-1])
    else:
        quant_example = "%s-f16.gguf" % name
    repo_note = ""
    if gguf_repo:
        repo_note = (" (GGUF files: [`%s`](https://huggingface.co/%s))"
                     % (gguf_repo, gguf_repo))
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(_MODEL_CARD.format(
            title=title, base=STUDENT, adapter=name,
            license_name=BASE_LICENSE_NAME, license_slug=BASE_LICENSE_SLUG,
            license_url=BASE_LICENSE_URL,
            repo=repo or "your-org/%s" % name,
            data_note=_data_note(name), gguf_table=_gguf_table(gguf_files),
            gguf_repo_note=repo_note, quant_example=quant_example))


def _copy_license(out_dir):
    """Ship the upstream license text with the export (license requirement).

    The cached base snapshot carries no LICENSE file, so fetch it from the Hub;
    if that fails, leave a notice file so the requirement stays visible.
    """
    dst = os.path.join(out_dir, "LICENSE")
    if os.path.exists(dst):
        return "LICENSE"
    snap = _snapshot_dir()
    for cand in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
        if snap and os.path.isfile(os.path.join(snap, cand)):
            shutil.copy2(os.path.join(snap, cand), dst)
            return "LICENSE"
    try:
        from huggingface_hub import hf_hub_download
        shutil.copy2(hf_hub_download(repo_id=STUDENT, filename="LICENSE"), dst)
        return "LICENSE"
    except Exception as e:  # noqa: BLE001 — offline / no upstream license file
        with open(dst + "-NOTICE.md", "w", encoding="utf-8") as fh:
            fh.write("# License\n\nThis model is a derivative of `%s` and remains "
                     "under the **%s**:\n\n%s\n\n(could not download the license "
                     "text automatically: %s)\n"
                     % (STUDENT, BASE_LICENSE_NAME, BASE_LICENSE_URL, e))
        return "LICENSE-NOTICE.md"


def _write_manifest(out_dir, rec):
    """export-manifest.json (machine-readable record of what was produced)."""
    with open(os.path.join(out_dir, "export-manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)


def merge(adapter_dir, out_dir, state):
    """Merge the LoRA adapter into base fp16 weights (CPU; GPU stays free)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    state.write("merging", stage="merge")
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.float16, trust_remote_code=True)  # CPU
    model = PeftModel.from_pretrained(base, adapter_dir)
    merged = model.merge_and_unload()
    state.write("saving", stage="save")
    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    tok.save_pretrained(out_dir)
    code = _copy_custom_code(out_dir)
    print("[export] merged + saved -> %s (custom code: %s)"
          % (out_dir, ", ".join(code) or "none found"), flush=True)
    return code


def to_gguf(out_dir, quants, state, imatrix=None):
    """Convert merged HF model -> f16 GGUF -> each requested quant.

    `quants` is a list of QUANT_TYPES keys; empty = leave GGUF alone. The f16
    conversion always runs (it is the source for every quantize pass) and is kept
    as the reference file. Returns the list of GGUF paths produced.
    """
    if not quants:
        return []
    if not os.path.isfile(CONVERT):
        raise SystemExit("convert script missing: %s" % CONVERT)
    state.write("converting", stage="gguf")
    base = os.path.basename(out_dir)
    f16 = os.path.join(out_dir, "%s-f16.gguf" % base)
    # Reuse an existing f16: converting 2.6B again costs minutes and the merged
    # weights are immutable once written (same run, same dir).
    if os.path.isfile(f16) and "f16" not in quants:
        print("[export] gguf f16: reusing %s" % f16, flush=True)
    else:
        cmd = [sys.executable, CONVERT, "--outfile", f16, "--outtype", "f16"]
        # --trust-remote-code exists on newer pins; probe --help so we don't
        # break on the pinned version if it lacks the flag.
        help_txt = subprocess.run([sys.executable, CONVERT, "--help"],
                                  capture_output=True, text=True).stdout
        if "--trust-remote-code" in help_txt:
            cmd.append("--trust-remote-code")
        cmd.append(out_dir)
        print("[export] gguf f16:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=_hf_env())
    files = [f16]
    for quant in quants:
        qtype = QUANT_TYPES.get(quant)
        if not qtype:
            continue  # "f16" — produced above
        if quant in IMATRIX_QUANTS and not imatrix:
            raise SystemExit(
                "%s needs an importance matrix — pass --imatrix <file> (build "
                "one with llama-imatrix; see docs/export.md)" % quant)
        qpath = os.path.join(out_dir, "%s-%s.gguf" % (base, quant))
        state.write("quantizing", stage="quantize", quant=quant)
        cmd = [QUANTIZE]
        if imatrix:
            cmd += ["--imatrix", imatrix]
        cmd += [f16, qpath, qtype]
        print("[export] quantize %s -> %s" % (qtype, qpath), flush=True)
        subprocess.run(cmd, check=True)
        files.append(qpath)
    return files


def _hub_token():
    """Resolve the HF token, independent of HF_HOME.

    The exporter sets HF_HOME to the G: cache for conversions, which would hide a
    token stored in the user's default cache (`~/.cache/huggingface/token`).
    Order: HF_TOKEN / HUGGING_FACE_HUB_TOKEN env, then the default user token
    file. Returns None when nothing is found (caller raises a clear error).
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        tok = (os.environ.get(var) or "").strip()
        if tok:
            return tok
    cands = [os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                          "token")]
    try:
        from huggingface_hub import constants
        cands.insert(0, constants.HF_TOKEN_PATH)
    except Exception:  # noqa: BLE001 — older/newer hub without that constant
        pass
    for path in cands:
        try:
            with open(path, encoding="utf-8") as fh:
                tok = fh.read().strip()
            if tok:
                return tok
        except OSError:
            continue
    return None


def push_hub(out_dir, repo, public, state, gguf_repo=None):
    """Upload the export to huggingface.co (private unless --public).

    HF-style layout: the transformers repo carries merged weights + tokenizer +
    card + license (GGUFs excluded), and the quants go to a sibling repo when
    `gguf_repo` is given.
    """
    from huggingface_hub import HfApi

    state.write("pushing", stage="hub", repo=repo, gguf_repo=gguf_repo)
    token = _hub_token()
    if not token:
        raise SystemExit(
            "no Hugging Face token found — set HF_TOKEN or run "
            "`venvs/ml/Scripts/hf.exe auth login` once (checked the default "
            "user cache, so HF_HOME does not matter)")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="model",
                    private=not public, exist_ok=True)
    api.upload_folder(folder_path=out_dir, repo_id=repo,
                      ignore_patterns=["*.gguf"],
                      commit_message="AETHER export (merged %s)" % STUDENT)
    url = "https://huggingface.co/%s" % repo
    print("[export] pushed -> %s (%s)"
          % (url, "public" if public else "private"), flush=True)

    ggufs = sorted(glob.glob(os.path.join(out_dir, "*.gguf")))
    if gguf_repo and ggufs:
        api.create_repo(repo_id=gguf_repo, repo_type="model",
                        private=not public, exist_ok=True)
        for extra in ("README.md", "LICENSE"):
            path = os.path.join(out_dir, extra)
            if os.path.isfile(path):
                api.upload_file(path_or_fileobj=path, path_in_repo=extra,
                                repo_id=gguf_repo,
                                commit_message="AETHER GGUF: %s" % extra)
        for path in ggufs:
            base = os.path.basename(path)
            print("[export]   upload %s -> %s" % (base, gguf_repo), flush=True)
            api.upload_file(path_or_fileobj=path, path_in_repo=base,
                            repo_id=gguf_repo,
                            commit_message="AETHER GGUF: %s" % base)
        print("[export] pushed GGUFs -> https://huggingface.co/%s" % gguf_repo,
              flush=True)
    return url


def main():
    ap = argparse.ArgumentParser(description="AETHER model export")
    ap.add_argument("--adapter", required=True,
                    help="adapter name under adapters/ or a path")
    ap.add_argument("--out", default=None,
                    help="merged model dir (default exports/<name>)")
    ap.add_argument("--name", default=None,
                    help="artifact name (default: adapter dir name) — drives the "
                         "export dir, GGUF file names, card and manifest")
    ap.add_argument("--title", default=None,
                    help="display title for the model card (default: --name with "
                         "dashes as spaces, e.g. jadis-1.0 -> Jadis 1.0)")
    ap.add_argument("--gguf", default=None,
                    help="comma-separated quant types, e.g. f16,q4_k_m,q3_k_m "
                         "(one of: %s)" % ", ".join(sorted(QUANT_TYPES)))
    ap.add_argument("--imatrix", default=None,
                    help="importance matrix for iq* quants (llama-imatrix output)")
    ap.add_argument("--hub", action="store_true", help="push merged model to HF")
    ap.add_argument("--push-only", action="store_true",
                    help="skip merge/GGUF/license and push an existing export dir "
                         "(retry a failed upload without re-merging); implies --hub")
    ap.add_argument("--gguf-only", action="store_true",
                    help="skip the merge and rebuild the GGUF set in an existing "
                         "export dir (e.g. after a chat-template change)")
    ap.add_argument("--repo", default=None,
                    help="HF repo id (default <your-username>/<name>, lowercased)")
    ap.add_argument("--gguf-repo", default=None,
                    help="repo for the GGUF files (default <owner>/<name>-GGUF)")
    ap.add_argument("--public", action="store_true",
                    help="make the HF repo public (default private)")
    args = ap.parse_args()

    adapter_dir = args.adapter
    if not os.path.isdir(adapter_dir):
        adapter_dir = os.path.join(ADAPTERS, args.adapter)
    if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        raise SystemExit("not an adapter (no adapter_config.json): %s"
                         % args.adapter)
    adapter_name = os.path.basename(os.path.normpath(adapter_dir))
    name = (args.name or adapter_name).strip()
    if not name or os.path.basename(name) != name:
        raise SystemExit("--name must be a bare slug (no separators): %r" % name)
    title = (args.title or name.replace("-", " ")).strip()
    quants = [q.strip() for q in (args.gguf or "").split(",") if q.strip()]
    bad = [q for q in quants if q not in QUANT_TYPES]
    if bad:
        raise SystemExit("unknown quant type(s): %s (one of: %s)"
                         % (", ".join(bad), ", ".join(sorted(QUANT_TYPES))))
    quants = list(dict.fromkeys(quants))  # de-dupe, keep the caller's order
    out_dir = args.out or os.path.join(EXPORTS, name)
    state = State(adapter_name, out_dir)

    try:
        if args.push_only:
            args.hub = True
            if not os.path.isdir(out_dir):
                raise SystemExit("--push-only: export dir missing: %s" % out_dir)
            gguf_files = sorted(glob.glob(os.path.join(out_dir, "*.gguf")))
            license_file = "LICENSE" \
                if os.path.isfile(os.path.join(out_dir, "LICENSE")) else None
            print("[export] push-only: reusing %s (%d gguf files)"
                  % (out_dir, len(gguf_files)), flush=True)
        else:
            if args.gguf_only:
                if not os.path.isdir(out_dir):
                    raise SystemExit("--gguf-only: export dir missing: %s" % out_dir)
                print("[export] gguf-only: rebuilding the GGUF set in %s" % out_dir,
                      flush=True)
            else:
                merge(adapter_dir, out_dir, state)
            gguf_files = to_gguf(out_dir, quants, state, imatrix=args.imatrix)
            license_file = _copy_license(out_dir)
        # Card + manifest BEFORE the push, so the Hub repo receives them too.
        repo = args.repo
        if args.hub and not repo:
            from huggingface_hub import HfApi
            try:
                repo = "%s/%s" % (HfApi(token=_hub_token()).whoami()["name"],
                                  name.lower())
            except Exception as e:  # noqa: BLE001 — not logged in
                raise SystemExit(
                    "cannot derive HF username (%s); set HF_TOKEN or run "
                    "`venvs/ml/Scripts/hf.exe auth login`, or pass "
                    "--repo owner/name" % e)
        gguf_repo = args.gguf_repo
        if args.hub and gguf_files and not gguf_repo:
            gguf_repo = "%s/%s-GGUF" % ((repo or "").split("/")[0], name)
        _write_card(out_dir, name, title, gguf_files, repo=repo,
                    gguf_repo=gguf_repo)
        manifest = {"name": name, "title": title, "base_model": STUDENT,
                    "adapter": adapter_name, "merged_utc": _now(),
                    "dtype": "fp16", "license": BASE_LICENSE_NAME,
                    "license_file": license_file, "gguf_repo": gguf_repo,
                    "gguf": [os.path.basename(f) for f in gguf_files],
                    "hub_url": None}
        _write_manifest(out_dir, manifest)
        if args.hub:
            manifest["hub_url"] = push_hub(out_dir, repo, args.public, state,
                                           gguf_repo=gguf_repo)
            _write_manifest(out_dir, manifest)
        state.write("done", name=name, title=title,
                    gguf=[os.path.basename(f) for f in gguf_files],
                    hub_url=manifest["hub_url"])
        print("[export] done -> %s" % out_dir, flush=True)
        return 0
    except (subprocess.CalledProcessError, SystemExit, Exception) as e:  # noqa: BLE001
        state.write("failed", error=str(e)[:500])
        print("[export] FAILED: %s" % e, file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

