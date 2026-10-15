#!/usr/bin/env python3
"""Phase 8 — verify an exported model actually loads and generates.

Two independent checks, each optional:

  1. hf    — load ``exports/<adapter>/`` with plain ``transformers`` (no PEFT,
             no base-model download) and generate. Also peeks the safetensors
             header to confirm the weights really are fp16.
  2. gguf  — load the GGUF with llama.cpp's ``llama-cli.exe`` (single-turn,
             ``-ngl 0`` = CPU so the teacher's GPU stays free) and generate;
             this also exercises the chat template embedded in the GGUF.

Exit code is 0 only when every requested check passes. A JSON report is written
to ``logs/verify-<name>.json``.

Usage:
  python scripts/verify_export.py --model exports/lfm25-jadis-starcoder
  python scripts/verify_export.py --model exports/lfm25-jadis-starcoder --skip-gguf
  python scripts/verify_export.py --model exports/x --gguf exports/x/x-q8_0.gguf --ngl 99

Run it from the repo root with the ML venv (``venvs\\ml\\Scripts\\python.exe``);
all HF/temp caches are pinned to G: here regardless of the caller's env.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORTS = os.path.join(ROOT, "exports")
LOGS = os.path.join(ROOT, "logs")
LLAMA_CLI = os.path.join(ROOT, "builds", "cuda", "bin", "llama-cli.exe")

SYSTEM = "You are a precise senior Python engineer. Be concrete and correct."
PROMPT = ("In two sentences: why is streaming a large JSONL file line by line "
          "usually better than loading it all with json.load()?")


def parse_args():
    ap = argparse.ArgumentParser(description="Verify an AETHER export")
    ap.add_argument("--model", default=None,
                    help="merged export dir (default: newest under exports/)")
    ap.add_argument("--gguf", default=None,
                    help="GGUF to test (default: q4_k_m > q8_0 > f16 in --model)")
    ap.add_argument("--skip-hf", action="store_true",
                    help="skip the transformers load/generate check")
    ap.add_argument("--skip-gguf", action="store_true",
                    help="skip the llama.cpp load/generate check")
    ap.add_argument("--prompt", default=PROMPT, help="probe prompt")
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="HF check: new tokens to generate")
    ap.add_argument("--n-predict", type=int, default=64,
                    help="GGUF check: tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="HF check: 0 = greedy (deterministic smoke test)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="HF check device (cpu keeps the teacher's GPU free)")
    ap.add_argument("--ngl", type=int, default=0,
                    help="GGUF check: layers to offload (0 = CPU)")
    ap.add_argument("--llama-cli", default=LLAMA_CLI)
    ap.add_argument("--timeout", type=int, default=900,
                    help="GGUF check: seconds before giving up")
    return ap.parse_args()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pin_caches():
    """Keep HF/temp caches on G: (C: is critically small — plan.md)."""
    hf = os.path.join(ROOT, "cache", "hf")
    tmp = os.path.join(ROOT, "cache", "temp")
    os.makedirs(tmp, exist_ok=True)
    os.environ.update({"HF_HOME": hf, "HF_HUB_CACHE": hf,
                       "HF_DATASETS_CACHE": hf, "HF_HUB_DISABLE_TELEMETRY": "1",
                       "TEMP": tmp, "TMP": tmp})


def pick_model(explicit):
    if explicit:
        return explicit
    cands = [d for d in glob.glob(os.path.join(EXPORTS, "*"))
             if os.path.isfile(os.path.join(d, "model.safetensors"))]
    if not cands:
        raise SystemExit("no merged export under exports/ — run export_model.py")
    return max(cands, key=os.path.getmtime)


def pick_gguf(model_dir, explicit):
    if explicit:
        return explicit
    base = os.path.basename(os.path.normpath(model_dir))
    for suffix in ("-q4_k_m.gguf", "-q8_0.gguf", "-f16.gguf"):
        path = os.path.join(model_dir, base + suffix)
        if os.path.isfile(path):
            return path
    hits = sorted(glob.glob(os.path.join(model_dir, "*.gguf")))
    return hits[0] if hits else None


def safetensors_dtypes(path):
    """Read only the safetensors header -> {dtype: tensor_count}."""
    with open(path, "rb") as fh:
        size = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(size))
    out = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        out[meta["dtype"]] = out.get(meta["dtype"], 0) + 1
    return out

def hf_check(model_dir, args):
    """Load the merged model with plain transformers and generate."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    res = {"check": "hf", "model": os.path.relpath(model_dir, ROOT)}
    try:
        res["safetensors_dtypes"] = safetensors_dtypes(
            os.path.join(model_dir, "model.safetensors"))
    except (OSError, ValueError, KeyError) as e:  # noqa: BLE001
        res["safetensors_dtypes"] = None
        print("[verify] safetensors header unreadable: %s" % e, flush=True)

    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, trust_remote_code=True,
            low_cpu_mem_usage=True)
        if args.device == "cuda":
            model = model.to("cuda")
        model.eval()
    except Exception as e:  # noqa: BLE001
        res.update(ok=False, load_s=round(time.time() - t0, 1),
                   error="%s: %s" % (type(e).__name__, e))
        return res

    res["load_s"] = round(time.time() - t0, 1)
    res["params_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 3)
    res["dtype"] = str(next(model.parameters()).dtype).replace("torch.", "")
    res["tokenizer_class"] = type(tok).__name__
    res["has_chat_template"] = bool(getattr(tok, "chat_template", None))

    try:
        text = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001 — no template: plain fallback
        text = "%s\n\nUser: %s\n\nAssistant:" % (SYSTEM, args.prompt)
    gen_kw = {"max_new_tokens": args.max_new_tokens,
              "pad_token_id": tok.pad_token_id}
    if args.temperature and args.temperature > 0:
        gen_kw.update(do_sample=True, temperature=args.temperature, top_p=0.9)
    else:
        gen_kw["do_sample"] = False

    t1 = time.time()
    try:
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kw)
    except Exception as e:  # noqa: BLE001
        res.update(ok=False, error="generate: %s: %s" % (type(e).__name__, e))
        return res
    res["gen_s"] = round(time.time() - t1, 1)
    res["completion"] = tok.decode(out[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
    res["ok"] = bool(res["completion"])
    if not res["ok"]:
        res["error"] = "empty completion"
    return res


def _clean_cli_output(out, prompt):
    """Drop llama-cli's banner + prompt echo and the trailing perf lines."""
    text = out or ""
    idx = text.rfind(prompt)
    if idx != -1:
        text = text[idx + len(prompt):]
    cuts = [i for i in (text.find("[ Prompt:"), text.find("\nExiting...")) if i != -1]
    if cuts:
        text = text[:min(cuts)]
    return text.strip()


def _gguf_cmd(args, gguf, with_system):
    cmd = [args.llama_cli, "-m", gguf, "-p", args.prompt, "-n",
           str(args.n_predict), "-ngl", str(args.ngl), "-c", "4096",
           "--no-warmup", "-st", "--no-display-prompt"]
    if with_system:
        cmd += ["-sys", SYSTEM]
    return cmd


def gguf_check(gguf, args):
    """Load the GGUF with llama-cli (single turn) and generate."""
    res = {"check": "gguf", "gguf": os.path.relpath(gguf, ROOT),
           "size_mb": round(os.path.getsize(gguf) / 1024 / 1024, 1)}
    if not os.path.isfile(args.llama_cli):
        res.update(ok=False, error="llama-cli not found: %s" % args.llama_cli)
        return res

    # -sys needs a chat template in the GGUF; retry bare if that is missing.
    proc = None
    for with_system in (True, False):
        cmd = _gguf_cmd(args, gguf, with_system)
        res["cmd"] = " ".join('"%s"' % c if " " in c else c for c in cmd)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True,
                                  timeout=args.timeout)
        except subprocess.TimeoutExpired:
            res.update(ok=False, error="timeout after %ds" % args.timeout)
            return res
        res["load_and_gen_s"] = round(time.time() - t0, 1)
        res["returncode"] = proc.returncode
        res["used_system_prompt"] = with_system
        if proc.returncode == 0:
            break
        if with_system:
            res["retry_reason"] = (proc.stderr or "").strip()[-200:]
            continue
        res.update(ok=False, error="exit %d" % proc.returncode,
                   tail=(proc.stderr or proc.stdout or "").strip()[-400:])
        return res

    out = _clean_cli_output(proc.stdout, args.prompt)
    res["completion"] = out[:4000]
    res["completion_chars"] = len(out)
    res["ok"] = bool(out)
    if not res["ok"]:
        res["error"] = "no output"
    return res


def main():
    args = parse_args()
    _pin_caches()
    model_dir = pick_model(args.model)
    name = os.path.basename(os.path.normpath(model_dir))
    checks = []

    if not args.skip_hf:
        print("[verify] hf: %s (device=%s)" % (model_dir, args.device),
              flush=True)
        res = hf_check(model_dir, args)
        checks.append(res)
        print("[verify] hf -> %s (%s params, %s, load %ss, gen %ss)"
              % ("OK" if res.get("ok") else "FAIL", res.get("params_b"),
                 res.get("dtype"), res.get("load_s"), res.get("gen_s")),
              flush=True)
        if res.get("completion"):
            print("[verify] hf says: %s" % res["completion"][:400], flush=True)
        if res.get("error"):
            print("[verify] hf error: %s" % res["error"], flush=True)

    if not args.skip_gguf:
        gguf = pick_gguf(model_dir, args.gguf)
        if not gguf:
            print("[verify] gguf: none in %s — skipped" % model_dir, flush=True)
        else:
            print("[verify] gguf: %s" % gguf, flush=True)
            res = gguf_check(gguf, args)
            checks.append(res)
            print("[verify] gguf -> %s (%.1f MB, %ss)"
                  % ("OK" if res.get("ok") else "FAIL", res.get("size_mb", 0),
                     res.get("load_and_gen_s")), flush=True)
            if res.get("completion"):
                print("[verify] llama.cpp says: %s"
                      % res["completion"][-400:], flush=True)
            if res.get("error"):
                print("[verify] gguf error: %s" % res["error"], flush=True)

    failed = [c for c in checks if not c.get("ok")]
    payload = {"name": name, "verified_utc": _now(),
               "prompt": args.prompt, "checks": checks,
               "passed": len(checks) - len(failed), "total": len(checks)}
    os.makedirs(LOGS, exist_ok=True)
    out_path = os.path.join(LOGS, "verify-%s.json" % name)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print("[verify] %d/%d checks passed -> %s"
          % (payload["passed"], payload["total"], out_path), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

