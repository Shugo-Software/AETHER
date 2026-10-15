# Model export (merge → GGUF → Hugging Face)

Turns a trained LoRA adapter into a **standalone model** anyone can load —
no PEFT, no base-model bookkeeping — plus optional GGUF for llama.cpp and an
optional push to huggingface.co.

## What it produces

| Artifact | Path | Loads with |
|---|---|---|
| Merged fp16 model (safetensors + tokenizer + chat template) | `exports/<name>/` | any `transformers` install with native LFM2 (≥5.2): `AutoModelForCausalLM.from_pretrained(path)` |
| f16 GGUF | `exports/<name>/<name>-f16.gguf` | llama.cpp (any build with `lfm2` support, pin `f9a95ae`) |
| Quantized GGUF (`f16` / `q8_0` / `q6_k` / `q5_k_s`·`q5_k_m` / `q4_k_s`·`q4_k_m` / `q3_k_s`·`q3_k_m` / `iq4_xs`) | `exports/<name>/<name>-<quant>.gguf` | llama.cpp — `q4_k_m` ≈ 1.56 GiB, `iq4_xs` ≈ 1.41 GiB (both fit a 3 GB GPU; see below) |
| Model card + export manifest | inside `exports/<name>/` | Hugging Face renders the card on push |

The merge uses `PeftModel.merge_and_unload()` on **CPU** (fp16) — it never
touches the GPU, so it can run while training occupies the 1060.

`merge()` also calls `_copy_custom_code()`, which copies `trust_remote_code`
files (`.py`, tokenizer assets) out of the cached base snapshot when the base
ships them. LFM2.5 does **not** — `model_type: lfm2` is native in transformers
5.2+, so the merged folder loads without any remote code (the log says
`custom code: none found`).

## From the dashboard (recommended)

**Export / push** card (under Training controls):

1. **adapter** — pick any adapter in `adapters/`.
2. **gguf** — `no` / `f16` / `q8_0` / `q6_k` / `q5_k_s` / `q5_k_m` / `q4_k_s` /
   `q4_k_m` / `q3_k_s` / `q3_k_m` / `iq4_xs` (the list comes from
   `export_model.QUANT_TYPES`). `iq4_xs` is marked *needs imatrix* in the UI — it
   requires `--imatrix`, so build it from the CLI.
3. **hub** — `no push` / `push private` (default on the server) / `push public`.
4. **repo** — leave blank for `<your-hf-username>/<adapter-dir>`; or `owner/name`.
5. **Export model** — status + log tail stream live; **Stop export** is
   PID-guarded (only kills a PID whose command line names `export_model.py`).

## From the CLI

```powershell
# one quant
venvs\ml\Scripts\python.exe scripts\export_model.py `
  --adapter jadis-1.0 --gguf q4_k_m

# a release: release name/title, several quants, imatrix, split repos on the Hub
venvs\ml\Scripts\python.exe scripts\export_model.py `
  --adapter jadis-1.0 --name jadis-1.0 --title "Jadis 1.0" `
  --gguf f16,q5_k_s,q4_k_m,q4_k_s,q3_k_m,q3_k_s,iq4_xs `
  --imatrix cache\quants\imatrix.gguf `
  --hub --repo ShugoSoftware/Jadis-1.0

# add:  --public   (default is private);  --gguf-repo owner/name to override
#       the default <owner>/<name>-GGUF for the quant files
```

With `--hub`, the export is split HF-style: the **model repo** gets the merged
weights, tokenizer, card, license and manifest (`*.gguf` excluded), and the
quants go to a sibling repo (`<owner>/<name>-GGUF` by default, which also gets a
copy of the card and license).

`--name` (artifact slug) and `--title` (card display name) let the **release
identity differ from the adapter directory** — the export dir, GGUF basenames,
card title and manifest all follow them. The name must be a bare slug.

The `--imatrix` file comes from llama-imatrix over a calibration text:

```powershell
builds\cuda\bin\llama-imatrix.exe -m exports\<name>\<name>-f16.gguf `
  -f cache\quants\calib.txt -o cache\quants\imatrix.gguf -ngl 99 -c 512 --chunks 48
```

Any text works as calibration; a few hundred KB of in-domain text (here: the 75
training rows, ~286 KB) is enough for the iq* quants.

## Verify an export

`scripts/verify_export.py` smoke-tests the artifacts independently — it is the
check to run before sharing anything:
```powershell
venvs\ml\Scripts\python.exe scripts\verify_export.py --model exports\lfm25-jadis-starcoder
```

1. **hf** — loads the merged folder with plain `transformers` (no PEFT, no base
   download), asserts the safetensors header is all `F16`, then generates.
2. **gguf** — loads the quantized GGUF with `builds\cuda\bin\llama-cli.exe`
   (`-st`, `-ngl 0` = CPU, so the teacher's GPU stays free) and generates.

Exit code is 0 only if every requested check passes; a JSON report lands in
`logs/verify-<name>.json`. Useful flags: `--skip-hf`, `--skip-gguf`,
`--gguf <path>`, `--n-predict`, `--ngl 99` (GPU offload), `--timeout`.

Last run of `lfm25-jadis-starcoder` (2026-10-15): **2/2 passed** — merged model
2.697 B params, all-`F16` header, loads in 6–18 s and generates 64 tokens in
30–48 s on CPU; Q4_K_M GGUF 1.6 GB, load + 64 tokens in ~9 s. Exact timings and
the raw completions land in `logs/verify-lfm25-jadis-starcoder.json`.

> **Reasoning prefill (observed):** the LFM2.5 chat template ends with
> `<|im_start|>assistant\n<think>`, so every generation opens *inside* a thinking
> block. The distilled completions carry no closing `</think>`, so the model was
> never trained to leave the block: at 256 new tokens
> `lfm25-jadis-starcoder` still produced analysis-style prose and never emitted
> `</think>`. Practical consequences: budget generous `max_new_tokens`, treat the
> answer as unmarked prose, and note llama.cpp displays the block as
> `[Start thinking]`. To make the student answer directly, fix it in the training
> data (`format_record` in `train_student.py`): append `</think>` before the
> completion so the model learns to close the block immediately.

## Running Jadis on the 3 GB GTX 1060 (and friends)

Measured, not estimated — full tables in `docs/benchmarks.md`, raw CSV in
`docs/benchmarks/three_gb_*.csv`. Rig: CUDA0 = RTX 4070 Ti, CUDA1 = 1060 6 GB,
**CUDA2 = 1060 3 GB** (nvidia-smi index 0 — the orderings disagree, so always
pass the llama.cpp device name, never a bare index).

| Quant | File size | Peak VRAM @4 k | Peak VRAM @16 k | tg t/s (CUDA) |
|---|---:|---:|---:|---:|
| `q3_k_s` | 1.18 GiB | 1387 MiB | 1625 MiB | ~30.7 |
| `q3_k_m` | 1.27 GiB | 1479 MiB | 1717 MiB | ~32.5 |
| `q4_k_s` | 1.49 GiB | 1705 MiB | 1943 MiB | ~40.5 |
| **`q4_k_m`** | **1.56 GiB** | **1773 MiB** | **2011 MiB** | **~40.6** |
| `q5_k_s` | 1.77 GiB | 1987 MiB | 2225 MiB | ~39.6 |

The card reports 3071 MiB total (~2965 free with the desktop idle), so the whole
range fits — `q4_k_m` at **16 k context** still leaves ~950 MiB of headroom. Use
the smallest quant you can tolerate for longer context or when other apps share
the card.

```powershell
# 4k context, fully offloaded to the 3 GB card
builds\cuda\bin\llama-cli.exe -m exports\jadis-1.0\jadis-1.0-q4_k_m.gguf `
  -dev CUDA2 -ngl 99 -c 4096 -cnv --jinja
```

Notes:

- **`-c` is the lever.** KV cost here is only ~16 KiB/token (8 of 30 layers are
  full attention with GQA 8×64), so context is cheap; the weights dominate.
- **Backend:** Vulkan generates faster on Pascal (~56–62 t/s vs ~40 on CUDA) but
  prompts slower (~300 vs ~470 t/s), and **Vulkan `q5_k_s` on this card faults the
  driver and has rebooted the machine** — see the constraint in `plan.md`. Prefer
  CUDA unless you specifically want Vulkan generation speed at `q4_k_m`/`q4_k_s`/
  `iq4_xs`.
- `-ngl` can be lowered (e.g. `-ngl 20`) to trade speed for headroom if the card
  is shared with a display or another process.
- `f16` (5.03 GiB) and `q8_0` (2.68 GiB) are **not** for this card; keep them as
  reference/CPU artifacts.

## Hugging Face auth (one-time, never via the web)

The exporter reads the token from `HF_TOKEN` (env var), else the stored token file
in the **user** cache (`~/.cache/huggingface/token`) — deliberately *not* from
`HF_HOME`, because the pipeline points `HF_HOME` at the G: cache, which holds no
token:

```powershell
venvs\ml\Scripts\hf.exe auth login      # hf.exe in the venv (no huggingface-cli)
```

The dashboard/API **never accepts a token** — the exporter process reads the
stored one. Pushes are **private** unless `--public` / "push public" is chosen.

## Release hygiene (ship a branded model, keep the lineage honest)

Every export writes a **branded card + the upstream license + a manifest**, and it
does so *before* any push so the Hub repo receives them:

| Item | What it gives you |
|---|---|
| `README.md` | Front-matter (`base_model`, `license_name`, tags), the release title, a lineage section, a method table built from the actual run record (rows/epochs/lr read from `logs/train-<name>.json`), a reasoning/"Thinking" section describing the `<think>` turn shape, runnable `transformers` + `llama.cpp` snippets, and a GGUF table with real sizes |
| `LICENSE` | The upstream license text, fetched from the base repo (or a `LICENSE-NOTICE.md` pointing at it if download fails) |
| `export-manifest.json` | `name`/`title`, `base_model`, source `adapter`, `merged_utc`, `dtype`, `license`, `gguf` file list, `gguf_repo`, `hub_url` |

Two separate ideas, keep them separate:

- **Brand name** — `--name`/`--title` (e.g. `jadis-1.0` → `Jadis 1.0`). This is
  what the release is called.
- **Lineage** — the card must still declare `base_model: LiquidAI/LFM2.5-2.6B`
  and the LFM Open License v1.0, because the license requires attribution for
  derivatives. A brand does not override a license.

Pushes default to **private**; `--public` (or "push public") is an explicit act.
Auth is never accepted from the web UI — the exporter reads `HF_TOKEN` or the
stored `hf auth login` token.

`scripts/eval_jadis.py` complements this: verify_export proves the *artifact*
loads and generates, while eval_jadis compares **base vs adapter** behaviour
(`logs/eval-<tag>.md`) so you can show the fine-tune changed the model's
behaviour rather than trusting the loss curve.

## Licensing (already in the generated card)

Jadis is a derivative of `LiquidAI/LFM2.5-2.6B` → **LFM Open License v1.0**
(derivatives explicitly permitted; commercial use under the $10M revenue
threshold — see `plan.md` open gates). The teacher (Qwen3.5-9B) shaped the
training data but its weights are not in the student.

## Plumbing

- Status: `logs/export-<adapter>.json` (stage: merging → saving → converting →
  quantizing → pushing → done/failed); active run: `logs/export-active.json`;
  log: `logs/export-<adapter>.log`. All shown in the dashboard snapshot
  (`/api/snapshot` → `export`).
- llama.cpp conversion uses the shared pinned source at
  `<LLAMA_CPP_SRC>` (**read-only** — never
  written by this pipeline) and quantizes with
  `builds\cuda\bin\llama-quantize.exe`.
- Safety: adapter must be a bare name under `adapters/`; single concurrent
  export; repo id must match `owner/name`.
- Env: `start_export` inherits the dashboard's environment and then overlays
  `train_control.TRAIN_ENV` (G: caches). Passing that dict **bare** fails the
  child with `WinError 10106` — without `SystemRoot`/`PATH`, Winsock cannot
  initialise, so the export dies ~20 s in before any stage is logged.
- Cost on this box (2.6B): merge ≈ 2.5 min, f16 GGUF write ≈ 40 s, then ≈ 20–90 s
  per extra quant type (`iq4_xs` is the slowest: ~90 s with an imatrix)
  → a 7-quant release is ~10 min locally, plus upload time.
- `--push-only` skips merge/GGUF and pushes an **existing** `exports/<name>/`
  (implying `--hub`), so a failed or interrupted upload is retried without
  re-merging the weights.
