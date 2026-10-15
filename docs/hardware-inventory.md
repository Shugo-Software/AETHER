# Hardware & Environment Inventory (Phase 1)

> **Status:** Complete. Inventory only — no packages installed, no drivers changed, no Cline
> settings modified, no models downloaded, no builds started, no training begun.
>
> **Captured:** 2026-10-13. All raw device-enumeration output is saved under `docs/raw/`
> (see [Raw evidence](#raw-evidence)). This file is the human-readable summary.

---

## 0. Executive summary

| Area | Finding | Impact |
|---|---|---|
| GPUs | **3× NVIDIA, all CUDA-capable** (updated 2026-10-14): RTX 4070 Ti (12 GB, cc 8.9), GTX 1060 6 GB (cc 6.1), GTX 1060 3 GB (cc 6.1). The AMD RX 580 was **removed** and its slot now holds the 1060 3 GB | Full device parity on CUDA — good for training-first; RX 580 kept aside as a later Vulkan-only inference experiment |
| Vulkan | Runtime **1.4.341** present; all 3 NVIDIA GPUs expose Vulkan | Vulkan build still works on all 3 (retained as a cross-check / future AMD path) |
| CUDA | Driver 581.57 (max CUDA 13.0); portable **CUDA 12.8.1** toolkit assembled on G: | CUDA build sees all 3 NVIDIA GPUs (CUDA0/1/2) |
| Existing llama.cpp | Official `b11565` (`f9a95ae`), **CPU + RPC only** — no Vulkan/CUDA backend compiled; `--list-devices` = `(none)` | Untouched; our new builds live separately under `AETHER\builds\` |
| Bonsai 27B | Present as **GGUF Q1_0** (`qwen35` arch) + a BF16 multimodal projector | Confirmed on disk; Q1_0 is an aggressive quant — GPU-backend support must be verified |
| Training stack | **torch 2.3.1+cpu (CPU-only)**, transformers 4.49, peft 0.21, accelerate 1.15; **no** trl/datasets/bitsandbytes | GPU training on the 4070 Ti is **currently blocked** until a CUDA torch wheel is installed (needs your approval) |
| Disk | **C: has only ~2.2 GB free**; G: has ~1.58 TB free | Builds/models/toolkits must live on **G:** or **F:**, never C: |
| Cline | Extension `saoudrizwan.claude-dev` installed and configured (cloud providers cached) | Provider profiles are isolated; a local profile can be added with clean rollback (Phase 5) |

**Bottom line:** The machine is now an **all-NVIDIA CUDA rig** (RTX 4070 Ti + 2× GTX 1060),
well-suited to the training-first plan. Design gotchas: (1) the existing llama.cpp has no GPU
backend (ours do, separately); (2) torch is CPU-only so no training starts until a CUDA torch is
installed; (3) **enumeration order is not stable** — the 1060 3 GB is `nvidia-smi` index 0 but
`CUDA2`/`Vulkan2` in llama.cpp, so always read real device IDs from `llama-cli --list-devices`.

> **Change log 2026-10-14:** AMD Radeon RX 580 removed; GTX 1060 3 GB installed in its PCI slot
> (`03:00.0`). Prior inventory (RX 580 / Vulkan-only cross-vendor) is superseded. Raw evidence in
> `docs/raw/` was re-captured for the new hardware.

---

## 1. Operating system

| Property | Value |
|---|---|
| OS | Microsoft Windows 11 Home |
| Version / Build | 10.0.22631 / **22631** |
| Architecture | 64-bit (x64) |
| Last boot | 2026-10-13 18:30:08 |

---

## 2. CPU & system RAM

| Component | Detail |
|---|---|
| CPU | AMD Ryzen 7 3700X, 8 cores / 16 threads, base 3.6 GHz |
| Motherboard/vendor | System manufacturer "System Product Name" (generic SMBIOS) |
| RAM | 2 × Corsair 16 GB DDR4-3200 (DIMM_A1, DIMM_A2) = **32 GB** (33.46 GB visible to OS) |

> The 32 GB RAM is the ceiling for host-side model buffers and any CPU/VRAM spill during
> heterogeneous inference. Ryzen 3700X is Zen 2 (no AVX-512); the existing llama.cpp ships a
> `ggml-cpu-zen4.dll` but will fall back to a compatible variant.

---

## 3. Storage

| Drive | Label | FS | Size | Free | Note |
|---|---|---|---|---|---|
| **C:** | — | NTFS | 476 GB | **2.1 GB** ⚠️ | Near-full. **Do not put builds or models here.** |
| D: | Blank | NTFS | 232 GB | 50 GB | |
| E: | System Reserved | NTFS | 0.1 GB | 0.1 GB | Boot partition |
| **F:** | Zero | NTFS | 1862 GB | 600 GB | Suitable for large models/builds |
| **G:** | My Book | exFAT | 3726 GB | **1592.8 GB** | **AETHER workspace lives here** — recommended for builds + models |
| I: | Death Records | NTFS | 465 GB | 248 GB | |

> G: is **exFAT**. exFAT is fine for large read-only model files, but it lacks POSIX symlinks
> and NTFS permissions. Git works but some tools that expect symlinks may not. If any build
> step needs symlinks, prefer F: (NTFS, 600 GB free).


---

## 4. GPUs — enumeration, IDs, VRAM, APIs

### 4.1 NVIDIA — all three GPUs (via `nvidia-smi`, authoritative; updated 2026-10-14)

| nvidia index | Model | VRAM | Compute cap | PCI bus | Driver | UUID |
|---|---|---|---|---|---|---|
| **0** | GeForce **GTX 1060 3GB** | 3072 MiB | **6.1** (Pascal) | 03:00.0 | 581.57 | GPU-<redacted-uuid> |
| **1** | GeForce **RTX 4070 Ti** | 12282 MiB | **8.9** (Ada) | 08:00.0 | 581.57 | GPU-<redacted-uuid> |
| **2** | GeForce **GTX 1060 6GB** | 6144 MiB | **6.1** (Pascal) | 09:00.0 | 581.57 | GPU-<redacted-uuid> |

- Driver max CUDA: **13.0**. Total VRAM across the rig: **21.5 GB**.
- The RTX 4070 Ti drives the display (1920×…), so a few hundred MiB are already resident.
- All three (cc 8.9 + 6.1) are supported by the assembled CUDA 12.8.1 toolkit (`arch 89;61`).
- The 1060 3 GB occupies the PCI slot the RX 580 used to be in (`…:0B` → now `03:00.0`).

### 4.2 AMD — removed

The **Radeon RX 580 was removed** on 2026-10-14 and replaced by the GTX 1060 3 GB. No AMD GPU is
currently installed. The Vulkan summary now lists only NVIDIA (vendorID 0x10de) devices. The
RX 580 is set aside for a later Vulkan-only inference experiment (per plan); when re-installed it
will be Vulkan-only (no ROCm/HIP, no `rocm-smi`).

Two other video controllers remain but are **virtual/non-compute** and must be ignored:
"Virtual Desktop Monitor" (status Error) and "Meta Virtual Monitor" — ROOT display devices.

### 4.3 Enumeration order is NOT stable — ⚠️ principle #4

The **same physical GPU has different indices in different tools**:

| Physical GPU | nvidia-smi index | llama.cpp CUDA | llama.cpp Vulkan |
|---|---|---|---|
| RTX 4070 Ti | 1 | **CUDA0** | Vulkan0 |
| GTX 1060 6GB | 2 | **CUDA1** | Vulkan1 |
| GTX 1060 3GB | **0** | **CUDA2** | Vulkan2 |

Note especially: the **1060 3 GB is nvidia-smi index 0 but CUDA2** — the two orderings disagree.
Therefore **device index ≠ physical GPU across tools**, and enumeration order must never be
assumed. Always read the real list from `llama-cli --list-devices` (or `llama-bench --list-devices`)
and match by **name/UUID** before selecting a device.

**Vulkan loader warnings present** (see `docs/raw/vulkaninfo-summary.txt`): NVIDIA Optimus/present
layers, OBS hook, Steam overlay/fossilize, Rockstar Social Club, EOS overlay. These can interfere
with compute-device visibility. If a card doesn't appear to llama.cpp, disabling overlay/hook
layers is the first thing to try — **not a driver change**.

---

## 5. Backend & runtime availability

### 5.1 Vulkan
- Loader: **1.4.341**. `vulkaninfo.exe` present (`C:\Windows\System32`).
- All three GPUs expose Vulkan. RTX 4070 Ti & GTX 1060 at 1.4.312 (NVIDIA); RX 580 at 1.3.217 (AMD).
- No extra SDK install is required for llama.cpp's Vulkan backend (it compiles shaders at build
  time via the bundled glslang; a runtime Vulkan loader + ICDs are already present).

### 5.2 CUDA
- Toolkit: **CUDA 12.4** at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`.
- `nvcc`: release 12.4, V12.4.131.
- Driver reports max supported CUDA 13.0 (driver is newer than the toolkit → fully compatible).

### 5.3 GPU monitoring utilities
| Tool | Status |
|---|---|
| `nvidia-smi` (+ `dmon`, `-q`, `--query`) | ✅ Present — covers both NVIDIA cards |
| `vulkaninfo` | ✅ Present |
| Windows Task Manager (GPU page) | ✅ Built-in — usable for the RX 580 |
| `rocm-smi` | ❌ Not installed (no AMD CLI monitoring) |
| GPU-Z / HWiNFO / MSI Afterburner | ❌ Not installed |

> For RX 580 VRAM/utilization during Phase 3/4, there is **no CLI monitor**. Options: Windows
> Task Manager "Performance → GPU" (video memory), or `nvidia-smi` for the NVIDIA pair only.
> Installing a monitoring tool would need your approval; not required for correctness.

---

## 6. Development toolchain

| Tool | Version / Path | Notes |
|---|---|---|
| git | `C:\Program Files\Git\cmd\git.exe` | ✅ Present |
| CMake | ❌ Not on PATH. **Bundled 3.31.6** at `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe` | Must add to PATH or invoke by full path in build scripts |
| MSVC (cl) | VS 2022 Community + VS 2022 BuildTools + VS 2019 BuildTools | ✅ Use VS 2022 x64 Native Tools |
| Clang / clang-cl | ❌ Not on PATH (the prebuilt llama.cpp used Clang 20.1.8, but it is not on PATH) | MSVC is the safe default |
| Ninja | ❌ Not on PATH | Optional; VS "Ninja" generator would need Ninja installed |
| Python (primary) | 3.10.6 (`…\AppData\Local\Programs\Python\Python310`) | Training env |
| Python (alt) | 3.13 (`<alt-python-tree>`) | Secondary |
| `uv` | Present (`…\Python310\Scripts\uv.exe`) | Fast env/package management available |

---

## 7. Existing llama.cpp installation (PRESERVE)

| Property | Value |
|---|---|
| Path | `<LLAMA_CPP_SRC>` |
| Source | Official `ggml-org/llama.cpp` |
| Git | tag **`b11565`**, commit **`f9a95ae`** (`f9a95aeb2096d0c0218e0cd6e6d901cd37be798a`), HEAD grafted |
| Version banner | `0.6.0-dev (build 11565, commit f9a95ae2)`, built with **Clang 20.1.8** for Windows x86_64 |
| Installed via | Prebuilt release asset `llama-b11565-bin-win-cpu-x64.zip` (see `fetch.json`) |
| Binaries | `llama-cli.exe`, `llama-server.exe`, `llama-bench.exe`, `llama-quantize.exe`, `llama-imatrix.exe`, `llama-perplexity.exe`, `ggml-rpc-server.exe`, `llama-tokenize.exe`, etc. |
| **Compiled backends** | **CPU only (all variants) + RPC**. Backends built as DLLs (`ggml-cpu-*.dll`, `ggml-rpc.dll`). **No `ggml-vulkan.dll`, no `ggml-cuda.dll`.** |
| Device check | `llama-cli --list-devices` → **`Available devices: (none)`** |

> **This is the pre-existing, working inference install (operating principle #2).** It is
> CPU+RPC only, so it cannot currently offload to any GPU. Phase 2 will create **new, separate
> build directories** (Vulkan and CUDA) and will **not** modify or overwrite this checkout.
> Note it uses `GGML_BACKEND_DL=ON` (dynamic backend loading) — relevant for how we configure
> the new builds.

A second, smaller llama.cpp-RPC tree exists at `<llama-rpc-tree>` (with a
`qwen2.5-0.5b-instruct-q4_k_m.gguf`). It is also to be preserved and is not modified here.

---

## 8. Model checkpoints on disk

### 8.1 Bonsai 27B (the primary model of interest)

| Property | Value |
|---|---|
| File | `%USERPROFILE%\.lmstudio\models\lmstudio-community\Bonsai-27B-GGUF\Bonsai-27B-Q1_0.gguf` |
| Size | **3.542 GB** |
| Last write | 2026-08-15 |
| **Architecture** | **`qwen35`** (Qwen3.5 family) |
| Basename | `prism-ml_Bonsai` |
| Size label / finetune | `27B` / `unpacked` |
| **Quantization** | `file_type = 40`, quantization_version 2, filename `Q1_0` (≈1.5–2 bits/weight, very aggressive) |
| Context length | **262144 (256K)** |
| embedding_length | 5120 |
| block_count (layers) | 64 |
| feed_forward_length | 17408 |
| attention heads | 24 query / **4 KV** (GQA) |
| rope freq base | 10,000,000 |
| Tokenizer | `gpt2` (BPE); BOS 248044, EOS 248046 |
| Vision projector | `mmproj-Bonsai-27B-BF16.gguf` (888 MB) — **multimodal** |
| llama.cpp support | ✅ `b11565` defines `LLM_ARCH_QWEN35` and the `prism.hadamard.*` KV keys this model uses |

**Format/licensing notes & flags:**
- **Format:** GGUF, already quantized to Q1_0. Because it is Q1_0 (an aggressive sub-2-bit
  quant), backend support is not guaranteed everywhere — the Vulkan and CUDA backends must be
  verified to actually load Q1_0 tensors in Phase 2, and single-GPU fit must be measured in
  Phase 3. If a backend rejects Q1_0, the fallback is to re-quantize with `llama-quantize`
  (e.g., Q4_K_M) from a higher-quality source — but no source weights were found on disk
  (see 8.4), so re-quantizing up is not currently possible locally.
- **Tokenizer/architecture:** supported by the checked-out revision. Good.
- **License:** the GGUF metadata does not embed an SPDX license string. The basename
  `prism-ml_Bonsai` implies an upstream Hugging Face origin. **The license and the teacher's
  terms for distillation/commercial use are NOT established from the local file and must be
  confirmed from the source model card before any Phase 6 dataset generation relies on it**
  (operating principle #6 + Phase 6 requirement). Flagged as an open item.

### 8.2 GLM-4.6V-Flash (candidate small instruct model, present)
- `GLM-4.6V-Flash-Q4_K_M.gguf` (5.88 GB), arch `glm4`, size_label 9.4B, gpt2 tokenizer, plus
  `mmproj-GLM-4.6V-Flash-F16.gguf` (1.7 GB). A viable **small, Q4_K_M, multimodal instruct
  model** for Phase 3 single-GPU benchmarks (fits comfortably per-card and is a standard quant
  the backends definitely support).

### 8.3 Qwen2.5-0.5B-Instruct (smallest, fully-supported test model)
- Safetensors: `<personality-project>\base_models\qwen2.5-0.5b-instruct\` (942 MB
  `model.safetensors` + tokenizer). License **Apache-2.0**.
- GGUF: `<llama-rpc-tree>\models\qwen2.5-0.5b-instruct-q4_k_m.gguf` (0.46 GB).
- Also cached in HF hub (`models--Qwen--Qwen2.5-0.5B-Instruct`). This is the natural
  **smoke-test model** for the Phase 2 "known-compatible GGUF must generate valid output" gate.

### 8.4 What is NOT on disk
- **No Bonsai 27B source/full-precision weights** (no `safetensors`, no higher quant). Only the
  single Q1_0 GGUF exists. Treat it as **read-only origin — do not overwrite (principle #8).**
- No other ≥13 GB LLM checkpoints anywhere on C:/D:/F:/G:/I: (the only >4 GB "model-like"
  files are game assets, ACE-Step audio checkpoints, and an Ubuntu ISO).

### 8.5 HF cache (`~\.cache\huggingface\hub`)
`ACE-Step/Ace-Step1.5`, `ACE-Step/acestep-5Hz-lm-0.6B`, `kemendev/russian-pop-lora`,
`Qwen/Qwen2.5-0.5B-Instruct`, `SamurAI-Official/NRR` (an ONNX rect model, 0.44 MB),
`sentence-transformers/all-MiniLM-L6-v2`, `tarn59/super_eurobeats_ACE_STEP-1.5-lora`.
None of these are the Bonsai model.


---

## 9. Training stack (Python 3.10.6) — current capability

| Package | Version | Note |
|---|---|---|
| **torch** | **2.3.1+cpu** | ⚠️ **CPU-only.** `torch.cuda.is_available() = False`, `device_count = 0` |
| transformers | 4.49.0 | ✅ |
| peft | 0.21.2 | ✅ LoRA/QLoRA adapters supported |
| accelerate | 1.15.0 | ✅ |
| safetensors | 0.5.2 | ✅ |
| gguf | 0.19.0 | ✅ (used above to read Bonsai metadata) |
| trl | ❌ not installed | Needed for DPO/preference optimization (Phase 8) |
| datasets | ❌ not installed | Needed for dataset loading (Phase 6) |
| bitsandbytes | ❌ not installed | Needed for QLoRA 4-bit (Windows support is limited) |
| vllm | ❌ not installed | — |

**Implication (Phase 7):** GPU training on the RTX 4070 Ti is **blocked** until a CUDA-enabled
torch wheel (e.g., `torch==2.x+cu121/cu124`) is installed. That is a **package install**, which
you asked me not to do yet. Rejection-SFT with PEFT/LoRA on the existing stack is possible on
**CPU** for a tiny smoke test, but any real student training requires CUDA torch + your approval.

**Related working reference:** the `Personality` project (`<personality-project>`) is a
mature SamurAI trainer that already does commit-pinned, weights-never-committed LoRA on
Qwen2.5-0.5B. Its `doctor` reports the LoRA backend ready. It is the best local template for the
Phase 6/7 dataset + adapter tooling and should be reused rather than reinvented.

---

## 10. Cline integration surface (PRESERVE — read-only inspection)

| Property | Value |
|---|---|
| Extension ID | `saoudrizwan.claude-dev` (Cline) |
| Global storage | `%USERPROFILE%\AppData\Roaming\Code\User\globalStorage\saoudrizwan.claude-dev` |
| MCP settings | `settings\cline_mcp_settings.json` → `{"mcpServers":{}}` (no MCP servers configured) |
| VS Code user settings | `%USERPROFILE%\AppData\Roaming\Code\User\settings.json` |
| State present | task history (`state\taskHistory.json`), 3 task folders, 1 git checkpoint, caches for OpenRouter/HiCap/VercAI gateway model catalogs |

**Rollback surface:** ~~Cline's API provider profiles and API keys are stored in VS Code
SecretStorage~~ — **corrected 2026-10-14 after direct inspection of Cline 4.1.24:**
profiles live in a plain file `%USERPROFILE%\.cline\data\settings\providers.json`
(OAuth tokens only in `secrets.json`). Phase 5 executed the safe pattern: added ONE new
`openai-compatible` profile (`qwen35-teacher` → `http://127.0.0.1:8080/v1`), all 5
pre-existing profiles verified byte-identical, `lastUsedProvider` untouched. Full
storage map, schema, verification and rollback = **`docs/cline-integration.md`**.

---

## 11. SamurAI ecosystem context (what exists around AETHER)

- `<personality-project>` — persona trainer (ShugoCore): commit-pinned model fetches,
  LoRA training, GGUF export, MIT repo with weights-never-committed discipline. Reusable pattern.
- `G:\Program Prototype\shugocore` — consumer runtime that talks to OpenAI-compatible endpoints,
  references llama.cpp server + RPC mesh (`mesh_rpc.py`), enforces timeout/size budgets per tier.
- Both use the same **hash-pinned fetch + reproduce-don't-inherit** philosophy this roadmap
  asks for. AETHER should follow it.


---

## 12. Recommended build configuration (PROPOSAL — for your review, not yet executed)

Aligned with Phase 2 and operating principles (no backend conflicts, separate dirs, documented
options). All from the **official ggml-org/llama.cpp** source already checked out at `b11565`.

**Where:** under the AETHER workspace on **G:** (e.g., `AETHER\builds\vulkan` and
`AETHER\builds\cuda`), **never on C:**.

**Toolchain to use:** VS 2022 x64 Native Tools Command Prompt; add the bundled CMake 3.31.6 to
PATH (`…\VS2022\BuildTools\…\CMake\CMake\bin`). Compiler = MSVC (cl). Keep `b11565` as the
source revision for reproducibility (or bump to a newer tag only with a documented reason).

### Build A — Vulkan (primary inference path)
Purpose: cross-vendor inference on all three GPUs.
- `-DGGML_VULKAN=ON`
- `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON` (match existing install style; lets us load
  backends as DLLs and avoids static backend conflicts)
- `-DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON`
- `-DLLAMA_CURL=OFF` (keep the local path cloud-free per principle #9)
- Verify with `llama-cli --list-devices` (expect RTX 4070 Ti, GTX 1060, RX 580 — watch the
  duplication quirk) and a generation smoke test with the Qwen2.5-0.5B GGUF.

### Build B — CUDA (NVIDIA comparison path)
Purpose: NVIDIA-only comparison on 4070 Ti + 1060.
- `-DGGML_CUDA=ON`
- `-DCMAKE_CUDA_ARCHITECTURES=89;61` (build for Ada 8.9 and Pascal 6.1 in one binary)
- `-DGGML_CUDA_FA=ON` (FlashAttention, benefits cc 8.9) — keep `GGML_CUDA_FORCE_CUBLAS` off
- `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON`, `-DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON`
- Verify both NVIDIA GPUs enumerate/selectable via `--list-devices` + a generation smoke test.

### Keep separate
Vulkan and CUDA in **different build directories** (separate CMake caches) so backend
configuration cannot conflict — exactly as the roadmap requires. Do **not** touch the existing
`Personality\runtime\llama.cpp` tree.

### Phase 2 exit gate (must pass before Phase 3)
1. Both builds configure + compile cleanly; backends listed.
2. `llama-cli --list-devices` shows the expected GPUs (by name/UUID, not assumed index).
3. A known-compatible GGUF (Qwen2.5-0.5B Q4_K_M) produces **valid, coherent output** on each
   backend. Only then proceed to benchmarks.

### Open items requiring your decision (not doing yet)
1. **CUDA torch install** to enable training on the 4070 Ti (package install — needs approval).
2. **Bonsai 27B license/teacher terms** confirmation from the upstream model card before Phase 6.
3. Whether to install an **AMD GPU CLI monitor** (optional; Task Manager is a stopgap).
4. Optional: bump source revision above `b11565` (only if a newer backend fixes a Phase 2/3 issue).

---

## Raw evidence

All commands below were re-run and their output saved to `docs/raw/`:

| File | Source command |
|---|---|
| `nvidia-smi-query-gpu.csv` | `nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap,pci.bus_id,uuid --format=csv` |
| `nvidia-smi-q.txt` | `nvidia-smi -q` |
| `os-win32-operatingsystem.txt` | `Get-CimInstance Win32_OperatingSystem \| Format-List *` |
| `os-win32-videocontroller.txt` | `Get-CimInstance Win32_VideoController \| Format-List *` |
| `os-win32-processor.txt` | `Get-CimInstance Win32_Processor \| Format-List *` |
| `os-win32-physicalmemory.txt` | `Get-CimInstance Win32_PhysicalMemory \| Format-List *` |
| `os-get-volume.txt` | `Get-Volume \| Format-List *` |
| `vulkaninfo-summary.txt` | `vulkaninfo --summary` |
| `nvcc-version.txt` | `nvcc --version` (CUDA 12.4) |
| `cmake-version.txt` | `cmake --version` (VS 2022 BuildTools bundled 3.31.6) |
| `llama-cli-version.txt` | `llama-cli --version` (b11565) |
| `llama-cli-list-devices.txt` | `llama-cli --list-devices` (`(none)` — CPU+RPC build) |
| `llama.cpp-git-rev.txt` | `git -C …\llama.cpp log -1 --oneline` |
| `bonsai-27b-gguf-meta.txt` | GGUF metadata dump of `Bonsai-27B-Q1_0.gguf` |
| `toolchain-paths.txt` | tool presence scan (git/cmake/python/…) |
| `python-training-stack.txt` | torch/transformers/peft/accelerate versions + CUDA availability |

The GGUF inspection helper used to produce the Bonsai metadata is kept at
`scripts/gguf_inspect.py` so the dump is reproducible.

| vulkaninfo / nvcc / nvidia-smi | Present (see §5) | |
