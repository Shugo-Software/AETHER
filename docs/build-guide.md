# Build Guide (Phase 2) — llama.cpp Vulkan + CUDA

> **Status:** ✅ **Vulkan build COMPLETE and validated on all 3 GPUs.** CUDA build pending a
> CUDA-toolkit fix (see [CUDA status](#3-cuda-blocker--toolkit-incomplete-measured)).
> No drivers changed. The existing CPU+RPC install is untouched.
>
> Source: official `ggml-org/llama.cpp` @ **`b11565`** (`f9a95ae`), already checked out at
> `<LLAMA_CPP_SRC>`. Builds are **out-of-source** into
> `AETHER\builds\{vulkan,cuda}` and never modify that tree.

## 0. Result summary (measured)

| Build | Status | Devices validated | Smoke test (Qwen2.5-0.5B Q4_K_M) |
|---|---|---|---|
| **Vulkan** | ✅ Built `AETHER\builds\vulkan\bin\Release` | RTX 4070 Ti, GTX 1060, RX 580 (all 3) | ✅ valid output on all 3 (gen **31.2 / 18.3 / 18.2** t/s) |
| **CUDA** | ✅ Built `AETHER\builds\cuda\bin` | RTX 4070 Ti, GTX 1060 (both NVIDIA) | ✅ valid output on both (gen **327.1 / 101.5** t/s) |

`llama-cli --list-devices`:
- Vulkan build → `docs/raw/vulkan-llama-list-devices.txt`:
  ```
  Vulkan0: NVIDIA GeForce RTX 4070 Ti (12210 MiB)
  Vulkan1: NVIDIA GeForce GTX 1060 6GB (6245 MiB)
  Vulkan2: Radeon RX 580 Series (8192 MiB)
  ```
- CUDA build → `docs/raw/cuda-llama-list-devices.txt`:
  ```
  CUDA0: NVIDIA GeForce RTX 4070 Ti (12281 MiB)
  CUDA1: NVIDIA GeForce GTX 1060 6GB (6143 MiB)
  ```

> **Headline:** On the RTX 4070 Ti, CUDA is ~**10×** faster than Vulkan for this model
> (327 vs 31 t/s gen). Vulkan's unique value is that it is the **only** backend that reaches the
> AMD RX 580. Both satisfy the Phase 2 exit gate. llama.cpp **de-duplicated** the NVIDIA cards
> that `vulkaninfo` listed twice; device indices are read from `--list-devices` every run (#4).

---

## 1. Toolchain (verified present — no install needed)

| Tool | Path / Version | Used for |
|---|---|---|
| MSVC (cl) | VS 2022 BuildTools 14.44 (`vcvars64.bat`) | C/C++ compiler |
| CMake | 3.31.6 (VS 2022 BuildTools bundled) | Configure |
| Ninja | `G:\ort-cuda-build\bin\ninja.exe` (on disk) | Optional faster generator |
| CUDA nvcc | 12.4.131 | CUDA build — **but toolkit is incomplete, see §3** |
| Vulkan loader | 1.4.341 runtime + ICDs | Runtime only; **build needs the SDK, see §4** |

Generator default in scripts: **Visual Studio 17 2022 / x64** (guaranteed present). Ninja can be
substituted by putting `G:\ort-cuda-build\bin` on PATH.

---

## 2. Reproducible build scripts

| Script | Purpose |
|---|---|
| `scripts/build-cuda.ps1` | CUDA build (both NVIDIA GPUs: arch `89;61`) |
| `scripts/build-vulkan.ps1` | Vulkan build (all 3 GPUs) — requires Vulkan SDK |
| `scripts/gguf_inspect.py` | GGUF metadata dumper (used in Phase 1) |

### CUDA configure flags (validated — see `scripts/build-cuda.bat`)
```
-G Ninja
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_MAKE_PROGRAM=G:\ort-cuda-build\bin\ninja.exe
-DCMAKE_CUDA_ARCHITECTURES=89;61      # Ada (4070 Ti) + Pascal (1060)
-DCUDAToolkit_ROOT=<AETHER_ROOT>\toolchains\cuda-12.8.1
-DGGML_CUDA=ON
-DGGML_CUDA_FA=ON                      # FlashAttention (benefits cc 8.9)
-DGGML_BACKEND_DL=ON                   # load backends as DLLs (matches existing install)
-DGGML_CPU_ALL_VARIANTS=ON             # REQUIRED with BACKEND_DL (GGML_NATIVE is incompatible)
-DBUILD_SHARED_LIBS=ON
-DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON
-DLLAMA_CURL=OFF                       # keep local inference path cloud-free
```
> Notes (all measured): `-DGGML_NATIVE=ON` **fails** with `GGML_BACKEND_DL=ON` — use
> `GGML_CPU_ALL_VARIANTS=ON`. The **Ninja** generator is used because the Visual Studio generator
> fails with "No CUDA toolset found" (needs VS CUDA-integration files). `cl`/`rc`/`link`/`mt` must
> be on PATH (run from an environment where `vcvars64.bat` was `call`ed).

### Vulkan configure flags (planned)
```
-DGGML_VULKAN=ON
-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DBUILD_SHARED_LIBS=ON
-DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON
-DLLAMA_CURL=OFF
```
The ggml-vulkan backend additionally requires, at **configure** time:
`find_package(Vulkan COMPONENTS glslc REQUIRED)` and `find_package(SPIRV-Headers CONFIG REQUIRED)`.

---

## 3. CUDA — incomplete v12.4, resolved with a portable G:-only toolkit ✅

The pre-existing `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4` is **incomplete**:
it has `nvcc` + math libs but is **missing the core runtime** (`cuda_runtime.h`, `cudart.lib`,
`cudart64_*.dll`), so CMake configure fails with `CUDA Toolkit not found (missing: CUDA_CUDART)`.

Because **C: is nearly full** (a normal toolkit install to `C:\Program Files\NVIDIA GPU Computing
Toolkit` needs ~6–8 GB; C: had ~2 GB free — a first winget attempt filled C: to 0 bytes and was
rolled back), we built a **portable, complete CUDA 12.8.1 toolkit entirely on G:**:

1. Downloaded official `cuda_12.8.1_572.61_windows.exe` → `G:\tmp`.
2. Extracted its payload with 7-Zip (it is a self-extracting 7z archive).
3. Merged the per-component folders (`cuda_nvcc`, `cuda_cudart`, `libcublas`, …) into a standard
   toolkit layout at **`<AETHER_ROOT>\toolchains\cuda-12.8.1`**
   (`bin`/`include`/`lib\x64`/`nvvm`). Script: **`scripts/assemble-cuda-toolkit.ps1`** (~4.5 GB, G:).

CMake is pointed at it with `-DCUDAToolkit_ROOT="<AETHER_ROOT>\toolchains\cuda-12.8.1"`.

**Generator note:** the Visual Studio generator fails with "No CUDA toolset found" (it needs the
CUDA VS-integration files installed into the VS IDE). The **Ninja generator** drives `nvcc`
directly and needs no VS integration — that is what we use. Build script: **`scripts/build-cuda.bat`**.

**Runtime DLLs (important):** after building, the CUDA runtime DLLs (`cudart64_12.dll`,
`cublas*`, …) must sit next to `llama-cli.exe` (or be on PATH); otherwise `ggml-cuda.dll` silently
fails to load and `--list-devices` shows `(none)`. Copy `<toolkit>\bin\*.dll` → `builds\cuda\bin`.

**Rollback:** delete `toolchains\cuda-12.8.1` and `builds\cuda`. The broken v12.4 under
`C:\Program Files` was left exactly as found (never modified).

---

## 4. Vulkan blocker — SDK not installed (measured)

- `VULKAN_SDK` env var: unset.
- `glslc` / `glslangValidator`: not on PATH, not found on C: or G:.
- `SPIRV-HeadersConfig.cmake`: not found.
- The Vulkan **runtime** (loader 1.4.341 + all 3 ICDs) is present — that is enough to *run*
  Vulkan, but **not to build** ggml-vulkan, which needs the SDK's `glslc` + SPIRV-Headers.
- `winget search --id LunarG.VulkanSDK` returned "No package found" (that winget ID is not in
  the index). The SDK is normally installed from the LunarG installer (sdk.lunarg.com) or a
  community package. → **install** → need approval.

---

## Status — both builds DONE and validated ✅

- **Vulkan (primary, all 3 GPUs):** Installed `KhronosGroup.VulkanSDK` 1.4.363.0 via winget
  (MIT/Apache-2.0; provided `glslc` + `SPIRV-Headers`; set `VULKAN_SDK=C:\VulkanSDK\1.4.363.0`).
  Built `ggml-vulkan.dll` and validated generation on RTX 4070 Ti, GTX 1060, **and RX 580**.
- **CUDA (NVIDIA comparison):** Built a portable CUDA 12.8.1 toolkit on G: (§3), built
  `ggml-cuda.dll`, and validated generation on RTX 4070 Ti and GTX 1060. ~10× faster than Vulkan
  on the 4070 Ti (327 vs 31 t/s gen).

**Both satisfy the Phase 2 exit gate.** Phase 3 (single-GPU benchmarks) can begin on either/both.

**Preserved regardless:**
- Source pinned at `b11565`; existing CPU+RPC install untouched (verified: no `ggml-vulkan.dll`/`ggml-cuda.dll` added there).
- Vulkan SDK install did **not** touch GPU drivers (dev SDK only; the 1.4.x loader was already present).
- The broken `C:\...\CUDA\v12.4` was left exactly as found; the working toolkit lives on G:.
- Configure/build/smoke logs saved to `docs/raw/`.

## Rollback

To fully undo the Phase 2 work:
1. Delete `<AETHER_ROOT>\builds\vulkan` and `<AETHER_ROOT>\builds\cuda`.
2. Delete `<AETHER_ROOT>\toolchains\cuda-12.8.1` (portable CUDA toolkit).
3. Uninstall the SDK: `winget uninstall --id KhronosGroup.VulkanSDK`.
4. The original `Personality\runtime\llama.cpp` CPU+RPC install, the broken `C:\...\CUDA\v12.4`,
   and all models were never modified.
