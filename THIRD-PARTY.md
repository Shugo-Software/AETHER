# Third-party components, data and licenses

This file is part of `LICENSE` and records the upstream material the AETHER
pipeline is built on. Their terms continue to apply and are **not** overridden
by our license (`LICENSE`). Manifests beside each derived artefact record the
same facts per dataset.

## Base model (redistributed — our models are derivatives)

| Component | License | How it is used | Obligation |
|---|---|---|---|
| [`LiquidAI/LFM2.5-2.6B`](https://huggingface.co/LiquidAI/LFM2.5-2.6B) | **LFM Open License v1.0** (HF tag `license:other`) | Base model for every trained student. The Jadis line (adapters, merged checkpoints, GGUF quantisations) are **derivative works**: architecture, tokenizer and chat template are unchanged. | Attribution + the upstream license text ships with every export (`LICENSE-UPSTREAM-LFM.md`); commercial use is subject to the upstream revenue thresholds; derivatives may carry their own additional license (ours adds non-commercial terms). |

## Teacher model (not redistributed)

| Component | License | How it is used |
|---|---|---|
| [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | **Apache License 2.0** (Copyright 2026 Alibaba Cloud) | Local distillation teacher: it *generates training text* and grades student answers. **None of its weights are present in any released model.** |

## Datasets (used to build corpora; corpora are not redistributed)

| Dataset | License (as declared) | How it is used |
|---|---|---|
| [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | MIT | Ingested via `scripts/hf_datasets.py`; 2,000 rows converted into a training corpus. |
| [`MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x`](https://huggingface.co/datasets/MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x) | Apache-2.0 | Pulled in full (15,000 rows) and measured; **rejected as SFT data and as rewrite input** (`docs/dataset-generation.md`). No rows were used for training. |
| [`FineEnvs/openai-math`](https://huggingface.co/datasets/FineEnvs/openai-math) | Apache-2.0 | **Evaluation only** (`datasets/eval/`, unreachable from the training control plane). |
| [`bigcode/starcoderdata`](https://huggingface.co/datasets/bigcode/starcoderdata) | Derived from *The Stack*, permissively-licensed source files only | Source of **prompts** for the teacher-generated corpora (`docs/dataset-generation.md`). All completions are teacher-written, not copied from the dataset. |

Corpora and their manifests are gitignored (`.gitignore`: `datasets/*.jsonl`) and
are not published; each manifest names its source repo, revision and parameters.

Teacher-written corpora and rewrite lessons are our own material and are covered
by `LICENSE`. Where a corpus is *derived from* the datasets above, those
datasets' terms continue to apply to the derived data.

## Toolchains and runtime dependencies (referenced, not redistributed)

| Component | License |
|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (all builds, GGUF tools) | MIT |
| [Vulkan SDK](https://vulkan.lunarg.com/) (headers + `glslc`) | Apache-2.0 / MIT |
| CUDA Toolkit | NVIDIA Software License Agreement (not redistributed) |
| PyTorch | BSD-3-Clause |
| `transformers`, `peft`, `datasets`, `accelerate`, `huggingface_hub` | Apache-2.0 |
| `bitsandbytes`, `triton` | MIT |

See `requirements.txt` for the pinned versions actually used.
