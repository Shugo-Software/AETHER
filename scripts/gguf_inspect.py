"""GGUF metadata dumper (Phase 1 inventory + export checks).

Usage (gguf-py comes from a llama.cpp checkout, not from the venv):
  set AETHER_LLAMA_CPP_SRC=<path-to-llama.cpp>
  set PYTHONPATH=%AETHER_LLAMA_CPP_SRC%\gguf-py
  venvs\ml\Scripts\python.exe scripts\gguf_inspect.py <model.gguf>
"""
import sys
from gguf.gguf_reader import GGUFReader

f = sys.argv[1]
r = GGUFReader(f)
keys = list(r.fields.keys())


def _raw(name):
    """One GGUF field value (unwrap 1-element lists / bytes)."""
    if name not in r.fields:
        return None
    c = r.fields[name].contents()
    if isinstance(c, (list, tuple)) and len(c) == 1:
        c = c[0]
    if isinstance(c, bytes):
        c = c.decode("utf-8", "replace")
    return c


# Never hardcode the arch: metadata keys are prefixed "<arch>.*" and the file
# says which (principle #4 — read it, don't assume).
arch = _raw("general.architecture") or "qwen35"

def g(name):
    if name in r.fields:
        c = r.fields[name].contents()
        # unwrap single-element lists / bytes
        if isinstance(c, list):
            return c
        if isinstance(c, bytes):
            try:
                return c.decode("utf-8", "replace")
            except Exception:
                return c
        return c
    return None

meta = [
    "general.architecture", "general.name", "general.basename",
    "general.size_label", "general.finetune", "general.file_type",
    "general.quantization_version",
    f"{arch}.context_length", f"{arch}.embedding_length",
    f"{arch}.block_count", f"{arch}.feed_forward_length",
    f"{arch}.attention.head_count", f"{arch}.attention.head_count_kv",
    f"{arch}.attention.layer_norm_rms_epsilon", f"{arch}.rope.freq_base",
    f"{arch}.vocab_size",
    "tokenizer.ggml.model", "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
]
for m in meta:
    v = g(m)
    if isinstance(v, list) and len(v) == 1:
        v = v[0]
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    print(f"{m} = {v}")

# vision / mmproj indicator
print("has_vision_block_count:", any(k.startswith(f"{arch}.vision") for k in keys))
print("total_tensors:", len(r.tensor_infos) if hasattr(r, 'tensor_infos') else len(list(r.tensors)))
# rough param/byte estimate
tensors = list(getattr(r, "tensors", []))
tot_elems = 0
for t in tensors:
    ne = getattr(t, "n_elements", None)
    if ne is None:
        # gguf-py ReaderTensor exposes shape via .shape
        try:
            ne = 1
            for d in t.shape:
                ne *= d
        except Exception:
            ne = 0
    tot_elems += int(ne)
print("total_tensor_elements:", tot_elems)
