# Cline Integration (Phase 5) — storage surface, local profile, rollback

> **Verified 2026-10-14** against Cline `saoudrizwan.claude-dev` **4.1.24** by direct
> inspection of the extension bundle and live files (principle #6 — nothing assumed).
> This doc fulfills the Phase-1 commitment in `hardware-inventory.md` §10.

## 1. Where Cline 4.1.24 actually stores provider profiles

The Phase-1 inventory assumed VS Code SecretStorage. **Correction:** profiles live in a
plain JSON file inside Cline's own data dir:

| Item | Path / value |
|---|---|
| **Provider profiles (the thing we add to)** | `%USERPROFILE%\.cline\data\settings\providers.json` |
| Backing secrets (OAuth tokens only) | `%USERPROFILE%\.cline\data\secrets.json` |
| Other Cline state | `%USERPROFILE%\.cline\data\globalState.json`, `models.json`, `global-settings.json` |
| VS Code global state (NOT used for profiles) | `...\Code\User\globalStorage\state.vscdb` — key `saoudrizwan.claude-dev` holds only rollout flags |
| Env overrides | `CLINE_PROVIDER_SETTINGS_PATH` (file), `CLINE_DATA_DIR`, `CLINE_DIR` |
| Write pattern | atomic tmp + rename; `chmod 0600`; OAuth lock files appear alongside during token refresh |
| Schema | `{version:1, lastUsedProvider, modes, providers:{<name>:{settings:{provider, model, baseUrl?, auth?, reasoning?}, updatedAt, tokenSource}}}` |

`openai-compatible` profile facts (from extension/webview bundles):
- Provider enum value: `"openai-compatible"`; UI label: **"OpenAI Compatible"**.
- Base URL field: `settings.baseUrl` (llama-server → `http://127.0.0.1:8080/v1`).
- API key is **optional** for this provider (`apiKeyOptional:true` in its registry);
  when set it lives in secrets storage with an `apiKeyLength` mask in the profile —
  llama-server is keyless, so we store no key at all.
- `tokenSource` values observed: `oauth` (cloud profiles), `manual` (local profiles).

## 2. Profiles present (after Phase 5)

| Profile | Provider | Model | Notes |
|---|---|---|---|
| `cline` | cline | cline-free/mimo-v2.6-flash | **untouched** — was & remains `lastUsedProvider` |
| `cline-pass` | cline-pass | cline-free/step-5-preview | untouched |
| `lmstudio` | lmstudio | prism-ml/bonsai-27b | untouched (historical — Bonsai retired) |
| `ollama` | ollama | gemma4:e4b | untouched |
| `sapaicore` | sapaicore | anthropic--claude-3.5-sonnet | untouched |
| **`qwen35-teacher`** | **openai-compatible** | **qwen3.5-9b-teacher** | **ADDED by Phase 5 → http://127.0.0.1:8080/v1** |

Verification performed at write time: all 5 pre-existing profiles deep-equal before/after;
top-level fields (including `lastUsedProvider`) unchanged; entry survived >30 s with Cline
running (no clobber).

## 3. The endpoint behind the profile

| Item | Value |
|---|---|
| Start | `powershell -File scripts\start-teacher-server.ps1` |
| Stop / status | `powershell -File scripts\stop-teacher-server.ps1 [-StatusOnly]` |
| Server | `builds\cuda\bin\llama-server.exe`, **pinned `--device CUDA0`** (4070 Ti), `-ngl 99`, ctx 8192, `--jinja`, loopback-only |
| Model | `models\Qwen3.5-9B-Q4_K_M.gguf`, alias `qwen3.5-9b-teacher` |
| Health | `GET http://127.0.0.1:8080/health` → 200 |
| Logs | `logs\teacher-server.log` (+`.err`), on G: |
| Measured | chat completion OK; ~67 t/s generation; `reasoning_content` exposed (Qwen3.5 thinking model) |

**Manual smoke test (proven 2026-10-14):**
```powershell
$body = @{model='qwen3.5-9b-teacher'; messages=@(@{role='user';content='Say exactly: endpoint ready'});
          max_tokens=512; temperature=0} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post `
  -ContentType application/json -Body $body   # -> "endpoint ready"
```

## 4. Using it in Cline (UI steps)

1. Start the endpoint: `powershell -File <AETHER_ROOT>\scripts\start-teacher-server.ps1`
   (waits for health itself; run in any terminal — or add to Startup).
2. In Cline's sidebar → gear/settings → **API Provider** → select **`qwen35-teacher`**
   (labeled "OpenAI Compatible"). The Model ID field should show `qwen3.5-9b-teacher`.
3. Send a test prompt. Expected: replies served by the 4070 Ti (check
   `nvidia-smi` utilization on GPU0 while it answers).
4. Switching back = select any other profile; **nothing else changes** — profiles are
   isolated keys in one file; `lastUsedProvider` only records the last selection.

## 5. Rollback (any of these, in order of preference)

1. **Delete the single profile:** in Cline UI (remove provider), or edit
   `providers.json` → remove `providers["qwen35-teacher"]` → save.
2. **Restore the pre-change backup:** `providers.json.bak-20261014T152517Z`
   (sha256-prefixed verification `2d714657b0b4798e`) copied over the file.
3. Server-side rollback: `stop-teacher-server.ps1` — the server holds no persistent state.

## 6. Known caveats

- Cline re-reads `providers.json` on demand; if a stale in-memory copy is ever observed
  after external edits, do a VS Code **Reload Window** (Cline re-reads at activation).
- The endpoint binds **127.0.0.1 only** — never expose it on LAN without auth (llama-server
  has none by default; use `--api-key` if it must leave loopback).
- Do not edit the file while an OAuth lock file (`providers.json.oauth-*.lock`) is present —
  wait for it to clear.
- Ollama (port 11434) and other user services run independently — our scripts refuse to
  bind taken ports and refuse to kill non-`llama-server` PIDs.
