<#
.SYNOPSIS
    Start the SamurAI TEACHER endpoint (Phase 5): Qwen3.5-9B Q4_K_M via llama-server,
    pinned to CUDA0 (RTX 4070 Ti), OpenAI-compatible on localhost:8080.

.DESCRIPTION
    Part of AETHER Phase 5 (Cline local provider). Uses the CUDA build (builds\cuda);
    runtime DLLs live next to the exe (principle: ggml-cuda.dll silently fails otherwise).
    Device ID is passed explicitly (--device CUDA0) per principle #4 — do not assume.
    Logs go to AETHER\logs\ (G:, never C:).

    Rollback: stop the process (it holds no persistent state); Cline profile rollback
    is documented separately in docs\cline-integration.md.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start-teacher-server.ps1
    powershell -File scripts\start-teacher-server.ps1 -Port 8081 -Device CUDA0
#>
param(
    [string]$Model = "$PSScriptRoot\..\models\Qwen3.5-9B-Q4_K_M.gguf",
    [string]$Device = "CUDA0",
    [int]$Port = 8080,
    [int]$Ctx = 8192,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$exe  = "$PSScriptRoot\..\builds\cuda\bin\llama-server.exe"
$logs = "$PSScriptRoot\..\logs"
$alias = "qwen3.5-9b-teacher"

if (-not (Test-Path $exe))   { Write-Error "llama-server.exe not found: $exe"; exit 2 }
if (-not (Test-Path $Model)) { Write-Error "model not found: $Model"; exit 2 }
New-Item -ItemType Directory -Force -Path $logs | Out-Null

# Refuse to collide with existing listeners (Ollama 11434 etc. must stay untouched).
$busy = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($busy) { Write-Error "port $Port already in use (PID $($busy.OwningProcess | Select-Object -First 1))"; exit 3 }

$srvArgs = @(
    "-m", $Model,
    "--device", $Device,
    "-ngl", "99",
    "-c", $Ctx,
    "--alias", $alias,
    "--jinja",
    "--host", "127.0.0.1",    # loopback only — never expose without auth
    "--port", $Port
)
$logFile = Join-Path $logs "teacher-server.log"

Write-Host "[teacher] model : $Model"
Write-Host "[teacher] device: $Device   port: $Port   ctx: $Ctx   alias: $alias"
Write-Host "[teacher] log   : $logFile"

if ($Foreground) {
    & $exe @srvArgs 2>&1 | Tee-Object -FilePath $logFile
    exit $LASTEXITCODE
}

# Start-Process joins array elements WITHOUT quoting — build an explicitly quoted
# command string so paths with spaces ("Program Prototype") survive intact.
$argLine = "-m `"$Model`" --device $Device -ngl 99 -c $Ctx --alias $alias --jinja --host 127.0.0.1 --port $Port"
$proc = Start-Process -FilePath $exe -ArgumentList $argLine `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err" `
        -WorkingDirectory (Split-Path $exe) -NoNewWindow -PassThru
Write-Host "[teacher] started PID $($proc.Id) — waiting for /health ..."

$ok = $false
for ($i = 0; $i -lt 45; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { Write-Error "server exited early (code $($proc.ExitCode)) — see $logFile.err"; exit 4 }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}
if (-not $ok) { Write-Error "health check timed out after 90s — see $logFile"; exit 5 }

Write-Host "[teacher] healthy ✔  PID=$($proc.Id)  endpoint=http://127.0.0.1:$Port/v1"
& (Join-Path $PSScriptRoot "stop-teacher-server.ps1") -StatusOnly   # prints status
