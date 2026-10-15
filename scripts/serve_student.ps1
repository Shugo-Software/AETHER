<#
.SYNOPSIS
    Start / stop a Jadis STUDENT endpoint: one GGUF served by llama-server, pinned
    to a single GPU, OpenAI-compatible on a chosen port.

.DESCRIPTION
    The recursive loop runs three endpoints at once on this rig:

      teacher   CUDA0  :8080  Qwen3.5-9B Q4_K_M   scripts\start-teacher-server.ps1
      student A CUDA2  :8082  Jadis  iq4_xs       GTX 1060 3 GB  (1.45 GiB)
      student B CUDA1  :8083  Jadis  f16 or q8_0  GTX 1060 6 GB

    Device ids are llama.cpp's (CUDA0/1/2) - they do NOT match nvidia-smi
    indices (principle #4; see docs/hardware-inventory.md section 4.3). Runtime
    DLLs must sit next to the exe (ggml-cuda.dll silently fails otherwise), so
    the working directory is the build's bin folder.

    Loopback-only by default: llama-server has no auth. Logs go under logs/.

.EXAMPLE
    powershell -File scripts\serve_student.ps1 `
        -Model exports\jadis-1.0\jadis-1.0-iq4_xs.gguf -Device CUDA2 -Port 8082 `
        -Alias jadis-a -Ctx 4096
.EXAMPLE
    powershell -File scripts\serve_student.ps1 -Stop -Port 8082
#>
param(
    [string]$Model = "$PSScriptRoot\..\exports\jadis-1.0\jadis-1.0-q4_k_m.gguf",
    [string]$Device = "CUDA2",
    [int]$Port = 8082,
    [int]$Ctx = 4096,
    [string]$Alias = "jadis",
    [string]$AliasPrefix = "jadis",
    [int]$Threads = 0,
    [switch]$Stop,
    [switch]$StopAll,
    [switch]$StatusOnly,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$exe  = "$PSScriptRoot\..\builds\cuda\bin\llama-server.exe"
$logs = "$PSScriptRoot\..\logs"

function Get-StudentServers {
    # STUDENT servers only: the teacher (start-teacher-server.ps1, alias
    # "qwen3.5-9b-teacher") is also llama-server.exe, so matching every
    # llama-server would let `-StopAll` kill the teacher (it did, once).
    Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match '--port\s+(\d+)' -and
                       $_.CommandLine -match ('--alias\s+' + [regex]::Escape($AliasPrefix)) } |
        ForEach-Object {
            [PSCustomObject]@{
                Pid   = $_.ProcessId
                Port  = [int]([regex]::Match($_.CommandLine, '--port\s+(\d+)').Groups[1].Value)
                Alias = [regex]::Match($_.CommandLine, '--alias\s+(\S+)').Groups[1].Value
            }
        }
}

function Stop-OnPort([int]$p) {
    $target = Get-StudentServers | Where-Object { $_.Port -eq $p }
    if (-not $target) { Write-Host "[student] nothing of ours on :$p"; return }
    foreach ($t in $target) {
        # PID-guarded: only processes whose alias matches the student prefix
        Stop-Process -Id $t.Pid -Force
        Write-Host "[student] stopped PID $($t.Pid) (port $p, alias $($t.Alias))"
    }
}

if ($StatusOnly -or $Stop -or $StopAll) {
    $all = Get-StudentServers
    if ($StopAll) {
        if (-not $all) { Write-Host "[student] no student servers running (teacher untouched)" }
        foreach ($t in $all) { Stop-Process -Id $t.Pid -Force; Write-Host "[student] stopped PID $($t.Pid) :$($t.Port) alias=$($t.Alias)" }
        exit 0
    }
    if ($Stop) { Stop-OnPort $Port; exit 0 }
    if ($all) { $all | Format-Table -AutoSize | Out-String | Write-Host }
    else { Write-Host "[student] no student servers running" }
    exit 0
}

if (-not (Test-Path $exe))   { Write-Error "llama-server.exe not found: $exe"; exit 2 }
if (-not (Test-Path $Model)) { Write-Error "model not found: $Model"; exit 2 }
# Resolve the model to an ABSOLUTE path: the server's working directory is the
# build's bin folder, so a repo-relative path would not resolve there.
$Model = (Resolve-Path -LiteralPath $Model).ProviderPath
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$busy = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($busy) { Write-Error "port $Port already in use (PID $($busy.OwningProcess | Select-Object -First 1))"; exit 3 }

$mb = [math]::Round((Get-Item $Model).Length / 1MB, 1)
$logFile = Join-Path $logs ("student-{0}-{1}.log" -f $Alias, $Port)
$threadArg = if ($Threads -gt 0) { "-t $Threads" } else { "" }

Write-Host "[student] model  : $Model ($mb MiB)"
Write-Host "[student] device : $Device   port: $Port   ctx: $Ctx   alias: $Alias"
Write-Host "[student] log    : $logFile"

# Start-Process joins array elements WITHOUT quoting -> build an explicitly quoted
# command line so a spaced path ("Program Prototype") survives.
$argLine = "-m `"$Model`" --device $Device -ngl 99 -c $Ctx --alias $Alias --jinja --host 127.0.0.1 --port $Port $threadArg".Trim()

if ($Foreground) { & $exe @($argLine -split ' ') 2>&1 | Tee-Object -FilePath $logFile; exit $LASTEXITCODE }

$proc = Start-Process -FilePath $exe -ArgumentList $argLine `
        -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err" `
        -WorkingDirectory (Split-Path $exe) -PassThru
Write-Host "[student] started PID $($proc.Id) - waiting for /health ..."

$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { Write-Error "server exited early (code $($proc.ExitCode)) - see $logFile.err"; exit 4 }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}
if (-not $ok) { Write-Error "health check timed out after 120s - see $logFile"; exit 5 }

Write-Host "[student] healthy  PID=$($proc.Id)  endpoint=http://127.0.0.1:$Port/v1  model=$Alias"
