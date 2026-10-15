<#
.SYNOPSIS
    Stop or check the SamurAI TEACHER endpoint (llama-server on localhost:8080).

.EXAMPLE
    powershell -File scripts\stop-teacher-server.ps1            # stop it
    powershell -File scripts\stop-teacher-server.ps1 -StatusOnly # just report
#>
param(
    [int]$Port = 8080,
    [switch]$StatusOnly
)

$ErrorActionPreference = "SilentlyContinue"
$conn = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Where-Object { $_.OwningProcess } | Select-Object -First 1

if (-not $conn) {
    if ($StatusOnly) { Write-Host "[teacher] nothing listening on :$Port (stopped)" }
    else             { Write-Host "[teacher] already stopped" }
    exit 0
}

$proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
$isServer = $proc -and $proc.Name -eq "llama-server"

if ($StatusOnly) {
    if ($isServer) { Write-Host "[teacher] running — PID $($proc.Id) on :$Port" }
    else           { Write-Host "[teacher] :$Port held by PID $($conn.OwningProcess) ($($proc.Name)) — NOT ours" }
    exit 0
}

if (-not $isServer) {
    Write-Error "refusing to kill PID $($conn.OwningProcess) ($($proc.Name)) — not llama-server"
    exit 3
}
Stop-Process -Id $proc.Id -Force
Write-Host "[teacher] stopped PID $($proc.Id)"
