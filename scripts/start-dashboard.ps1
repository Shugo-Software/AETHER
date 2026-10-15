# AETHER dashboard launcher. Stdlib-only.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/start-dashboard.ps1
#          [-Port 8091] [-Bind tailscale|loopback|<explicit IP>]
# Bind modes:
#   tailscale (default) - bind the Tailscale interface IPv4 (auto-detected);
#                         falls back to loopback if Tailscale is down.
#   loopback            - 127.0.0.1 only (nothing remote can connect).
#   <explicit IP>       - bind that address as-is (e.g. 0.0.0.0 = all interfaces,
#                         which exposes the control plane to every network).
# NOTE: a tailscale bind only answers tailnet devices; local browsers should use
# the printed URL. Binding 0.0.0.0 needs a scoped firewall rule to stay safe.
param(
  [int]$Port = 8090,
  [string]$Bind = "tailscale"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$py = Join-Path $root "venvs\ml\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "venv python missing: $py"; exit 1 }
$busy = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Where-Object { $_.LocalPort -eq $Port }
if ($busy) { Write-Output "[dashboard] port $Port already in use."; exit 0 }
$hostIp = "127.0.0.1"
if ($Bind -eq "tailscale") {
  $ip = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Tailscale" `
      -ErrorAction SilentlyContinue | Select-Object -First 1).IPAddress
  if ($ip) { $hostIp = $ip }
  else { Write-Output "[dashboard] Tailscale interface not found; falling back to loopback." }
} elseif ($Bind -ne "loopback") {
  $hostIp = $Bind
}
$log = Join-Path $root "logs\dashboard-$Port.log"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null
$p = Start-Process -FilePath $py `
  -ArgumentList "-u", "`"$here\dashboard.py`"", "--port", "$Port", "--host", "$hostIp" `
  -WorkingDirectory $here `
  -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
Write-Output "[dashboard] PID $($p.Id) bind $hostIp (log: $log)"
if ($hostIp -eq "127.0.0.1") {
  Write-Output "[dashboard] http://127.0.0.1:$Port/ (loopback only)"
} elseif ($hostIp -eq "0.0.0.0") {
  Write-Output "[dashboard] ALL interfaces - local http://127.0.0.1:$Port/ + http://<this-machine>:$Port/"
} else {
  Write-Output "[dashboard] tailnet http://${hostIp}:$Port/"
}
