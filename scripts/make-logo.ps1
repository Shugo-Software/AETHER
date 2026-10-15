# AETHER logo assets — reproducible generation from the source emblem.
#
# The source is a 1254x1254 PNG of black line art on near-white, with no alpha
# channel. This script derives what the dashboard and docs actually need:
#
#   assets/aether-logo.png            the source, untouched (docs/README)
#   assets/aether-icon.png            128px, black art, transparent (light favicon)
#   assets/aether-icon-dark.png       128px, white art, transparent (dark favicon
#                                     + the dashboard header, which is dark)
#   assets/aether-logo-dark-256.png   256px, white art, transparent (high-DPI header)
#
# Matte extraction for monochrome line art: alpha = 255 - luminance, ink colour
# forced to black (or white). That keeps the anti-aliased edges instead of
# hard-thresholding them into jaggies.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\make-logo.ps1 `
#            -Source "$env:USERPROFILE\Downloads\Samurai Rising Sun Emblem.png"
param(
  [Parameter(Mandatory = $true)][string]$Source,
  [string]$OutDir = "assets"
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

if (-not (Test-Path $Source)) { Write-Error "source image not found: $Source"; exit 1 }
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$out = Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $out | Out-Null

$full = Join-Path $out "aether-logo.png"
Copy-Item $Source $full -Force
Write-Output ("[logo] {0} (source, {1} KB)" -f $full, [int]((Get-Item $full).Length / 1KB))

function New-MattedIcon {
  param([string]$Path, [int]$Size, [bool]$White)
  $img = [System.Drawing.Image]::FromFile($Source)
  try {
    $fmt = [System.Drawing.Imaging.PixelFormat]::Format32bppArgb
    $bmp = New-Object System.Drawing.Bitmap -ArgumentList $Size, $Size, $fmt
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    try {
      $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
      $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
      $g.Clear([System.Drawing.Color]::Transparent)
      $g.DrawImage($img, 0, 0, $Size, $Size)
    } finally { $g.Dispose() }
    $ink = 0
    if ($White) { $ink = 255 }
    for ($y = 0; $y -lt $Size; $y++) {
      for ($x = 0; $x -lt $Size; $x++) {
        $c = $bmp.GetPixel($x, $y)
        $lum = [int](0.299 * $c.R + 0.587 * $c.G + 0.114 * $c.B)
        $a = 255 - $lum
        if ($a -lt 0) { $a = 0 }
        $bmp.SetPixel($x, $y, [System.Drawing.Color]::FromArgb($a, $ink, $ink, $ink))
      }
    }
    $bmp.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    Write-Output ("[logo] {0} ({1}x{1}, {2} art, {3} KB)" -f `
        $Path, $Size, $(if ($White) { "white" } else { "black" }), `
        [int]((Get-Item $Path).Length / 1KB))
  } finally { $img.Dispose() }
}

New-MattedIcon -Path (Join-Path $out "aether-icon.png") -Size 128 -White $false
New-MattedIcon -Path (Join-Path $out "aether-icon-dark.png") -Size 128 -White $true
New-MattedIcon -Path (Join-Path $out "aether-logo-dark-256.png") -Size 256 -White $true
