<#
.SYNOPSIS
  Assemble a complete, buildable CUDA toolkit on G: from the extracted CUDA 12.8.1
  installer payload — WITHOUT installing to C: (which is nearly full).

.DESCRIPTION
  The official CUDA Windows installer is a self-extracting 7z archive. We extract its
  payload with 7-Zip, then merge the per-component folders (cuda_nvcc, cuda_cudart,
  libcublas, ...) into a single standard toolkit layout:

      <Root>\bin, <Root>\include, <Root>\lib\x64, <Root>\nvvm

  The result satisfies CMake's FindCUDAToolkit (nvcc.exe + cuda_runtime.h + cudart.lib).

.NOTES
  - Nothing is written to C:. Everything lands under the repo's toolchains\ folder.
  - Rollback = delete that folder.
  - Driver (581.57, max CUDA 13.0) is compatible with a 12.8 toolkit. Both GPUs
    (sm_61, sm_89) are covered.
#>
[CmdletBinding()]
param(
    [string]$Extract = "$PSScriptRoot\..\cache\cuda_extract",
    [string]$Root    = "$PSScriptRoot\..\toolchains\cuda-12.8.1"
)

$ErrorActionPreference = 'Stop'
$7z = 'C:\Program Files\7-Zip\7z.exe'
if (-not (Test-Path $7z)) { throw "7-Zip not found at $7z" }

# 1) Extract installer payload if not already present
if (-not (Test-Path (Join-Path $Extract 'cuda_nvcc'))) {
    $installer = 'G:\tmp\cuda_12.8.1_windows.exe'
    if (-not (Test-Path $installer)) {
        $url = 'https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_572.61_windows.exe'
        Write-Host "Downloading CUDA 12.8.1 installer to $installer ..."
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing -TimeoutSec 900
    }
    New-Item -ItemType Directory -Force -Path $Extract | Out-Null
    Write-Host "Extracting installer payload with 7-Zip ..."
    & $7z x $installer -o"$Extract" -y | Out-Null
}

# 2) Build the standard toolkit root
New-Item -ItemType Directory -Force -Path "$Root\bin","$Root\include","$Root\lib\x64","$Root\nvvm" | Out-Null

# Map: each extracted component folder -> its inner standard subdir
$components = @{
    'cuda_nvcc\nvcc'           = $true   # nvcc, cudafe++, ptxas, nvlink, fatbinary, bin2c
    'cuda_cudart\cudart'       = $true   # cudart64_12.dll, cuda_runtime.h, cudart.lib
    'cuda_cccl\thrust'         = $true   # cub/thrust headers (used by some ggml code)
    'cuda_nvrtc\nvrtc'         = $true   # nvrtc
    'cuda_nvrtc\nvrtc_dev'     = $true
    'cuda_profiler_api\cuda_profiler_api' = $true
    'cuda_sanitizer_api\sanitizer'         = $true
    'cuda_nvml_dev\nvml_dev'   = $true
    'cuda_nvtx\nvtx'           = $true
    'libnvjitlink\nvjitlink'   = $true
    'libnvfatbin\nvfatbin'     = $true
    'cuda_cuobjdump\cuobjdump' = $true
    'cuda_nvdisasm\nvdisasm'   = $true
    'cuda_nvprune\nvprune'     = $true
}

function Merge-Component($srcRoot) {
    if (-not (Test-Path $srcRoot)) { return }
    foreach ($sub in 'bin','include','nvvm') {
        $s = Join-Path $srcRoot $sub
        if (Test-Path $s) { Copy-Item -Path "$s\*" -Destination "$Root\$sub" -Recurse -Force -ErrorAction SilentlyContinue }
    }
    $libx64 = Join-Path $srcRoot 'lib\x64'
    if (Test-Path $libx64) { Copy-Item -Path "$libx64\*" -Destination "$Root\lib\x64" -Recurse -Force -ErrorAction SilentlyContinue }
    # some components keep libs directly under lib\
    $lib = Join-Path $srcRoot 'lib'
    if ((Test-Path $lib) -and -not (Test-Path (Join-Path $lib 'x64'))) {
        Copy-Item -Path "$lib\*" -Destination "$Root\lib\x64" -Recurse -Force -ErrorAction SilentlyContinue
    }
}

foreach ($comp in $components.Keys) {
    $srcRoot = Join-Path $Extract $comp
    if (Test-Path $srcRoot) { Write-Host "Merging $comp"; Merge-Component $srcRoot }
}

# Math libraries (cublas etc.) ship their DLLs in bin\ and import libs in lib\x64
foreach ($math in 'libcublas\cublas','libcublas\cublas_dev','libcufft\cufft','libcurand\curand','libcusolver\cusolver','libcusparse\cusparse','libnpp\npp') {
    $srcRoot = Join-Path $Extract $math
    if (-not (Test-Path $srcRoot)) { continue }
    Write-Host "Merging $math"
    foreach ($sub in 'bin','include','nvvm') {
        $s = Join-Path $srcRoot $sub
        if (Test-Path $s) { Copy-Item -Path "$s\*" -Destination "$Root\$sub" -Recurse -Force -ErrorAction SilentlyContinue }
    }
    $libx64 = Join-Path $srcRoot 'lib\x64'
    if (Test-Path $libx64) { Copy-Item -Path "$libx64\*" -Destination "$Root\lib\x64" -Recurse -Force -ErrorAction SilentlyContinue }
}

# 3) Verify the essentials exist
$must = @(
    "$Root\bin\nvcc.exe",
    "$Root\bin\ptxas.exe",
    "$Root\include\cuda_runtime.h",
    "$Root\lib\x64\cudart.lib",
    "$Root\bin\cudart64_12.dll"
)
$missing = $must | Where-Object { -not (Test-Path $_) }
if ($missing) { throw "Assembled toolkit is missing: $($missing -join ', ')" }

Write-Host "== CUDA toolkit assembled at $Root ==" -ForegroundColor Green
& "$Root\bin\nvcc.exe" --version
Write-Host "Use with CMake: -DCUDAToolkit_ROOT=`"$Root`""
