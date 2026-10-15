<#
.SYNOPSIS
  Reproducible CUDA build of llama.cpp (SamurAI / AETHER — Phase 2).

.DESCRIPTION
  Builds the official ggml-org/llama.cpp source with the NVIDIA CUDA backend,
  targeting BOTH installed NVIDIA GPUs in a single binary:
    - RTX 4070 Ti : compute capability 8.9 (Ada)
    - GTX 1060    : compute capability 6.1 (Pascal)

  Out-of-source build into <repo>\builds\cuda. Does NOT modify any existing
  CPU+RPC llama.cpp install.

.NOTES
  - Run from a normal PowerShell window; this script sets up the MSVC x64
    environment itself (vcvars64) so `cl.exe` and `cmake` resolve correctly.
  - Requires: VS 2022 BuildTools (bundled CMake 3.31.6), CUDA Toolkit 12.4.
#>
[CmdletBinding()]
param(
    [string]$Src = $(if ($env:AETHER_LLAMA_CPP_SRC) { $env:AETHER_LLAMA_CPP_SRC }
                    else { "$PSScriptRoot\..\vendor\llama.cpp" }),
    [string]$Out = "$PSScriptRoot\..\builds\cuda"
)

$ErrorActionPreference = 'Stop'

$vcvars   = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
$cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'

foreach ($p in @($vcvars, "$cmakeBin\cmake.exe", "$cudaRoot\bin\nvcc.exe", "$Src\CMakeLists.txt")) {
    if (-not (Test-Path $p)) { throw "Missing prerequisite: $p" }
}

# Log the exact environment for reproducibility
$logDir = "$PSScriptRoot\..\docs\raw"
New-Item -ItemType Directory -Force -Path $logDir, $Out | Out-Null

Write-Host "== Configuring CUDA build ==" -ForegroundColor Cyan
Write-Host "   SRC : $Src"
Write-Host "   OUT : $Out"

# Build inside a cmd shell that has run vcvars64 (gives us cl.exe) and has
# cmake + nvcc on PATH. Using 'Ninja' would need ninja installed; use the
# Visual Studio generator (MSVC) which is guaranteed present.
$cmakeArgs = @(
    '-S', "`"$Src`"",
    '-B', "`"$Out`"",
    '-G', 'Visual Studio 17 2022',
    '-A', 'x64',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_CUDA_ARCHITECTURES=89;61',   # Ada (4070 Ti) + Pascal (1060)
    '-DGGML_CUDA=ON',
    '-DGGML_CUDA_FA=ON',                    # FlashAttention (benefits cc 8.9)
    '-DGGML_BACKEND_DL=ON',                 # load backends as DLLs (matches existing install)
    '-DBUILD_SHARED_LIBS=ON',
    '-DLLAMA_BUILD_SERVER=ON',
    '-DLLAMA_BUILD_TOOLS=ON',
    '-DLLAMA_BUILD_EXAMPLES=ON',
    '-DLLAMA_CURL=OFF',                     # keep local inference path cloud-free
    '-DGGML_NATIVE=ON'                      # optimize for this Ryzen 3700X
) -join ' '

$cmd = "`"$vcvars`" >nul && set `"PATH=$cmakeBin;$cudaRoot\bin;%PATH%`" && cmake $cmakeArgs 2>&1"
Write-Host "== Running: cmake configure ==" -ForegroundColor Cyan
cmd /c $cmd | Tee-Object -FilePath (Join-Path $logDir 'build-cuda-configure.log')

if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE). See build-cuda-configure.log" }

Write-Host "== Building (Release) ==" -ForegroundColor Cyan
$buildCmd = "`"$vcvars`" >nul && set `"PATH=$cmakeBin;$cudaRoot\bin;%PATH%`" && cmake --build `"$Out`" --config Release --parallel 2>&1"
cmd /c $buildCmd | Tee-Object -FilePath (Join-Path $logDir 'build-cuda-build.log')

if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE). See build-cuda-build.log" }

Write-Host "== CUDA build complete ==" -ForegroundColor Green
Write-Host "Binaries in: $Out\bin\Release (or $Out\Release for VS generator)"
