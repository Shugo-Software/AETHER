<#
.SYNOPSIS
  Reproducible Vulkan build of llama.cpp (SamurAI / AETHER — Phase 2, primary path).

.DESCRIPTION
  Builds the official ggml-org/llama.cpp source with the Vulkan backend so that ALL THREE
  GPUs are usable for inference:
    - RTX 4070 Ti (NVIDIA, Vulkan 1.4.312)
    - GTX 1060    (NVIDIA, Vulkan 1.4.312)
    - Radeon RX 580 (AMD, Vulkan 1.3.217)

  Out-of-source build into <repo>\builds\vulkan. Does NOT modify any existing
  CPU+RPC llama.cpp install.

.PREREQUISITE (BLOCKER as of 2026-10-13)
  The ggml-vulkan backend requires the Vulkan SDK at CONFIGURE time:
    - glslc (Vulkan shader compiler)
    - SPIRV-Headers (SPIRV-HeadersConfig.cmake)
  Neither is installed (VULKAN_SDK unset). The Vulkan RUNTIME is present, but the SDK is
  needed to BUILD. Install the LunarG Vulkan SDK (sets VULKAN_SDK) before running this.

.NOTES
  Run from a normal PowerShell window; sets up MSVC x64 env itself.
  Requires: VS 2022 BuildTools (bundled CMake), Vulkan SDK with glslc + SPIRV-Headers.
#>
[CmdletBinding()]
param(
    [string]$Src = $(if ($env:AETHER_LLAMA_CPP_SRC) { $env:AETHER_LLAMA_CPP_SRC }
                    else { "$PSScriptRoot\..\vendor\llama.cpp" }),
    [string]$Out = "$PSScriptRoot\..\builds\vulkan"
)

$ErrorActionPreference = 'Stop'

$vcvars   = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$cmakeBin = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'

foreach ($p in @($vcvars, "$cmakeBin\cmake.exe", "$Src\CMakeLists.txt")) {
    if (-not (Test-Path $p)) { throw "Missing prerequisite: $p" }
}

# Vulkan SDK must be present to configure ggml-vulkan
if (-not $env:VULKAN_SDK) {
    throw "VULKAN_SDK is not set. Install the LunarG Vulkan SDK (provides glslc + SPIRV-Headers) first. See docs/build-guide.md section 4."
}
$glslc = Get-Command glslc -ErrorAction SilentlyContinue
if (-not $glslc) {
    throw "glslc not on PATH. Ensure the Vulkan SDK Bin directory is on PATH (or run from a Vulkan SDK shell)."
}

$logDir = "$PSScriptRoot\..\docs\raw"
New-Item -ItemType Directory -Force -Path $logDir, $Out | Out-Null

Write-Host "== Configuring Vulkan build ==" -ForegroundColor Cyan
Write-Host "   SRC : $Src"
Write-Host "   OUT : $Out"
Write-Host "   VULKAN_SDK: $env:VULKAN_SDK"

$cmakeArgs = @(
    '-S', "`"$Src`"",
    '-B', "`"$Out`"",
    '-G', 'Visual Studio 17 2022',
    '-A', 'x64',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DGGML_VULKAN=ON',
    '-DGGML_BACKEND_DL=ON',                 # load backends as DLLs (matches existing install)
    '-DGGML_CPU_ALL_VARIANTS=ON',           # required with BACKEND_DL
    '-DBUILD_SHARED_LIBS=ON',
    '-DLLAMA_BUILD_SERVER=ON',
    '-DLLAMA_BUILD_TOOLS=ON',
    '-DLLAMA_BUILD_EXAMPLES=ON',
    '-DLLAMA_CURL=OFF'                      # keep local inference path cloud-free
) -join ' '

$cmd = "`"$vcvars`" >nul && set `"PATH=$cmakeBin;%PATH%`" && cmake $cmakeArgs 2>&1"
Write-Host "== Running: cmake configure ==" -ForegroundColor Cyan
cmd /c $cmd | Tee-Object -FilePath (Join-Path $logDir 'build-vulkan-configure.log')

if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE). See build-vulkan-configure.log" }

Write-Host "== Building (Release) ==" -ForegroundColor Cyan
$buildCmd = "`"$vcvars`" >nul && set `"PATH=$cmakeBin;%PATH%`" && cmake --build `"$Out`" --config Release --parallel 2>&1"
cmd /c $buildCmd | Tee-Object -FilePath (Join-Path $logDir 'build-vulkan-build.log')

if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE). See build-vulkan-build.log" }

Write-Host "== Vulkan build complete ==" -ForegroundColor Green
Write-Host "Verify with: llama-cli --list-devices  (expect RTX 4070 Ti, GTX 1060, RX 580)"
Write-Host "Then a smoke test with the Qwen2.5-0.5B Q4_K_M GGUF (see docs/build-guide.md)."
