@echo off
REM ============================================================================
REM  Reproducible CUDA build of llama.cpp (SamurAI / AETHER - Phase 2)
REM  - Ninja generator (drives nvcc directly; no VS CUDA integration needed)
REM  - Portable CUDA 12.8 toolkit assembled on G: by assemble-cuda-toolkit.ps1
REM  - Targets both NVIDIA GPUs: sm_89 (RTX 4070 Ti) + sm_61 (GTX 1060)
REM  Usage: build-cuda.bat            (configure + build)
REM         build-cuda.bat configure  (configure only)
REM ============================================================================
setlocal EnableDelayedExpansion

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CMKBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
set "ROOT=%~dp0.."
if "%NINJA%"=="" set "NINJA=%ROOT%\toolchains\ninja\bin"
set "CUDA=%ROOT%\toolchains\cuda-12.8.1"
set "SRC=%ROOT%\vendor\llama.cpp"
set "OUT=%ROOT%\builds\cuda"
set "LOG=%ROOT%\docs\raw\build-cuda-configure.log"

call "%VCVARS%" >nul
if errorlevel 1 ( echo [x] vcvars failed & exit /b 1 )
set "PATH=%CMKBIN%;%CUDA%\bin;%NINJA%;%PATH%"

echo == Configuring CUDA build ==
cmake -S "%SRC%" -B "%OUT%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%\ninja.exe" ^
  -DCMAKE_CUDA_ARCHITECTURES=89;61 ^
  -DCUDAToolkit_ROOT="%CUDA%" ^
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON ^
  -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DBUILD_SHARED_LIBS=ON ^
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON ^
  -DLLAMA_CURL=OFF
if errorlevel 1 ( echo [x] configure failed - see %LOG% & exit /b 1 )
echo [+] configure OK

if /I "%~1"=="configure" exit /b 0

echo == Building Release ==
cmake --build "%OUT%" --config Release --parallel
if errorlevel 1 ( echo [x] build failed & exit /b 1 )
echo [+] CUDA build complete: %OUT%\bin
echo     Verify: llama-cli --list-devices   (expect the two NVIDIA GPUs)
