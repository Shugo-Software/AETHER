@echo off
REM Trainer launcher. Resolves the repo root from this script's own location, so
REM it works from any checkout (and survives the space in "Program Prototype").
for %%i in ("%~dp0..") do set "ROOT=%%~fi"
REM Single-card default: only the 1060 6GB is visible (torch idx 0 after mask).
set "HF_HUB_CACHE=%ROOT%\cache\hf"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set "PIP_CACHE_DIR=%ROOT%\cache\pip"
REM C: is critically small/volatile (plan.md): keep ALL temp inside the repo
REM cache so the trainer and its caches never fill C:.
set "TEMP=%ROOT%\cache\temp"
set "TMP=%ROOT%\cache\temp"
set "TMPDIR=%ROOT%\cache\temp"
if not exist "%ROOT%\cache\temp" mkdir "%ROOT%\cache\temp"
if "%CUDA_VISIBLE_DEVICES%"=="" set CUDA_VISIBLE_DEVICES=1
"%ROOT%\venvs\ml\Scripts\python.exe" -u "%ROOT%\scripts\train_student.py" %*
