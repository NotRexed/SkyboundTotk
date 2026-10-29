@echo off
setlocal
cd /d "%~dp0"

:: ── RAM-safe build for SkyboundTOTK ──────────────────────────────────────────
::
:: Your 16 GB system crashes because the default build launches one clang job
:: per logical core (22 on an Ultra 7 155H).  Each -O3 job on TOTK's giant
:: lifted C files uses 3-8 GB of RAM → 22 × 3 GB = 66 GB needed → OOM crash.
::
:: The Python script now picks a safe job count automatically.  You can also
:: set these environment variables before running:
::
::   RX_JOBS=4   force exactly 4 parallel compile jobs (override auto-detect)
::   RX_OPT=O1   use -O1 instead of -O3 (much faster build, slightly slower game)
::   RX_OPT=O2   middle ground
::
:: Examples:
::   set RX_JOBS=3 && build.bat     (3 jobs, ~10.5 GB peak)
::   set RX_OPT=O1 && build.bat     (fast build, ~1 GB/job)
:: ─────────────────────────────────────────────────────────────────────────────

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer was not found on PATH. See README.md.
  pause
  exit /b 1
)
python "%~dp0build_game.py"
set "BUILD_RESULT=%ERRORLEVEL%"
if not "%BUILD_RESULT%"=="0" echo Build failed with exit code %BUILD_RESULT%.
pause
exit /b %BUILD_RESULT%
