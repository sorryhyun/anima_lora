# anima_lora bootstrap installer (Windows / PowerShell).
#
#   irm https://raw.githubusercontent.com/sorryhyun/anima_lora/main/install.ps1 | iex
#
# Installs uv if missing, selects CUDA or ROCm, guides the CUDA 13.2 toolkit
# install if needed,
# downloads the latest release tarball (no git required), seeds the update
# baseline so the first `make update` is clean, and runs `uv sync`. Mirrors
# scripts/update.py — keep the two in sync.
#
# Options (env vars, since args don't pass through `irm | iex`):
#   $env:ANIMA_VERSION    = 'v1.4.0'   install a specific tag   (default: latest)
#   $env:ANIMA_DIR        = 'C:\path'  target directory         (default: .\anima_lora)
#   $env:ANIMA_BACKEND    = 'auto'     auto | cuda | rocm        (default: auto)
#   $env:ANIMA_SKIP_CUDA  = '1'        skip the CUDA 13.2 toolkit install/check

$ErrorActionPreference = 'Stop'
$Repo    = 'sorryhyun/anima_lora'
$Version = $env:ANIMA_VERSION
$Dir     = if ($env:ANIMA_DIR) { $env:ANIMA_DIR } else { 'anima_lora' }
$Backend = if ($env:ANIMA_BACKEND) { $env:ANIMA_BACKEND.ToLowerInvariant() } else { 'auto' }

function Say($m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Die($m)  { Write-Host "error: $m" -ForegroundColor Red; exit 1 }
function Warn($m) { Write-Host "warning: $m" -ForegroundColor Yellow }

if ($Backend -notin @('auto', 'cuda', 'rocm')) {
  Die "ANIMA_BACKEND must be auto, cuda, or rocm (got '$Backend')"
}

if ($Backend -eq 'auto') {
  try {
    $gpuNames = @(Get-CimInstance Win32_VideoController -ErrorAction Stop |
      ForEach-Object { "$($_.Name) $($_.AdapterCompatibility)" })
  } catch {
    $gpuNames = @()
    Warn 'GPU vendor detection failed; preserving the CUDA default'
  }
  $hasNvidia = [bool]($gpuNames -match '(?i)NVIDIA')
  $hasAmd = [bool]($gpuNames -match '(?i)AMD|Advanced Micro Devices|Radeon')
  if ($hasNvidia) {
    $Backend = 'cuda' # Preserve the existing default on mixed-vendor systems.
  } elseif ($hasAmd) {
    $Backend = 'rocm'
  } else {
    $Backend = 'cuda'
  }
}
Say "selected $Backend backend (override with ANIMA_BACKEND=auto|cuda|rocm)"

# 0. neutralize an active Conda env (GH #21) ---------------------------------
# With `conda activate base` live, conda's Library\bin sits on PATH. uv builds
# its own .venv with the correct PySide6 DLLs, but at GUI launch Windows loads
# conda's mismatched Qt DLLs first and PySide6 dies with
#   ImportError: DLL load failed while importing QtCore
# We strip conda dirs from PATH and clear CONDA_* for THIS install session so
# `uv sync` and the GUI launch below are clean, then warn the user to
# `conda deactivate` before relaunching later (their shell still has it active).
$CondaActive = $env:CONDA_PREFIX -or $env:CONDA_DEFAULT_ENV
if ($CondaActive) {
  Warn "an active Conda environment was detected (CONDA_PREFIX=$env:CONDA_PREFIX)."
  Say  'neutralizing Conda on PATH for this install session'
  $condaRoots = @($env:CONDA_PREFIX, $env:CONDA_ROOT, $env:_CONDA_ROOT) |
    Where-Object { $_ } | ForEach-Object { $_.TrimEnd('\') }
  $env:Path = ($env:Path -split ';' | Where-Object {
    $p = $_.TrimEnd('\')
    if (-not $p) { return $false }
    foreach ($r in $condaRoots) { if ($p -like "$r*") { return $false } }
    # Catch conda dirs even when the roots env vars are unset.
    return ($p -notmatch '(?i)\\(ana|mini)conda[^\\]*\\' -and $p -notmatch '(?i)\\condabin')
  }) -join ';'
  Remove-Item Env:CONDA_PREFIX,Env:CONDA_DEFAULT_ENV,Env:CONDA_SHLVL,Env:CONDA_PROMPT_MODIFIER -ErrorAction SilentlyContinue
}

# 1. uv ----------------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Say 'installing uv (https://astral.sh/uv)'
  irm https://astral.sh/uv/install.ps1 | iex
  $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Die 'uv install failed; open a new PowerShell and re-run'
}

# 1.5 CUDA 13.2 toolkit (required for torch.compile / Triton ptxas) ----------
# The bundled torch wheels carry the CUDA *runtime*, but this repo's compile
# path (Triton -> ptxas) needs the system toolkit. Done HERE, before the repo
# is downloaded, so that if the CUDA installer demands a reboot you can simply
# re-run this one-liner: CUDA is detected + skipped and the install continues
# (re-running after the repo dir exists would abort at the not-empty check).
function Test-Cuda132 {
  $nvcc = (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
  if (-not $nvcc) {
    $p = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe'
    if (Test-Path $p) { $nvcc = $p } else { return $false }
  }
  try { return [bool]((& $nvcc --version 2>$null) -match 'release 13\.2([^0-9]|$)') } catch { return $false }
}
if ($Backend -eq 'rocm') {
  Say 'ROCm selected - skipping the NVIDIA CUDA toolkit check'
} elseif ($env:ANIMA_SKIP_CUDA) {
  Say 'ANIMA_SKIP_CUDA set — skipping the CUDA 13.2 toolkit check'
} elseif (Test-Cuda132) {
  Say 'CUDA 13.2 toolkit detected'
} else {
  Warn 'CUDA 13.2 toolkit not found — required for torch.compile / Triton (ptxas)'
  $cudaUrl = 'https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_windows.exe'
  $cudaExe = Join-Path ([System.IO.Path]::GetTempPath()) 'cuda_13.2.0_windows.exe'
  Say "downloading the CUDA 13.2 installer (~4 GB) -> $cudaExe"
  $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
  if ($curl) {
    & $curl.Source -L --fail --retry 5 --retry-all-errors --retry-delay 2 -o $cudaExe $cudaUrl
    if ($LASTEXITCODE -ne 0) { Die "CUDA installer download failed (curl exit $LASTEXITCODE)" }
  } else {
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $cudaUrl -OutFile $cudaExe -UseBasicParsing
  }
  Say 'launching the CUDA installer — choose "Express (Recommended)"'
  Start-Process -FilePath $cudaExe   # the installer's manifest triggers UAC elevation
  Warn 'If the installer asks to REBOOT, do so, then re-run this one-liner — CUDA will be detected and skipped.'
  Read-Host 'Press Enter once the CUDA installer has finished' | Out-Null
  Remove-Item $cudaExe -ErrorAction SilentlyContinue
  if (-not (Test-Cuda132)) {
    Die 'CUDA 13.2 still not detected. Install from https://developer.nvidia.com/cuda-13-2-0-download-archive then re-run.'
  }
  Say 'CUDA 13.2 toolkit detected'
}

# 2. resolve the release tag -------------------------------------------------
if (-not $Version) {
  Say "resolving latest release of $Repo"
  $rel = irm "https://api.github.com/repos/$Repo/releases/latest" `
            -Headers @{ Accept = 'application/vnd.github+json' }
  $Version = $rel.tag_name
  if (-not $Version) { Die 'could not resolve latest release tag from GitHub API' }
}
Say "installing $Repo @ $Version -> $Dir\"

if ((Test-Path $Dir) -and (Get-ChildItem -Force $Dir | Select-Object -First 1)) {
  Die "$Dir\ already exists and is not empty - set `$env:ANIMA_DIR to a different path"
}

# 3. download + extract ------------------------------------------------------
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("anima-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
try {
  # Use the zipball + .NET unzip, NOT the bundled tar.exe: Windows' bsdtar
  # decodes archive entry names with the active ANSI code page and chokes on
  # the non-ASCII guidebook filenames under docs/guidelines/ (가이드북.md,
  # ガイドブック.md, 指南书.md) with "Invalid empty pathname". GitHub's zipball
  # flags entry names as UTF-8 and .NET's ZipFile honors that.
  $Zipball = "https://github.com/$Repo/archive/refs/tags/$Version.zip"
  Say "downloading $Zipball"
  $zip = Join-Path $Tmp 'release.zip'
  # Stream to disk with curl.exe (ships with Windows 10 1803+): it follows the
  # codeload redirect and retries mid-stream resets, which `irm -OutFile` does
  # not -- a dropped packet there aborts the whole install. Fall back to
  # Invoke-WebRequest on older boxes that lack curl.exe.
  $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
  if ($curl) {
    & $curl.Source -L --fail --retry 5 --retry-all-errors --retry-delay 2 -o $zip $Zipball
    if ($LASTEXITCODE -ne 0) { Die "download failed (curl exit $LASTEXITCODE): $Zipball" }
  } else {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $ProgressPreference = 'SilentlyContinue'  # progress bar throttles IWR badly
    Invoke-WebRequest -Uri $Zipball -OutFile $zip -UseBasicParsing
  }
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $Tmp, [System.Text.Encoding]::UTF8)
  $top = Get-ChildItem -Directory $Tmp | Select-Object -First 1
  if (-not $top) { Die 'unexpected archive layout' }
  New-Item -ItemType Directory -Force -Path $Dir | Out-Null
  Copy-Item -Path (Join-Path $top.FullName '*') -Destination $Dir -Recurse -Force
} finally {
  Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}

Set-Location $Dir

# Persist the install choice for scripts/update.py. This local file is ignored
# by git and survives release updates.
Set-Content -LiteralPath '.anima_backend' -Value $Backend -Encoding ascii

# 4. seed the update baseline (before uv sync, so .venv isn't hashed) --------
Say 'seeding update baseline (.anima_release.json)'
try {
  uv run --no-project python scripts/update.py --seed-manifest --version $Version
} catch {
  Say 'manifest seed skipped (first `make update` will back up instead - harmless)'
}

# 5. dependencies ------------------------------------------------------------
# Best-effort Defender exclusions so the install dir + uv cache are skipped by
# real-time scanning. uv populates .venv\Scripts\ with unsigned trampoline
# .exe launchers (uv-trampoline-*.exe, written-then-renamed); Defender often
# locks/quarantines those mid-write and `uv sync` dies with "Access is denied".
# Add-MpPreference needs elevation + Defender as the active AV -- if either is
# missing this no-ops silently and we fall back to the retry loop below.
$uvCache = Join-Path $env:LOCALAPPDATA 'uv'
try {
  Add-MpPreference -ExclusionPath (Resolve-Path '.').Path, $uvCache -ErrorAction Stop
  Say 'added Windows Defender exclusions for the install dir + uv cache'
} catch {
  # not elevated, third-party AV, or Defender disabled -- retry loop handles it
}

# cuda-windows is a DEFAULT dependency group (GH #92: a plain `uv sync` must
# land the CUDA stack); ROCm swaps that group out explicitly.
$SyncArgs = if ($Backend -eq 'rocm') { @('--no-group', 'cuda-windows', '--group', 'rocm-windows') } else { @() }
Say "running uv sync $($SyncArgs -join ' ') (may take a while)"
$syncOk = $false
for ($attempt = 1; $attempt -le 3; $attempt++) {
  uv sync @SyncArgs
  if ($LASTEXITCODE -eq 0) { $syncOk = $true; break }
  if ($attempt -lt 3) {
    Say "uv sync failed (exit $LASTEXITCODE); retrying ($attempt/2) in 3s -- often a transient antivirus lock on a trampoline .exe"
    Start-Sleep -Seconds 3
  }
}
if (-not $syncOk) {
  Write-Host ""
  Write-Host "uv sync did not complete after 3 attempts." -ForegroundColor Red
  Write-Host @"
This is almost always Windows Defender (or another antivirus) blocking uv's
trampoline .exe files. To fix:

  1. Open 'Windows Security' -> 'Virus & threat protection' -> 'Manage settings'
     and add folder exclusions for:
        $((Resolve-Path '.').Path)
        $uvCache
        $env:TEMP
     (or run this installer from an elevated PowerShell so it can add them
     automatically), then re-run:  uv sync
  2. If that still fails, briefly turn off 'Real-time protection', run
     'uv sync', then turn it back on.
"@ -ForegroundColor Yellow
  Die 'dependency install incomplete'
}

if ($Backend -eq 'rocm') {
  Say 'verifying the ROCm PyTorch runtime'
  uv run --no-sync python tests/rocm_smoke_test.py
  if ($LASTEXITCODE -ne 0) { Die 'ROCm PyTorch smoke test failed' }
}

# 6. shortcuts (best-effort — never abort the install over this) -------------
# Always lands an 'Anima LoRA GUI.lnk' in the install dir (policy-proof), and a
# desktop copy when login/OneDrive policy allows it.
Say 'creating shortcuts (Anima LoRA GUI)'
# The environment was synced above. A flagless uv run would sync it again with
# the default CUDA group and replace a validated ROCm install (GH #105).
try {
  uv run --no-sync python tasks.py gui-shortcut
  if ($LASTEXITCODE -ne 0) { throw "gui-shortcut exited $LASTEXITCODE" }
} catch {
  Say "shortcut skipped; find 'Anima LoRA GUI.lnk' in $Dir or run: uv run --no-sync python tasks.py gui-shortcut"
}

Write-Host ""
Write-Host "[OK] installed to $Dir\" -ForegroundColor Green
Write-Host @"

The GUI is opening now. To finish setup from inside it:
  - sign in to Hugging Face (built into the GUI now — no terminal step)
  - use the Models dialog to fetch the DiT + Qwen3 text encoder + VAE

Re-launch later from the "Anima LoRA GUI" desktop shortcut,
or run:  cd $Dir; python tasks.py gui

Update later with:  python tasks.py update
"@

if ($CondaActive) {
  Write-Host @"

NOTE: a Conda environment was active. This installer ran clean by ignoring it,
but a NEW terminal with Conda active will fail to launch the GUI
('DLL load failed while importing QtCore'). Run 'conda deactivate' first,
or launch from the desktop shortcut (which does not inherit Conda's PATH).
"@ -ForegroundColor Yellow
}

# 7. launch the GUI (best-effort, detached — never abort a finished install) -
# Start-Process so the installer returns immediately with the message above
# visible; a launch failure on a headless box just falls back to the shortcut.
Say 'launching the Anima LoRA GUI'
try {
  Start-Process -FilePath 'uv' -ArgumentList 'run', '--no-sync', 'python', 'tasks.py', 'gui' -WorkingDirectory (Resolve-Path '.').Path -WindowStyle Hidden
} catch {
  Say 'GUI launch skipped; start it later with: python tasks.py gui'
}
