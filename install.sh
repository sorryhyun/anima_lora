#!/usr/bin/env sh
# anima_lora bootstrap installer (Linux / macOS).
#
#   curl -LsSf https://raw.githubusercontent.com/sorryhyun/anima_lora/main/install.sh | sh
#
# Installs uv if missing, guides the CUDA 13.2 toolkit install if missing,
# downloads the latest release tarball (no git required), seeds the update
# baseline so the first `make update` is clean, and runs `uv sync`. The
# resolve-latest / tarball / manifest logic mirrors scripts/update.py — keep
# the two in sync.
#
# Options (env vars, since args are awkward through a pipe):
#   ANIMA_VERSION=v1.4.0   install a specific tag        (default: latest)
#   ANIMA_DIR=./somewhere  target directory              (default: ./anima_lora)
#   ANIMA_SKIP_CUDA=1      skip the CUDA 13.2 toolkit install/check
# Or with explicit args:  sh -s -- [version] [dir]
set -eu

REPO="sorryhyun/anima_lora"
VERSION="${ANIMA_VERSION:-${1:-}}"
DIR="${ANIMA_DIR:-${2:-anima_lora}}"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v tar  >/dev/null 2>&1 || die "tar is required"

# 1. uv ----------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (https://astral.sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Make uv visible to this shell without a re-login.
  # shellcheck disable=SC1090
  [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv install failed; open a new shell and re-run"

# 1.5 CUDA 13.2 toolkit (required for torch.compile / Triton ptxas) ----------
# torch wheels bundle the CUDA *runtime*, but this repo's compile path
# (Triton → ptxas) needs the system toolkit. Done BEFORE the repo download so
# a reboot-then-rerun stays clean (CUDA is detected + skipped on the next run).
# Reads the prompt from /dev/tty because stdin is the piped script (curl | sh).
cuda_ok() {
  for n in nvcc /usr/local/cuda-13.2/bin/nvcc /usr/local/cuda/bin/nvcc; do
    { command -v "$n" >/dev/null 2>&1 || [ -x "$n" ]; } || continue
    "$n" --version 2>/dev/null | grep -qE 'release 13\.2([^0-9]|$)' && return 0
  done
  return 1
}
if [ -n "${ANIMA_SKIP_CUDA:-}" ]; then
  say "ANIMA_SKIP_CUDA set — skipping the CUDA 13.2 toolkit check"
elif cuda_ok; then
  say "CUDA 13.2 toolkit detected"
elif [ ! -r /dev/tty ]; then
  say "CUDA 13.2 not found and no terminal is available to drive the installer."
  say "Install it manually, then re-run this installer:"
  say "  https://developer.nvidia.com/cuda-13-2-0-download-archive"
  die "CUDA 13.2 toolkit required (set ANIMA_SKIP_CUDA=1 to bypass)"
else
  say "CUDA 13.2 toolkit not found — required for torch.compile / Triton (ptxas)"
  RUN_URL="https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_595.45.04_linux.run"
  cuda_run=$(mktemp)
  say "downloading the CUDA 13.2 installer (~4 GB) → $cuda_run"
  curl -LsSf "$RUN_URL" -o "$cuda_run" || die "CUDA installer download failed"
  say 'launching the CUDA installer with sudo — accept the EULA (deselect the Driver if yours is already ≥595)'
  sudo sh "$cuda_run" </dev/tty || say "installer exited non-zero — verifying anyway"
  printf 'Press Enter once CUDA has finished installing... '
  read -r _ </dev/tty
  rm -f "$cuda_run"
  cuda_ok || die "CUDA 13.2 still not detected; install from https://developer.nvidia.com/cuda-13-2-0-download-archive and re-run"
  say "CUDA 13.2 toolkit detected"
fi

# 2. resolve the release tag -------------------------------------------------
if [ -z "$VERSION" ]; then
  say "resolving latest release of $REPO"
  VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep -m1 '"tag_name"' \
    | sed -E 's/.*"tag_name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')
  [ -n "$VERSION" ] || die "could not resolve latest release tag from GitHub API"
fi
say "installing $REPO @ $VERSION → $DIR/"

[ -e "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ] && \
  die "$DIR/ already exists and is not empty — pass a different ANIMA_DIR"

# 3. download + extract ------------------------------------------------------
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
TARBALL="https://github.com/$REPO/archive/refs/tags/$VERSION.tar.gz"
say "downloading $TARBALL"
curl -LsSf "$TARBALL" -o "$TMP/release.tar.gz" || die "download failed"
tar -xzf "$TMP/release.tar.gz" -C "$TMP"
# GitHub source tarballs contain a single top-level dir (anima_lora-<tag>).
TOP=$(find "$TMP" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | head -n1)
[ -n "$TOP" ] || die "unexpected tarball layout"
mkdir -p "$DIR"
# move contents (including dotfiles) into the target dir
(cd "$TOP" && tar -cf - .) | (cd "$DIR" && tar -xf -)

cd "$DIR"

# 4. seed the update baseline (before uv sync, so .venv doesn't get hashed) ---
say "seeding update baseline (.anima_release.json)"
uv run --no-project python scripts/update.py --seed-manifest --version "$VERSION" \
  || say "manifest seed skipped (first \`make update\` will back up instead — harmless)"

# 5. dependencies ------------------------------------------------------------
say "running uv sync (this resolves torch + flash-attn; may take a while)"
uv sync

cat <<EOF

$(printf '\033[1;32m✓ installed to %s/\033[0m' "$DIR")

Next steps:
  cd $DIR
  make gui                 # sign in to Hugging Face + download models in the GUI
                           #   (Hugging Face auth is built into the GUI now)

Prefer the CLI? After signing in once (the GUI stores your HF token):
  make download-models     # DiT + Qwen3 text encoder + VAE into models/
  make lora                # start training

Update later with:  make update
EOF
