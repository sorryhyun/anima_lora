# anima_lora runtime image — environment only, no model weights.
#
# torch 2.14 + cu132 and flash-attn 2.8.3 come from uv.lock (the torch wheels
# carry the CUDA runtime as nvidia-* packages). The CUDA 13.2 base adds only
# ptxas / headers for the torch.compile → Triton path, matching install.sh.
#
# Weights live on a volume at /app/models and are fetched on first start by
# docker/entrypoint.sh. Build with docker/build.sh (context = `git archive <ref>`).
FROM nvidia/cuda:13.2.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-13-2 cuda-cudart-dev-13-2 \
        build-essential git make curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV PATH=/usr/local/cuda/bin:$PATH \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_FROZEN=1 \
    ANIMA_HOME=/app \
    HF_HOME=/app/models/.cache/huggingface

WORKDIR /app

# Dependencies first so a code-only change reuses this (large) layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

ENV PATH=/app/.venv/bin:$PATH

COPY docker/entrypoint.sh /usr/local/bin/anima-entrypoint
RUN chmod +x /usr/local/bin/anima-entrypoint

VOLUME ["/app/models"]
ENTRYPOINT ["anima-entrypoint"]
CMD ["bash"]
