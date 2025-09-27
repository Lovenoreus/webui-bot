# syntax=docker/dockerfile:1
# Initialize device type args
# use build args in the docker build command with --build-arg="BUILDARG=true"
ARG USE_CUDA=false
ARG USE_OLLAMA=false
ARG USE_SLIM=false
ARG USE_PERMISSION_HARDENING=false
# Tested with cu117 for CUDA 11 and cu121 for CUDA 12 (default)
ARG USE_CUDA_VER=cu128
# any sentence transformer model; models to use can be found at https://huggingface.co/models?library=sentence-transformers
# Leaderboard: https://huggingface.co/spaces/mteb/leaderboard 
# IMPORTANT: If you change the embedding model (sentence-transformers/all-MiniLM-L6-v2) and vice versa, you aren't able to use RAG Chat with your previous documents loaded in the WebUI! You need to re-embed them.
#ARG USE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG USE_RERANKING_MODEL=""
# Tiktoken encoding name; models to use can be found at https://huggingface.co/models?library=tiktoken
ARG USE_TIKTOKEN_ENCODING_NAME="cl100k_base"
ARG BUILD_HASH=dev-build
# Override at your own risk - non-root configurations are untested
ARG UID=0
ARG GID=0
######## WebUI frontend ########
FROM --platform=$BUILDPLATFORM node:22-alpine3.20 AS build
ARG BUILD_HASH
# Increase Node heap (helps large builds)
ENV NODE_OPTIONS="--max-old-space-size=4096"
ENV OFFLINE_MODE=true

WORKDIR /app
# Corporate CA (kept from your original)
COPY certs/fw.cer /usr/local/share/ca-certificates/fw.crt
RUN ls -l /usr/local/share/ca-certificates/fw.crt

ENV NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/fw.crt
# Optional fallback for npm directly (just in case)
# RUN npm config set cafile /usr/local/share/ca-certificates/fw.cer \
#  && npm config set strict-ssl true
# Tools you previously considered useful for build metadata
#RUN apk add --no-cache git
# Prevent onnxruntime-node postinstall from attempting network fetches
ENV ONNXRUNTIME_NODE_INSTALL=skip
# --- Pre-downloaded ONNX binaries cache ---
# Make sure you have local-binaries/onnxruntime-linux-x64-gpu-1.20.1.tgz in build context
COPY local-binaries/ /tmp/onnx-cache/
# Install deps with conservative npm settings (network-friendly), skipping scripts
COPY package.json package-lock.json ./
# RUN npm config set maxsockets 1 \
#  && npm config set fetch-retries 3 \
#  && npm config set fetch-retry-maxtimeout 60000 \
#  && npm config set fetch-retry-mintimeout 10000 \
#  && npm install --legacy-peer-deps --ignore-scripts --verbose \
#  && npm install y-protocols prosemirror-state prosemirror-view prosemirror-model prosemirror-transform prosemirror-commands prosemirror-keymap prosemirror-history --legacy-peer-deps --ignore-scripts --verbose
RUN npm set maxsockets 1
#RUN npm set jobs 1
RUN npm config set maxsockets 1
RUN npm config set fetch-retries 3
RUN npm config set fetch-retry-maxtimeout 60000
RUN npm config set fetch-retry-mintimeout 10000
RUN npm cache clean --force
RUN npm install --legacy-peer-deps --ignore-scripts --verbose
RUN npm install y-protocols prosemirror-state prosemirror-view prosemirror-model prosemirror-transform prosemirror-commands prosemirror-keymap prosemirror-history --legacy-peer-deps --ignore-scripts --verbose
# Manually extract/copy ONNX runtime libs into onnxruntime-node expected path
RUN if [ -f /tmp/onnx-cache/onnxruntime-linux-x64-gpu-1.20.1.tgz ]; then \
      mkdir -p node_modules/onnxruntime-node/bin/napi-v3/linux/x64/ && \
      cd /tmp && \
      tar -xzf /tmp/onnx-cache/onnxruntime-linux-x64-gpu-1.20.1.tgz && \
      find . -name "libonnxruntime*" -type f -exec cp {} /app/node_modules/onnxruntime-node/bin/napi-v3/linux/x64/ \; 2>/dev/null || true; \
    fi
# App source + build
COPY . .
ENV APP_BUILD_HASH=${BUILD_HASH}
RUN npm run build
######## WebUI backend ########
FROM python:3.11-slim-bookworm AS base
# Disable TLS verification for these hosts by marking them as trusted.
# (This affects both pip and `uv pip`.)
RUN printf "[global]\n\
trusted-host = pypi.org\n\
\tfiles.pythonhosted.org\n\
\tpypi.python.org\n\
\tdownload.pytorch.org\n" > /etc/pip.conf
# Use args
ARG USE_CUDA
ARG USE_OLLAMA
ARG USE_CUDA_VER
ARG USE_SLIM
ARG USE_PERMISSION_HARDENING
ARG USE_EMBEDDING_MODEL
ARG USE_RERANKING_MODEL
ARG UID
ARG GID
## Basis ##
ENV ENV=prod \
    PORT=8080 \
    # pass build args to the build
    USE_OLLAMA_DOCKER=${USE_OLLAMA} \
    USE_CUDA_DOCKER=${USE_CUDA} \
    USE_SLIM_DOCKER=${USE_SLIM} \
    USE_CUDA_DOCKER_VER=${USE_CUDA_VER} \
    USE_EMBEDDING_MODEL_DOCKER=${USE_EMBEDDING_MODEL} \
    USE_RERANKING_MODEL_DOCKER=${USE_RERANKING_MODEL}
## Basis URL Config ##
ENV OLLAMA_BASE_URL="/ollama" \
    OPENAI_API_BASE_URL=""
## API Key and Security Config ##
ENV OPENAI_API_KEY="" \
    WEBUI_SECRET_KEY="" \
    SCARF_NO_ANALYTICS=true \
    DO_NOT_TRACK=true \
    ANONYMIZED_TELEMETRY=false
#### Other models #########################################################
## whisper TTS model settings ##
ENV WHISPER_MODEL="base" \
    WHISPER_MODEL_DIR="/app/backend/data/cache/whisper/models"
## RAG Embedding model settings ##
ENV RAG_EMBEDDING_MODEL="$USE_EMBEDDING_MODEL_DOCKER" \
    RAG_RERANKING_MODEL="$USE_RERANKING_MODEL_DOCKER" \
    SENTENCE_TRANSFORMERS_HOME="/app/backend/data/cache/embedding/models"
## Tiktoken model settings ##
ENV TIKTOKEN_ENCODING_NAME="cl100k_base" \
    TIKTOKEN_CACHE_DIR="/app/backend/data/cache/tiktoken"
## Hugging Face download cache ##
ENV HF_HOME="/app/backend/data/cache/embedding/models"
ENV HF_HUB_CA_CERTS_PATH=/etc/ssl/certs/ca-certificates.crt
ENV HF_HUB_DISABLE_SSL_VERIFICATION=1
## Torch Extensions ##
# ENV TORCH_EXTENSIONS_DIR="/.cache/torch_extensions"
#### Other models ##########################################################
WORKDIR /app/backend
ENV HOME=/root
# Create user and group if not root
RUN if [ $UID -ne 0 ]; then \
    if [ $GID -ne 0 ]; then \
    addgroup --gid $GID app; \
    fi; \
    adduser --uid $UID --gid $GID --home $HOME --disabled-password --no-create-home app; \
    fi
RUN mkdir -p $HOME/.cache/chroma \
 && echo -n 00000000-0000-0000-0000-000000000000 > $HOME/.cache/chroma/telemetry_user_id
# Make sure the user has access to the app and root directory
RUN chown -R $UID:$GID /app $HOME
# Corporate CA
COPY certs/fw.cer /usr/local/share/ca-certificates/fw.crt
# System deps + CA refresh
# RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates && \
#     apt-get install -y --no-install-recommends \
#     git build-essential pandoc gcc netcat-openbsd curl jq \
#     python3-dev \
#     ffmpeg libsm6 libxext6 \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update

RUN apt-get install -y ca-certificates

RUN update-ca-certificates

RUN apt-get install -y --no-install-recommends \
    git build-essential pandoc gcc netcat-openbsd curl jq \
    python3-dev \
    ffmpeg libsm6 libxext6

RUN rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# install python dependencies
COPY --chown=$UID:$GID ./backend/requirements.txt ./requirements.txt
ENV PIP_NO_VERIFY_CERTS=1         
#RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --index-url http://pypi.org/simple torch
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org --no-cache-dir uv
#RUN pip3 install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org pip-system-certs
RUN pip3 install --no-cache-dir uv && \
    if [ "$USE_CUDA" = "true" ]; then \
      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$USE_CUDA_DOCKER_VER --no-cache-dir && \
      uv pip install --system -r requirements.txt --no-cache-dir && \
      python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['RAG_EMBEDDING_MODEL'], device='cpu')" && \
      python -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8', download_root=os.environ['WHISPER_MODEL_DIR'])" && \
      python -c "import os; import tiktoken; tiktoken.get_encoding(os.environ['TIKTOKEN_ENCODING_NAME'])"; \
    else \
      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir && \
      uv pip install --system -r requirements.txt --no-cache-dir && \
      if [ "$USE_SLIM" != "true" ]; then \
        python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['RAG_EMBEDDING_MODEL'], device='cpu')" && \
        python -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8', download_root=os.environ['WHISPER_MODEL_DIR'])" && \
        python -c "import os; import tiktoken; tiktoken.get_encoding(os.environ['TIKTOKEN_ENCODING_NAME'])"; \
      fi; \
    fi && \
    mkdir -p /app/backend/data && chown -R $UID:$GID /app/backend/data/
# Install Ollama if requested
RUN if [ "$USE_OLLAMA" = "true" ]; then \
    date +%s > /tmp/ollama_build_hash && \
    echo "Cache broken at timestamp: `cat /tmp/ollama_build_hash`" && \
    curl -fsSL https://ollama.com/install.sh | sh; \
    fi
# copy built frontend files
COPY --chown=$UID:$GID --from=build /app/build /app/build
COPY --chown=$UID:$GID --from=build /app/CHANGELOG.md /app/CHANGELOG.md
COPY --chown=$UID:$GID --from=build /app/package.json /app/package.json
# copy backend files
COPY --chown=$UID:$GID ./backend .
EXPOSE 8080
HEALTHCHECK CMD curl --silent --fail http://localhost:${PORT:-8080}/health | jq -ne 'input.status == true' || exit 1
# Minimal, atomic permission hardening for OpenShift (arbitrary UID):
# - Group 0 owns /app and /root
# - Directories are group-writable and have SGID so new files inherit GID 0
RUN if [ "$USE_PERMISSION_HARDENING" = "true" ]; then \
    set -eux; \
    chgrp -R 0 /app /root || true; \
    chmod -R g+rwX /app /root || true; \
    find /app -type d -exec chmod g+s {} + || true; \
    find /root -type d -exec chmod g+s {} + || true; \
    fi
USER $UID:$GID
ARG BUILD_HASH
ENV WEBUI_BUILD_VERSION=${BUILD_HASH}
ENV DOCKER=true
ENV OFFLINE_MODE=true
CMD [ "bash", "start.sh"]