# syntax=docker/dockerfile:1
# Initialize device type args
# use build args in the docker build command with --build-arg="BUILDARG=true"
ARG USE_CUDA=false
ARG USE_OLLAMA=false
ARG USE_SLIM=false
ARG USE_PERMISSION_HARDENING=false
ARG USE_CERTS=true
# Tested with cu117 for CUDA 11 and cu121 for CUDA 12 (default)
ARG USE_CUDA_VER=cu128
# any sentence transformer model; models to use can be found at https://huggingface.co/models?library=sentence-transformers
# Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
# for better performance and multilangauge support use "intfloat/multilingual-e5-large" (~2.5GB) or "intfloat/multilingual-e5-base" (~1.5GB)
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
ARG USE_CERTS

# Set Node.js options (heap limit Allocation failed - JavaScript heap out of memory)
ENV NODE_OPTIONS="--max-old-space-size=4096"

WORKDIR /app

# to store git revision in build
RUN apk add --no-cache git

# Corporate CA (conditionally applied when USE_CERTS=true)
RUN if [ "$USE_CERTS" = "true" ]; then \
        mkdir -p /usr/local/share/ca-certificates && \
        if [ -f certs/fw.cer ]; then \
            cp certs/fw.cer /usr/local/share/ca-certificates/fw.crt && \
            ls -l /usr/local/share/ca-certificates/fw.crt; \
        fi; \
    fi

# Conditionally set NODE_EXTRA_CA_CERTS when USE_CERTS=true
RUN if [ "$USE_CERTS" = "true" ]; then \
        echo "export NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/fw.crt" >> /etc/profile; \
    fi

# Copy pre-downloaded binaries
COPY local-binaries/ /tmp/onnx-cache/

COPY package.json package-lock.json ./

# Install with pre-downloaded ONNX runtime and missing peer dependencies
RUN npm config set maxsockets 1 && \
    npm config set fetch-retries 3 && \
    npm config set fetch-retry-maxtimeout 60000 && \
    npm config set fetch-retry-mintimeout 10000 && \
    npm install --legacy-peer-deps --ignore-scripts && \
    npm install y-protocols prosemirror-state prosemirror-view prosemirror-model prosemirror-transform prosemirror-commands prosemirror-keymap prosemirror-history --legacy-peer-deps --ignore-scripts && \
    # Manually extract the ONNX runtime
    if [ -f /tmp/onnx-cache/onnxruntime-linux-x64-gpu-1.20.1.tgz ]; then \
        mkdir -p node_modules/onnxruntime-node/bin/napi-v3/linux/x64/ && \
        cd /tmp && \
        tar -xzf /tmp/onnx-cache/onnxruntime-linux-x64-gpu-1.20.1.tgz && \
        find . -name "libonnxruntime*" -type f -exec cp {} /app/node_modules/onnxruntime-node/bin/napi-v3/linux/x64/ \; 2>/dev/null || true; \
    fi

COPY . .
ENV APP_BUILD_HASH=${BUILD_HASH}
RUN npm run build

######## WebUI backend ########
FROM python:3.11-slim-bookworm AS base

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
ARG USE_CERTS

# Disable TLS verification for these hosts by marking them as trusted (when USE_CERTS=true)
RUN if [ "$USE_CERTS" = "true" ]; then \
        printf "[global]\n\
trusted-host = pypi.org\n\
\tfiles.pythonhosted.org\n\
\tpypi.python.org\n\
\tdownload.pytorch.org\n" > /etc/pip.conf; \
    fi

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

# Conditionally set HuggingFace cert configs when USE_CERTS=true
RUN if [ "$USE_CERTS" = "true" ]; then \
        echo "export HF_HUB_CA_CERTS_PATH=/etc/ssl/certs/ca-certificates.crt" >> /etc/environment; \
        echo "export HF_HUB_DISABLE_SSL_VERIFICATION=1" >> /etc/environment; \
    fi

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

RUN mkdir -p $HOME/.cache/chroma
RUN echo -n 00000000-0000-0000-0000-000000000000 > $HOME/.cache/chroma/telemetry_user_id

# Make sure the user has access to the app and root directory
RUN chown -R $UID:$GID /app $HOME

# Corporate CA (conditionally applied when USE_CERTS=true)
RUN if [ "$USE_CERTS" = "true" ]; then \
        mkdir -p /usr/local/share/ca-certificates && \
        if [ -f certs/fw.cer ]; then \
            cp certs/fw.cer /usr/local/share/ca-certificates/fw.crt; \
        fi; \
    fi

# Install common system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git build-essential pandoc gcc netcat-openbsd curl jq \
    python3-dev \
    ffmpeg libsm6 libxext6 \
    ca-certificates && \
    if [ "$USE_CERTS" = "true" ]; then \
        update-ca-certificates; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Conditionally set certificate environment variables when USE_CERTS=true
RUN if [ "$USE_CERTS" = "true" ]; then \
        echo "export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" >> /etc/environment; \
        echo "export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" >> /etc/environment; \
    fi

# install python dependencies
COPY --chown=$UID:$GID ./backend/requirements.txt ./requirements.txt

# Conditionally set PIP_NO_VERIFY_CERTS when USE_CERTS=true
RUN if [ "$USE_CERTS" = "true" ]; then \
        echo "export PIP_NO_VERIFY_CERTS=1" >> /etc/environment; \
    fi

RUN if [ "$USE_CERTS" = "true" ]; then \
        pip3 install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org --no-cache-dir uv; \
    else \
        pip3 install --no-cache-dir uv; \
    fi && \
    if [ "$USE_CUDA" = "true" ]; then \
    # If you use CUDA the whisper and embedding model will be downloaded on first use
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$USE_CUDA_DOCKER_VER --no-cache-dir && \
    uv pip install --system -r requirements.txt --no-cache-dir && \
    python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['RAG_EMBEDDING_MODEL'], device='cpu')" && \
    python -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8', download_root=os.environ['WHISPER_MODEL_DIR'])"; \
    python -c "import os; import tiktoken; tiktoken.get_encoding(os.environ['TIKTOKEN_ENCODING_NAME'])"; \
    else \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir && \
    uv pip install --system -r requirements.txt --no-cache-dir && \
    if [ "$USE_SLIM" != "true" ]; then \
    python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['RAG_EMBEDDING_MODEL'], device='cpu')" && \
    python -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8', download_root=os.environ['WHISPER_MODEL_DIR'])"; \
    python -c "import os; import tiktoken; tiktoken.get_encoding(os.environ['TIKTOKEN_ENCODING_NAME'])"; \
    fi; \
    fi; \
    mkdir -p /app/backend/data && chown -R $UID:$GID /app/backend/data/ && \
    rm -rf /var/lib/apt/lists/*;

# Install Ollama if requested
RUN if [ "$USE_OLLAMA" = "true" ]; then \
    date +%s > /tmp/ollama_build_hash && \
    echo "Cache broken at timestamp: `cat /tmp/ollama_build_hash`" && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*; \
    fi

# copy embedding weight from build
# RUN mkdir -p /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2
# COPY --from=build /app/onnx /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx

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

CMD [ "bash", "start.sh"]