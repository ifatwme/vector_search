# syntax=docker/dockerfile:1
# One image for the Streamlit UI and the `vector-search` CLI (ingest/evaluate/doctor).
# CPU-only PyTorch keeps it ~1.5 GB instead of ~6 GB with CUDA.

FROM python:3.12.16-slim-trixie AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# Dependencies first so code edits do not invalidate the (large) dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable

FROM python:3.12.16-slim-trixie
ARG UID=10001
RUN groupadd --gid ${UID} app && useradd --uid ${UID} --gid ${UID} --no-create-home \
        --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data/images /data/models /data/reports /app \
    && chown -R ${UID}:${UID} /data
COPY --from=build /opt/venv /opt/venv
COPY app.py /app/app.py
COPY docker/streamlit.toml /app/.streamlit/config.toml
WORKDIR /app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp \
    PARSED_DIR=/app/dataset/parsed IMAGE_DIR=/data/images CACHE_DIR=/data/models \
    REPORTS_DIR=/data/reports HF_HOME=/data/models/hf HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false
USER ${UID}:${UID}
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status != 200)"
CMD ["streamlit", "run", "app.py"]
