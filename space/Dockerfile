# Single-container demo for a free Hugging Face CPU Space (2 vCPU / 16 GB).
# Same code as the Compose deployment; services are started by start.sh.
FROM cr.weaviate.io/semitechnologies/weaviate:1.40.5 AS weaviate
FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

# llama.cpp CPU server image (Ubuntu) as the base: it provides /app/llama-server.
FROM ghcr.io/ggml-org/llama.cpp:server@sha256:086942d6ac097c894d9d5bbaf8085c53af85e113fbb9d6d61b3748863c86aa5f

USER root
# Spaces run the container as uid 1000; the Ubuntu base already has a user with that id.
RUN (userdel -r ubuntu 2>/dev/null || true) && useradd -m -u 1000 user
COPY --from=weaviate /bin/weaviate /usr/local/bin/weaviate
COPY --from=uv /uv /usr/local/bin/uv

USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    UV_PYTHON_INSTALL_DIR=/home/user/.python UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 \
    LD_LIBRARY_PATH=/app \
    PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    CACHE_DIR=/home/user/models IMAGE_DIR=/home/user/data/images REPORTS_DIR=/tmp/reports \
    CHAT_BACKEND=llm LLM_URL=http://127.0.0.1:8081/v1 LLM_MODEL=qwen2.5-1.5b-instruct \
    APP_NOTICE="نسخه نمایشی: ۲٬۶۰۵ کالا از دیجی‌کالا · مدل‌ها روی CPU رایگان اجرا می‌شوند، پاسخ دستیار ممکن است تا یک دقیقه طول بکشد."
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --python 3.12
COPY --chown=user src ./src
RUN uv sync --frozen --no-dev --no-editable
COPY --chown=user app.py start.sh ./
COPY --chown=user streamlit.toml .streamlit/config.toml

# Bake models and the demo index into the image so a waking Space needs no downloads.
RUN python - <<'PY'
import io
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from vector_search.config import Settings
from vector_search.embeddings import Embedder

embedder = Embedder(Settings.from_env())
embedder.encode_text("سلام")
buffer = io.BytesIO()
Image.new("RGB", (32, 32)).save(buffer, format="PNG")
embedder.encode_image(buffer.getvalue())
hf_hub_download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                local_dir="/home/user/models/llm")
snapshot_download("ifatwme/vector-search-demo-data", repo_type="dataset",
                  local_dir="/home/user/demo-data", allow_patterns=["*.gz", "*.tar"])
PY

EXPOSE 7860
ENTRYPOINT []
CMD ["./start.sh"]
