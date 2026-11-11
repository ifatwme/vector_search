#!/usr/bin/env bash
# Hugging Face Space entrypoint: Weaviate + llama.cpp in the background, restore the
# prebuilt index, then Streamlit in the foreground on port 7860.
set -euo pipefail

export PERSISTENCE_DATA_PATH=/tmp/weaviate AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
       DEFAULT_VECTORIZER_MODULE=none ENABLE_MODULES="" CLUSTER_HOSTNAME=node1 \
       DISABLE_TELEMETRY=true LIMIT_RESOURCES=true GOMEMLIMIT=1500MiB
weaviate --host 127.0.0.1 --port 8080 --scheme http > /tmp/weaviate.log 2>&1 &

/app/llama-server -m "$HOME/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
    --host 127.0.0.1 --port 8081 --alias qwen2.5-1.5b-instruct \
    --ctx-size 4096 --parallel 1 --threads 2 --no-webui > /tmp/llm.log 2>&1 &

for _ in $(seq 1 120); do
    curl -fs http://127.0.0.1:8080/v1/.well-known/ready > /dev/null && break
    sleep 1
done
vector-search restore "$HOME/demo-data"

exec streamlit run app.py
