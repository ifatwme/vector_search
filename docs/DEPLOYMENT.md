# Deployment

The production instance runs on the VPS **Testilonia (65.109.162.75)** from
`/root/vector_search` and is served at **https://65.109.162.75:8443/**.

Ports 80/443 on this host belong to the `ai-robotics-trends` dashboard, so this project
uses **8443** and reuses that project's Let's Encrypt IP certificate
(`/root/ai-robotics-trends/deploy/tls`, renewed by `aitrends-cert-renew.timer`). Our
nginx reloads every 6 hours to pick up renewed certificates.

## First deployment

```bash
cd /root && git clone git@github-vector-search:ifatwme/vector_search.git && cd vector_search
cp .env.example .env
# enable the server override + TLS (uncomment these three lines in .env):
#   COMPOSE_FILE=docker-compose.yml:deploy/compose.server.yml
#   TLS_DIR=/root/ai-robotics-trends/deploy/tls
#   HTTPS_PORT=8443
docker compose build app
docker compose up -d                                  # weaviate, llm, app, web
docker compose run --rm cli ingest --shards 0-4,143   # light subset (2.6k products, ~14 min)
docker compose run --rm cli doctor
install -m 0644 deploy/systemd/vector-search.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable vector-search.service
```

Containers have `restart: unless-stopped`, so Docker restarts them after a reboot; the
systemd unit additionally runs `docker compose up -d` at boot in case containers were removed.

## Resource budget (4 vCPU, 7.6 GB RAM host)

| Service | mem limit | cpus | Typical use |
|---|---|---|---|
| app | 3 GB | 2 | ~60 MB idle, ~1.3 GB after the first search (CLIPfa loaded) |
| llm | 2.5 GB | 3 | ~1 GB (Qwen2.5-1.5B Q4_K_M, 4k context) |
| weaviate | 1.5 GB | 1.5 | ~100–300 MB for the light subset |
| web | 64 MB | 0.5 | ~8 MB |
| cli (jobs only) | 3 GB | 4 | ~1 GB while ingesting |

## Operations

| Task | Command |
|---|---|
| Status | `docker compose ps` |
| Logs | `docker compose logs -f app` (or `llm`, `weaviate`, `web`) |
| Health | `docker compose run --rm cli doctor` · `curl https://65.109.162.75:8443/healthz` |
| Add more products | `docker compose run --rm cli ingest --shards 5-20` |
| Ingest everything | `docker compose run --rm cli ingest` (70k products, ≈4–5 h on 4 cores; resumable) |
| Retrieval quality | `docker compose run --rm cli evaluate` |
| Last ingest report | `docker compose run --rm --entrypoint cat cli /data/reports/ingest-latest.json` |
| Restart | `docker compose restart app` |
| Stop (keep data) | `docker compose down` |

Ingestion is **idempotent and resumable**: interrupting it loses nothing; re-running skips
every product whose content and model fingerprint are already stored.

## Upgrading

```bash
cd /root/vector_search
git pull
docker compose build app
docker compose up -d            # recreates only changed services
```

* Changing `TEXT_MODEL`/`IMAGE_MODEL` or their revisions changes the fingerprint: run
  `ingest` again (it re-embeds in place); search refuses to mix vector spaces until then.
* Changing the LLM: set `LLM_HF_MODEL` (and `LLM_MODEL_NAME`) in `.env`, then
  `docker compose up -d llm`. The GGUF is downloaded into the `llm-models` volume.
* Weaviate upgrades: bump the tag in `docker-compose.yml` one minor version at a time and
  back up first.

## Backups

The only state that is expensive to rebuild is the Weaviate volume (embeddings) and the
image cache. Everything can be regenerated from `dataset/` with `ingest`.

```bash
docker compose stop weaviate
docker run --rm -v vector-search_weaviate-data:/data -v "$PWD":/backup alpine \
  tar czf /backup/weaviate-$(date +%F).tgz -C /data .
docker compose start weaviate
```

## Security

* Only nginx (8443) is public; Streamlit is bound to 127.0.0.1, Weaviate and the LLM are
  reachable only on the Compose network (anonymous Weaviate access is safe only because of that).
* All containers drop every Linux capability, use `no-new-privileges`, and have memory,
  CPU and PID limits; app, cli and web run read-only as non-root users.
* No secrets are needed. If you ever expose Weaviate, set `AUTHENTICATION_APIKEY_*` and
  `WEAVIATE_API_KEY`.
* The Google API key from the original code is in the public Git history: revoke it.

## Access from Iran

If the page does not open from Iran while `curl https://65.109.162.75:8443/healthz`
works on the server, the foreign IP is being filtered on the client side; use a VPN.
