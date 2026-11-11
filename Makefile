# Common tasks. `make help` lists them.
COMPOSE ?= docker compose
SHARDS  ?= $(or $(INGEST_SHARDS),0-4,143)

.PHONY: help install test lint build up down ps logs ingest ingest-all evaluate doctor scan

help:            ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:         ## local dev environment (uv + CPU torch)
	uv sync --frozen

test:            ## offline unit tests
	uv run pytest -q

lint:            ## ruff
	uv run ruff check .

build:           ## build the app image
	$(COMPOSE) build app

up:              ## start weaviate + llm + app (+ web on the server)
	$(COMPOSE) up -d

down:            ## stop the stack (data volumes are kept)
	$(COMPOSE) down

ps:              ## container status
	$(COMPOSE) ps

logs:            ## follow logs
	$(COMPOSE) logs -f --tail=100

scan:            ## dataset quality report
	$(COMPOSE) run --rm cli scan

ingest:          ## ingest the light subset (SHARDS=0-4,143)
	$(COMPOSE) run --rm cli ingest --shards $(SHARDS)

ingest-all:      ## ingest all 70k products (several CPU hours, resumable)
	$(COMPOSE) run --rm cli ingest

evaluate:        ## retrieval recall@k on dataset/eval
	$(COMPOSE) run --rm cli evaluate

doctor:          ## check weaviate / llm / data
	$(COMPOSE) run --rm cli doctor
