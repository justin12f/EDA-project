SHELL := /bin/sh
COMPOSE := docker compose -f infra/docker-compose.yml

.PHONY: help setup dev down migrate reset test test-engine test-api test-web \
        web-dev api-dev worker-dev check-config fmt lint

help:
	@echo "Lumen"
	@echo ""
	@echo "  make setup          install every dependency (engine, api, web)"
	@echo "  make dev            supabase + redis + api + worker + web"
	@echo "  make migrate        apply supabase/migrations"
	@echo "  make check-config   show which providers this .env actually selects"
	@echo "  make test           every suite"
	@echo "  make down           stop everything"
	@echo ""
	@echo "No API key is needed. With none set, agents run the deterministic"
	@echo "MockProvider: real profiling, real proposals, locally generated wording."

# ── setup ────────────────────────────────────────────────────────────────────

setup:
	uv sync --directory engine --extra dev --extra embeddings
	uv sync --directory services/api --extra dev
	cd apps/web && npm install
	@test -f .env || (cp .env.example .env && echo "Created .env — fill in the Supabase block.")

# ── run ──────────────────────────────────────────────────────────────────────

dev:
	supabase start
	$(COMPOSE) up -d redis
	@echo ""
	@echo "  Supabase Studio  http://localhost:54323"
	@echo "  API              http://localhost:8000/docs"
	@echo "  Web              http://localhost:3000"
	@echo ""
	@echo "  Now run, in three terminals:  make api-dev / make worker-dev / make web-dev"

api-dev:
	uv run --directory services/api uvicorn lumen_api.main:app --reload --port 8000

worker-dev:
	uv run --directory services/worker arq lumen_worker.main.WorkerSettings

web-dev:
	cd apps/web && npm run dev

down:
	-$(COMPOSE) down
	-supabase stop

# ── database ─────────────────────────────────────────────────────────────────

migrate:
	supabase db push

reset:
	supabase db reset

# ── verification ─────────────────────────────────────────────────────────────

check-config:
	@uv run --directory services/api python -c "\
from lumen_api.settings import get_settings; s = get_settings();\
print('environment      ', s.environment);\
print('llm mode         ', s.llm_mode, '->', s.resolved_llm_mode);\
print('anthropic key    ', 'set' if s.has_anthropic else 'not set');\
print('groq key         ', 'set' if s.has_groq else 'not set');\
print('embeddings       ', s.embedding_provider, s.embedding_model, f'({s.embedding_dimensions}d)');\
print('supabase url     ', s.supabase_url);\
print('database         ', s.database_url.split('@')[-1] if '@' in s.database_url else s.database_url)"

test: test-engine test-api test-web

test-engine:
	uv run --directory engine pytest tests -q

test-api:
	uv run --directory services/api pytest tests -q

test-web:
	cd apps/web && npm test

fmt:
	uv run --directory engine ruff format src tests
	uv run --directory services/api ruff format src tests
	cd apps/web && npm run format

lint:
	uv run --directory engine ruff check src tests
	uv run --directory services/api ruff check src tests
	cd apps/web && npm run typecheck
