# ============================================================
# Trading Platform — Makefile
# One-command operations for dev, training, and production
# ============================================================

.PHONY: help install train train-full train-ai train-ai-quick train-ai-full download-data run dev deploy stop logs retrain test lint

# Default
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Local Development ──

install:  ## Install all dependencies
	pip install -r requirements.txt

train:  ## Train AI model (5 default stocks, fast ~30s)
	PYTHONPATH=. python3 scripts/train_alpha_model.py

train-full:  ## Train AI model on ALL 500 NSE stocks (~10min)
	TRAIN_SYMBOLS=all PYTHONPATH=. python3 scripts/train_alpha_model.py

# ── AI Auto-Train Pipeline ──

download-data:  ## Download market data from best available source
	PYTHONPATH=. python3 scripts/auto_train_all.py --data-only

train-ai:  ## Train ALL AI models — quick mode (5 stocks, 10 epochs, ~5min)
	PYTHONPATH=. python3 scripts/auto_train_all.py --quick

train-ai-full:  ## Train ALL AI models — full mode (70 stocks, 50 epochs, ~30min)
	PYTHONPATH=. python3 scripts/auto_train_all.py --full

train-ai-quick:  ## Train specific models fast: make train-ai-quick MODELS=lstm,transformer
	PYTHONPATH=. python3 scripts/auto_train_all.py --quick --models $(MODELS)

run:  ## Start backend API server
	PYTHONPATH=. uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

dev:  ## Start both backend + frontend (parallel)
	@echo "Starting backend..."
	PYTHONPATH=. uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload &
	@echo "Starting frontend..."
	cd trading-ui && npm run dev

test:  ## Run tests
	PYTHONPATH=. pytest tests/ -v

lint:  ## Run linter
	ruff check src/ scripts/

# ── Production (Docker) ──

deploy:  ## Deploy everything (auto-trains + starts all services)
	cd deploy/docker && docker-compose up -d --build

stop:  ## Stop all services
	cd deploy/docker && docker-compose down

logs:  ## View API logs (follow)
	cd deploy/docker && docker-compose logs -f api

retrain:  ## Force retrain in production (inside running container)
	cd deploy/docker && docker-compose exec api python scripts/train_alpha_model.py

status:  ## Check service health
	@echo "API:      $$(curl -s http://localhost:8000/health | head -c 100)"
	@echo "Frontend: $$(curl -s http://localhost:3000 | head -c 50 && echo ' OK' || echo ' DOWN')"
