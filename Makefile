.PHONY: help install test backend-run frontend-run lint format clean

help:
	@echo "MedShield-FL Makefile Commands:"
	@echo "  make install       Install project dependencies"
	@echo "  make test          Run pytest suite"
	@echo "  make backend-run   Start backend FastAPI server"
	@echo "  make frontend-run  Start frontend Vite development server"
	@echo "  make lint          Run linter checks"
	@echo "  make clean         Clean temporary files and caches"

install:
	pip install -e .

test:
	pytest tests

backend-run:
	uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

frontend-run:
	cd frontend && npm run dev

lint:
	ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
