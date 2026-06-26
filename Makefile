.PHONY: help install install-dev test test-cov lint format type-check clean run docker-build docker-run benchmark

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make lint          - Run linters (ruff)"
	@echo "  make format        - Format code (black, ruff)"
	@echo "  make type-check    - Run type checker (mypy)"
	@echo "  make clean         - Clean up build artifacts"
	@echo "  make run           - Run the API server"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make benchmark     - Run benchmarks"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

test:
	$(PYTHON) -m pytest tests/test_*.py -q -m "not slow and not benchmark"

test-cov:
	$(PYTHON) -m pytest tests/test_*.py -q -m "not slow and not benchmark" --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ tests/

format:
	black src/ tests/ benchmarks/
	ruff check --fix src/ tests/ benchmarks/

type-check:
	mypy src/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t vector-db:latest -f Dockerfile .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

benchmark:
	$(PYTHON) benchmarks/benchmark.py

pre-commit:
	pre-commit run --all-files
