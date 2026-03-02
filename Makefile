.PHONY: help install install-dev test test-unit test-integration test-validation test-all lint format clean reproduce reproduce-quick docker-build docker-run docs

PYTHON := python
PIP := pip
PYTEST := pytest

help:
	@echo "Phase-Transition-Hardness - Makefile Commands"
	@echo "=============================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install       Install package"
	@echo "  make install-dev   Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run quick tests"
	@echo "  make test-unit     Run unit tests only"
	@echo "  make test-integration  Run integration tests"
	@echo "  make test-validation   Run manuscript validation tests"
	@echo "  make test-all      Run complete test suite"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run linters (ruff, mypy)"
	@echo "  make format        Format code (black, isort)"
	@echo ""
	@echo "Reproduction:"
	@echo "  make reproduce     Full manuscript reproduction (~4.5 hours)"
	@echo "  make reproduce-quick   Quick validation (~5 minutes)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          Generate documentation"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         Remove build artifacts"
	@echo "  make clean-results Remove experiment results"
	@echo ""

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTEST) tests/unit -v --tb=short -x

test-unit:
	$(PYTEST) tests/unit -v --tb=short

test-integration:
	$(PYTEST) tests/integration -v --tb=short

test-validation:
	$(PYTEST) tests/validation -v --tb=short

test-all:
	$(PYTEST) tests/ -v --tb=short --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/ experiments/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ experiments/ --line-length 100
	isort src/ tests/ experiments/ --profile black

reproduce:
	bash scripts/reproduce_all_figures.sh results/manuscript

reproduce-quick:
	bash scripts/quick_test.sh

docker-build:
	docker build -t phase-transition:latest .

docker-run:
	docker run -v $(PWD)/results:/app/results phase-transition:latest

docs:
	@echo "Documentation available in docs/ directory"
	@echo "  - docs/API.md"
	@echo "  - docs/ARCHITECTURE.md"
	@echo "  - docs/REPRODUCIBILITY.md"
	@echo "  - docs/MATHEMATICAL_PROOFS.md"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-results:
	rm -rf results/
	rm -rf figures/
	mkdir -p results figures

benchmark:
	$(PYTHON) -m pytest tests/unit --benchmark-only
