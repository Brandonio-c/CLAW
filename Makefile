.PHONY: install install-dev test format lint clean demo

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,llama,quantization]"

# Testing
test:
	pytest

test-cov:
	pytest --cov=claw --cov-report=html

# Code quality
format:
	black claw/ tests/ examples/
	isort claw/ tests/ examples/

lint:
	mypy claw/
	black --check claw/ tests/ examples/
	isort --check-only claw/ tests/ examples/

# Development
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Demo
demo:
	python -m claw.cli.main demo-diplomacy --help

# Documentation
docs:
	@echo "Documentation generation not implemented yet"

# All checks
check: format lint test
	@echo "All checks passed!"
