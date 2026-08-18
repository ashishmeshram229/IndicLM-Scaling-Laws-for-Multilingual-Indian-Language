.PHONY: install dev-install lint typecheck test doctor clean

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests

typecheck:
	mypy src

test:
	pytest -q

doctor:
	python -m indiclm.cli.main doctor

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
