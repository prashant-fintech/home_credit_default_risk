.PHONY: install lint test train serve

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests scripts
	mypy src

test:
	pytest tests/ -v --cov=src/home_credit --cov-report=term-missing

train:
	python scripts/train.py

predict:
	python scripts/predict.py

serve:
	uvicorn home_credit.serving.app:app --reload --host 0.0.0.0 --port 8000
