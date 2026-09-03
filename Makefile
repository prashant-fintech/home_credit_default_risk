.PHONY: install lint test train serve azure-provision azure-upload dbx-validate dbx-deploy dbx-run

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

# --- Azure / Databricks track (see docs/azure-databricks.md) ---------------
azure-provision:
	bash infra/azure/provision.sh

azure-upload:
	bash infra/azure/upload_raw.sh

# Bundle targets use the defaults in databricks/databricks.yml; override with
#   make dbx-deploy DBX_VARS='--var="raw_path=/Volumes/workspace/default/raw"'
DBX_VARS ?=

dbx-validate:
	cd databricks && databricks bundle validate $(DBX_VARS)

dbx-deploy:
	cd databricks && databricks bundle deploy -t dev $(DBX_VARS)

dbx-run:
	cd databricks && databricks bundle run -t dev home_credit_medallion $(DBX_VARS)
