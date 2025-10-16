.PHONY: help install train test clean docker-build docker-up docker-down docker-logs lint format

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make train         - Train all models"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start Docker containers"
	@echo "  make docker-down   - Stop Docker containers"
	@echo "  make docker-logs   - View Docker logs"
	@echo "  make clean         - Clean up generated files"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

train:
	python train.py

test:
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/test_main.py tests/test_model.py -v

test-integration:
	pytest tests/test_integration.py -v -m integration

test-fast:
	pytest tests/ -v -m "not slow and not load"

test-coverage:
	pytest tests/ --cov=app --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint:
	flake8 app/ tests/ --max-line-length=100
	mypy app/ --ignore-missing-imports
	pylint app/ tests/

format:
	black app/ tests/ train.py
	isort app/ tests/ train.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-restart:
	docker-compose restart

run-api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-streamlit:
	streamlit run app/streamlit_app.py

mlflow-ui:
	mlflow ui --backend-store-uri file:./mlruns --port 5000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

evaluate:
	python evaluate_models.py

monitor:
	python monitoring/drift_detection.py

explain:
	python explainability/explain_model.py

ab-test:
	python ab_testing/experiment.py

validate-data:
	python data_validation/validate_data.py

feature-store:
	python data_validation/feature_store.py

data-quality:
	make validate-data
	make feature-store

dvc-pull:
	dvc pull data/tech_support_qa.json.dvc

dvc-push:
	dvc add data/tech_support_qa.json
	dvc push

setup: install
	mkdir -p models data mlruns mlartifacts
	@echo "Setup complete!"
