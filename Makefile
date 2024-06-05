API_DOCKERFILE = docker/api/Dockerfile
API_IMAGE_NAME = berkmonder/anomaly-platform-api
API_PORT = 8000

help:
	@echo "available commands"
	@echo " - install    : installs all requirements"
	@echo " - dev        : installs all development requirements"
	@echo " - test       : run all unit tests"
	@echo " - clean      : cleans up all folders"
	@echo " - flake      : runs flake8 style checks"
	@echo " - check      : runs all checks (tests + style)"
	@echo " - serve      : runs uvicorn server"

install:
	pip install -r requirements.txt

dev: install
	pip install -r dev-requirements.txt

test:
	pytest test_request.py

clean:
	rm -rf __pycache__ .pytest_cache

flake:
	flake8 ./*.py

check: flake test clean

train:
	python model_train.py

serve:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

build:
	docker compose up -d

build-api:	## Build the Anomaly Detection API Docker Image
	docker build -t $(API_IMAGE_NAME) --file $(API_DOCKERFILE) .

run-api: build-api		## Build and Run a Prediction API Container
	docker run -d -p $(API_PORT):$(API_PORT) $(API_IMAGE_NAME)
	@echo -------------------------------------------
	@echo Prediction API Endpoint: http://localhost:$(API_PORT)/