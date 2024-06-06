help:
	@echo "available commands"
	@echo " - install    : installs all requirements"
	@echo " - test       : run all unit tests"
	@echo " - clean      : cleans up all folders"
	@echo " - flake      : runs flake8 style checks"
	@echo " - check      : runs all checks (tests + style)"
	@echo " - serve      : runs uvicorn server"
	@echo " - build      : builds docker containers"

install:
	pip install -r requirements.txt

test:
	pytest test_request.py

clean:
	rm -rf __pycache__ .pytest_cache

flake:
	flake8 ./*.py

check: flake test clean

serve:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

train:
	python model_train.py

build:
	docker compose up -d --build --remove-orphans
