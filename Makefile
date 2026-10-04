# Makefile
install:
	pip install -r requirements.txt

preprocess:
	python src/preprocess.py

train:
	python src/train.py

run-api:
	uvicorn api.main:app --reload --port 8000

test:
	pytest tests/

pipeline: preprocess train test