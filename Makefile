.PHONY: install ingest train evaluate build-index serve test docker-up docker-down

install:
	pip install -r requirements.txt

ingest:
	python -m src.pipeline.process

train:
	python -m src.models.train

train-all:
	python -m src.models.train_all

train-sentiment:
	python -m src.sentiment.analyzer

train-segments:
	python -m src.segments.clustering

train-demand:
	python -m src.models.demand

drift:
	python -m src.monitoring.drift

evaluate:
	python -m src.models.evaluate

build-index:
	python -m src.rag.embedder

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests -q

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down
