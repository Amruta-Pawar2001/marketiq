# MarketIQ

MarketIQ is a Marketing Data Intelligence platform built from the assignment plan in `MarketIQ_Project_Plan.docx`.
It uses the Amazon dataset to power XGBoost discount-band prediction and a hybrid OpenAI-backed RAG assistant grounded in product descriptions, reviews, and structured product analytics.

## Stack

- FastAPI API
- XGBoost classification for discount-band strategy
- `SentenceTransformer("all-MiniLM-L6-v2")` embeddings
- FAISS vector search
- OpenAI API for grounded response generation, with structured routing for recommendations and analytics
- Lightweight metrics, health checks, tests, Docker packaging, and the provided MarketIQ HTML dashboard

## Setup

```bash
cd marketiq
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your OpenAI key to `.env`:

```text
OPENAI_API_KEY=your_key
```

## Run End To End

```bash
make ingest
make train
make train-all
make build-index
make serve
```

On Windows PowerShell without `make`, run the equivalent commands:

```powershell
python -m src.pipeline.process
python -m src.models.train
python -m src.models.train_all
python -m src.rag.embedder
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- Dashboard: `http://localhost:8000/`
- Swagger docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Intelligence Tracks

- `python -m src.models.train` trains the XGBoost discount-band model for `/predict_discount`.
- `python -m src.models.demand` trains an XGBoost demand-score model for `/predict_demand`.
- `python -m src.sentiment.analyzer` trains a TF-IDF + Logistic Regression review sentiment model for `/sentiment/{product_id}`.
- `python -m src.segments.clustering` trains K-Means product segments for `/segments`.
- `python -m src.monitoring.drift` writes a drift report for `/drift`.
- `python -m src.models.train_all` runs all model and monitoring artifact generation.

## Chatbot Behavior

`/answer_question` uses a hybrid assistant design:

- Structured product tools handle ranking, recommendations, deal discovery, and category analytics.
- FAISS RAG handles review and product-context questions.
- OpenAI generates grounded natural-language answers from retrieved context.
- Conversation memory sends recent browser chat turns so short follow-ups such as `What about under 1000?` can inherit the previous topic.
- Small talk such as `Thanks` or `Hello` gets a normal conversational reply.
- Clearly unrelated questions are refused with a MarketIQ scope message.

Useful demo questions:

- `Which product should I buy in electronics now?`
- `What about under 1000?`
- `What is our worst-rated product?`
- `Which category has highest discount?`
- `What is the best deal under 1000 in electronics?`
- `Which USB products have quality complaints?`

## Live vs Planned UI

The FastAPI backend is live for prediction, retrieval, sentiment, segmentation, drift, health, and metrics endpoints.

The dashboard also includes polished presentation panels from the original UI reference. These are marked in the UI as `demo`, `sample`, `planned`, or `future scope` when they are not currently calculated live.

Live:

- `/predict_discount` and the prediction form result after clicking **Run prediction**
- `/predict_demand`
- `/answer_question`
- `/rag_trace`
- `/overview_metrics`
- `/discount_distribution`
- `/model_health`
- `/review_insights`
- `/segment_actions`
- `/vector_index_health`
- `/monitoring_dashboard`
- `/top_products`
- `/sentiment/{product_id}`
- `/sentiment_heatmap`
- `/segments`
- `/drift`
- `/health`
- `/metrics`

Planned/demo presentation areas:

- Revenue trend charts
- Revenue trend and top movers
- SHAP bars in the prediction view
- Weekly sentiment trend
- Customer RFM-style cards and LLM-generated segment actions
- Automated retraining execution

## API Examples

```bash
curl -X POST http://localhost:8000/predict_discount ^
  -H "Content-Type: application/json" ^
  -d "{\"category\":\"electronics\",\"actual_price\":2499,\"discounted_price\":1699,\"rating\":4.2,\"rating_count\":1240}"
```

```bash
curl -X POST http://localhost:8000/predict_demand ^
  -H "Content-Type: application/json" ^
  -d "{\"category\":\"electronics\",\"actual_price\":2499,\"discounted_price\":1699,\"rating\":4.2,\"rating_count\":1240}"
```

```bash
curl -X POST http://localhost:8000/answer_question ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What USB cables are available under Rs500?\",\"top_k\":5}"
```

Conversation-memory example:

```bash
curl -X POST http://localhost:8000/answer_question ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What about under 1000?\",\"top_k\":5,\"history\":[{\"role\":\"user\",\"content\":\"Which product should I buy in electronics?\"}]}"
```

```bash
curl http://localhost:8000/segments
curl http://localhost:8000/drift
curl http://localhost:8000/sentiment/B07JW9H4J1
```

## Tests

```bash
make test
```

Windows PowerShell:

```powershell
python -m pytest tests
```

## Docker

```bash
docker-compose up --build
```

The API loads model/index artifacts from mounted `models/` and `data/` directories. Run `make train` and `make build-index` first for fastest startup, or allow the app to generate artifacts on first use.
