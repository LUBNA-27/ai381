# AI381 Lab 2 · Churn prediction service

![CI](https://github.com/LUBNA-27/ai381/actions/workflows/ci.yml/badge.svg)

A logistic-regression churn model served by FastAPI, packaged in a multi-stage,
non-root Docker image, and built, scanned (Trivy) and pushed to GHCR by GitHub Actions.

## Run it (one command)

```bash
docker run -p 8000:8000 ghcr.io/lubna-27/ai381:3b489252ed96bc9ccc797e2c7365c131f7ddb3a4
```

Then:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"tenure_months": 5, "monthly_charges": 89.9, "contract": "month-to-month", "internet_service": "fiber", "support_calls": 3}'
# {"churn_probability":0.895}
```

Image digest: `sha256:472847071f9f663f5cc0b7786b67b5c5daf04e8fd9f4ed4ccbcec7ad4796b5de`

## Project layout

| Path | What it is |
|---|---|
| `src/train/train.py` | Loads data (with a schema check), trains, saves `model.joblib` |
| `src/app/main.py` | FastAPI app: `/health` and `/predict` with pydantic validation |
| `tests/` | 8 pytest tests: contract, model loads, input schema, edge cases, deterministic prediction |
| `Dockerfile` | Multi-stage (train in builder, serve in runtime), pinned base image, non-root user, HEALTHCHECK |
| `.github/workflows/ci.yml` | PRs: lint + test. `main`: build, Trivy scan, push to GHCR tagged with the commit SHA |

## Develop locally

```bash
pip install -r requirements-dev.txt
python -m src.train.train
python -m pytest -v
uvicorn src.app.main:app --port 8000
```test
