# Text Similarity Service

A minimal FastAPI service to compute text similarity using external embedding server or a local fallback embedding.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/api/v1/status
```

Compute similarity:

```bash
curl -X POST http://localhost:8000/api/v1/compute_topk_similarity \
  -H 'Content-Type: application/json' \
  -d '{
    "query_texts": ["hello world", "fast api"],
    "candidate_texts": ["hello", "api", "python"],
    "top_k": 2
  }'
```

## Configure embeddings

Set environment variables or edit `configs/base_config.yml`.

Env overrides:

- `EMBED_MODEL_KEY`: key in YAML under `embeddings` (e.g., `bge`)
- `EMBED_MODEL_NAME`: model name string
- `EMBED_SERVER_URL`: embedding server URL
- `EMBED_CONTEXT_LENGTH`: max input context length
- `LOCAL_EMBED_DIM`: dimension for local fallback embedding (default 384)

If no server is configured or reachable, a deterministic local embedding fallback is used.

## Docker

```bash
docker build -t text-sim-service .
docker run --rm -p 8000:8000 \
  -e EMBED_MODEL_KEY=bge \
  text-sim-service
```