# Lexigo document Q&A API



## Requirements

- Python 3.10+
- A Cerebras API key for answer generation
- A Pinecone API key and a Pinecone serverless index

The corpus is fictional. Do not add real client or production data.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` with real keys. Never commit `.env`.

### Pinecone index

The application creates the configured serverless index during ingestion if it does not already exist. Set `PINECONE_CLOUD` and `PINECONE_REGION` to a region available in your Pinecone account. Embeddings now come from Pinecone inference using `llama-text-embed-v2`, which produces 1024-dimensional vectors, so `EMBEDDING_DIMENSION` should be `1024`.

Required Pinecone-related variables are `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, `PINECONE_REGION`, and `PINECONE_NAMESPACE`.

Required Cerebras variables are `CEREBRAS_API_KEY`, `CEREBRAS_BASE_URL`, and `CHAT_MODEL`.

## Ingest

From the repository root:

```bash
python -m lexigo.ingest
```

The command reads Markdown files from `CORPUS_DIR`, chunks them, embeds them with Pinecone inference, and upserts them into Pinecone. To replace the namespace first:

```bash
python -m lexigo.ingest --reset-namespace
```

Running ingest twice without `--reset-namespace` is safe: chunk IDs are deterministic hashes of the relative source path, position, and text, so Pinecone upserts the same IDs rather than creating duplicates. `--reset-namespace` is useful when removing old documents. If you delete the whole index, the next ingest recreates it with the configured dimension.

## Start the API

```bash
uvicorn lexigo.api:app --reload
```

Check configuration:

```bash
curl http://127.0.0.1:8000/health
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How long is the employment notice period?","include_trace":true}'
```

The response includes `answer`, `grounded`, and `citations`. Each citation contains the Pinecone chunk ID, source file, chunk index, relevance score, and a short excerpt. `trace` is returned only when `include_trace` is true.

An out-of-corpus question returns HTTP 200 with `grounded: false`, no citations, and: `I cannot find that information in the provided documents.` This is intentional; absence of evidence is not answered using model knowledge.

## LangGraph map

The graph is documented in [`docs/langgraph.md`](docs/langgraph.md). In short:

```text
START -> retrieve -> assess -- relevant --> answer -> END
                         \-- insufficient --> fallback -> END
```

`retrieve` searches Pinecone. `assess` checks the top cosine score and lexical overlap with the retrieved evidence. The conditional edge prevents weak evidence from reaching the LLM. `answer` is given only retrieved excerpts and must cite chunk IDs. `fallback` answers that the fact cannot be found. Every invocation carries a step counter and uses LangGraph's recursion limit (`MAX_GRAPH_STEPS`) as a second guard against runaway execution.

## Self-test set

The 12-question self-test is in [`sample_test_cases.json`](sample_test_cases.json), with a readable table in [`sample_test_cases.md`](sample_test_cases.md). It covers direct lookups, multi-document questions, and out-of-corpus refusal cases. The expected files are the citation contract; wording may differ.

Run the local unit tests:

```bash
pytest
```

The live self-test requires configured Cerebras and Pinecone keys and a running API. After starting the server, run the HTTP client below; it checks the expected citation files and grounded flag without providing a separate question-answering service:

```bash
python scripts/evaluate.py
```



