# AI Act Knowledge Graph

Knowledge-graph powered assistant for exploring the EU Artificial Intelligence Act.  
The repo contains the scraping pipeline that turns the legal text into structured JSON, loaders that populate a Memgraph/Neo4j instance with richly linked nodes (articles, annexes, recitals, paragraphs), and a FastAPI app that serves a custom ReAct-style Ollama agent with custom observability, a lightweight frontend, and Valkey-backed conversational memory.

---

## Highlights
- **End-to-end pipeline** · Scrape articles/annexes/recitals, normalize identifiers, and ingest everything into Memgraph with hierarchical `CONTAINS`, sequential `NEXT`, and cross-document `REFERENCES` edges.
- **Vector-enhanced retrieval** · Optional Gemma embeddings (via Ollama) power similarity search through Memgraph’s `vector_search` module for semantic lookups.
- **Ollama Agent** · A custom Ollama agent that performs ReAct-style tool calling with tight control over reasoning steps for on-device or self-hosted LLM use.
- **FastAPI + frontend** · REST endpoints (`/chat`, `/explain`, `/health`) plus Matrix-inspired HTML templates hosted straight from FastAPI.
- **Observability and memory** · Custom handler captures traces, while `ValkeyMemoryStore` persists per-session short-term memory for iterative chats.
- **Docker-first** · `docker-compose.yml` spins up Memgraph, Memgraph Lab, Valkey, and the FastAPI app. Local development works with `uv` and Python 3.12.

---

## Architecture
```
┌─────────────┐      ┌───────────┐      ┌───────────────┐
│  Scrapers   │ ---> │  JSON in  │ ---> │  generate_... │
│ (articles,  │      │  data/*   │      │  graph.py     │
│ annexes...) │      └───────────┘      │  (Memgraph)   │
└─────┬───────┘                         └─────┬─────────┘
      │                                      │ BOLT
      │                               ┌──────▼────────┐
      │                               │ Memgraph +    │
      │                               │ vector index  │
      │                               └──────┬────────┘
      │                                      │  
      │                                      │ custom Ollama agent
┌─────▼────────┐                       ┌─────▼────────────┐
│ FastAPI app  │ <--> Valkey memory    │ Frontend &       │
│ (main.py)    │                       │ REST clients     │
└──────────────┘                       └──────────────────┘
```

---

## Getting Started

### Prerequisites
- Python 3.12+ and [uv](https://github.com/astral-sh/uv) (the repo uses `uv` for dependency management)
- Docker & Docker Compose (Memgraph, Valkey, optional web service)
- Running Ollama instance with the models you plan to use (e.g., `qwen2.5:32b`, `embeddinggemma:latest`)
- Optional: custom instance for tracing, Memgraph Lab (bundled in compose) for visualization

### Local Python workflow
```bash
uv sync
cp .env.example .env    # create your env file if needed
uv run uvicorn main:app --reload --port 8000
```
Point your browser to `http://localhost:8000` (or `/explain`) after the graph and tools are ready.

### Docker Compose workflow
```bash
docker compose up memgraph lab valkey      # start persistence services
docker compose up --build web              # build + run FastAPI app (listens on :8001)
```
- Memgraph is exposed on `bolt://localhost:7690`; Memgraph Lab UI lives on `http://localhost:3001`.
- Valkey (Redis-compatible) runs for chat memory.
- The `web` service mounts `.env` and runs `uvicorn main:app --host 0.0.0.0 --port 8000` (mapped to `localhost:8001`).

Stop services with `docker compose down` when you are done.

---

## Environment Variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `BOLT_URI` | Neo4j/Memgraph Bolt endpoint used by loaders and tools | `bolt://localhost:7690` |
| `OLLAMA_URL` | Location of the Ollama server | `http://localhost:11434` |
| `VALKEY_URI` | Redis/Valkey URL for session memory | `redis://:@localhost:6379` |

Add anything else (API keys, alternate models) to `.env` and the FastAPI service will pick them up.

---

## Build the Knowledge Graph

1. **Scrape the EU AI Act**  
   Use `article_scraper.py` (or the notebooks) to turn HTML pages into structured JSON. Typical command:
   ```bash
   python article_scraper.py --url https://artificialintelligenceact.eu/article/2/ \
     > data/articles/article_2.json
   ```
   The repo already ships with scraped JSON under `data/articles`, `data/annexes`, and `data/recitals`.

2. **Run the Memgraph loader**  
   ```bash
   python generate_graph.py --embeddings --embed-parent-context
   ```
   Flags:
   - `--embeddings` builds Gemma embeddings via Ollama and stores them on `EmbeddingNode` vertices for vector search.
   - `--embed-parent-context` includes ancestor paragraph text during embedding to improve recall.

   The script:
   - clears Memgraph,
   - optionally creates a `VECTOR INDEX` named `embed_index`,
   - loads all JSON files, generating consistent IDs (e.g., `Art.5.1`, `Annex.III`),
   - wires `CONTAINS`, `NEXT`, and `REFERENCES` relationships (plus inverse edges).

3. **Inspect with Memgraph Lab** (optional) to verify node counts and sample queries.

---

## Running the Agents & API

Once Memgraph and Valkey are reachable, start the FastAPI server (either via `uv run` or Docker). Key routes:

- `GET /` – renders `frontend/templates/chat.html`
- `GET /explain` – renders `frontend/templates/explain.html`
- `POST /chat` – custom Ollama agent with explicit ReAct loop and memory
- `GET /health` – simple readiness probe

Example request to the chat endpoint:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
        "question": "Summarize the transparency obligations in Article 50.",
        "session_id": "demo-session-1",
        "metadata": {"source": "cli"}
      }'
```

The response contains `answer` plus an `observation_log` showing each tool invocation performed by the Ollama agent.

---

## Knowledge-Graph Tools

| Tool | File | Description |
| --- | --- | --- |
| `node_retrieval` | `tools/node_retrieval.py` | Embeds free-form queries with `embeddinggemma`, queries Memgraph’s vector index, and returns the top-k nodes with similarity scores. |
| `graph_traversal` / `TraversalTool` | `tools/graph_traversal.py`, `tools/traversal2.py` | Expands a node into its hierarchical text, gathers `REFERENCES`, and surfaces parent/child context. |
| `single_node_retriever` | `tools/single_node_retriever.py` | Lightweight fetch by node ID with labels + properties (no traversal). |

`ollama_agent.ReActAgent` publishes the tools as Ollama function-calling schemas. You can also run each script via CLI to debug outputs.

---

## Memory & Observability
- `agent/memory.py` defines a `ShortTermMemory` object that automatically reflects on conversation history and stores condensed notes.  
- `ValkeyMemoryStore` serializes that memory per `session_id` so `/chat` calls can resume context even after agent restarts.  
- Observability is built-in to the `OllamaAgent`

---

## Utilities, Notebooks & Frontend
- `tool_demo.ipynb`, `scraper.ipynb`, `fill_graph_playground.ipynb` demonstrate scraping, traversal, and ingestion concepts.
- `frontend/static` holds a minimal Matrix-inspired interface used by `/` and `/explain`, so you can chat without needing another UI.
- `schema_description.py` retrieves the graph schema and provides it in a (LLM) readable format

---

## Troubleshooting
- **Memgraph connection errors** · Ensure `docker compose up memgraph` is running and that `BOLT_URI` points to `bolt://localhost:7690`.  
- **Ollama errors / missing models** · Run `ollama pull qwen2.5:32b` (or your preferred model) and check that `OLLAMA_URL` is reachable from both the host and containers.  
- **Vector search fails** · Confirm you executed `generate_graph.py --embeddings` at least once so `EmbeddingNode` vertices and the `embed_index` exist.  
- **Session memory resets** · Provide a `session_id` field in your POST body; otherwise, each call gets an ephemeral memory instance.

---

## Contributing
Please run `uv sync` before opening PRs and document any new environment variables or commands in this README.
