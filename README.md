# Card Oracle backend

## Development Setup

### Environment

- Python `3.12`
- MongoDB `8.2+` (Vector Search)
- Mongo Atlas CLI `1.52`
- Ollama for LLM models (`ollama`)
- Huggingface for transformers (via `uvx hf`)

#### Transformers

- Model: `mixedbread-ai/mxbai-embed-xsmall-v1` (via Sentence Transformers)
  - Dimensions: 384 (Enough for short text)

```bash
uvx hf auth login
uvx hf download mixedbread-ai/mxbai-embed-xsmall-v1 --local-dir models/mixedbread-ai/mxbai-embed-xsmall-v1
```

#### LLMs

- Model: `mistral:7b` (via Ollama)

```bash
ollama run mistral
```

### Installation

1. Install `uv`
2. Install Python: `uv python install 3.12`
3. Install [MongoDB 8.2](https://www.mongodb.com/docs/manual/administration/install-community/?operating-system=macos&macos-installation-method=homebrew). 8.2 has vector database support so this is the minimal version required.
4. Create a Mongo atlas: `atlas local setup`. Choose `Connection string` as the interaction method.
5. Add the connection string to `MONGODB_URI` in `.env`
6. Setup `EMBED_MODEL` in `.env` with the location of your installed model: `models/<model>`

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Development

Run the FastAPI server:

```
fastapi dev main.py
```

## Datasets

- Source: [Scryfall](https://scryfall.com/docs/api/bulk-data)

## RAG Pipeline

### 1. Ingestion

Verify the pipeline works:

```bash
python scripts/create_embeddings.py > logs/embedding-sample.log 2>&1 --limit 50
```

This will take ~15-30 seconds.

Then connect to Mongo via CLI and query the database:

```bash
> mongosh
> use mtg
> db.card_embeddings.find()
{
  _id: 'bea12617-ebaa-45f6-a2e8-b71190708129:0',
  source_id: 'bea12617-ebaa-45f6-a2e8-b71190708129',
  chunk_index: 0,
  chunk_count: 1,
  chunk_text: "Affinity for Phyrexians (This spell costs {1} less to cast for each Phyrexian you control.) Flying Phyrexian Broodstar's power and toughness are each equal to the number of Phyrexians you control.",
  oracle_text: "Affinity for Phyrexians (This spell costs {1} less to cast for each Phyrexian you control.) Flying Phyrexian Broodstar's power and toughness are each equal to the number of Phyrexians you control.",
  has_oracle_text: true,
  price_usd: null,
  cmc: 8,
  mana_cost: '{6}{U}{U}',
  set_name: 'Unknown Event',
  name: 'Phyrexian Broodstar',
  type_line: 'Creature — Phyrexian Beast',
  rarity: 'rare',
  embeddings: [<384 entries here>]
}
```

If the above went right, you're ready to run the entire ingestion pipeline:

```bash
> python scripts/create_embeddings.py > logs/embedding-sample.log 2>&1
```

Note: This will take anywhere around 10 minutes on a Macbook with no GPU.

Create the MongoDB search index:

```bash
> atlas local ls
NAME                 MDB VER    STATE
<DEPLOYMENT_NAME>    8.2.4      running
# This command below doesn't work in Atlas 1.52, use the old one for now
# > atlas local search indexes create --deploymentName <DEPLOYMENT> --file vector-index.json 
> atlas deployments search indexes create --file vector-index.json
you're using an old search index definition
Search index created with ID: 69884584b41ae52dc72ff971
```

Verify the index setup worked:

```bash
> atlas local search indexes list --deploymentName <DEPLOYMENT> --output json --db mtg --collection card_embeddings
{"outcome":"success","indexes":[{"id":"698842c3b41ae52dc72ff96f","name":"vector_index","database":"mtg","collectionName":"card_embeddings","status":"READY","type":"vectorSearch"}]}
```

### 2. Query (RAG)

Use the local Sentence Transformers embedder for the query, run vector search, and
have Ollama answer with retrieved context:

```bash
python scripts/query_rag.py "Which cards care about Phyrexians?"
```

Environment variables for tuning the query flow:

- `VECTOR_INDEX_NAME` (default: `vector_index`)
- `VECTOR_EMBED_PATH` (default: `embeddings`)
- `VECTOR_NUM_CANDIDATES` (default: `100`)
- `VECTOR_LIMIT` (default: `5`)
- `OLLAMA_MODEL` (default: `mistral:7b`)
- `RAG_MAX_CONTEXT_CHARS` (default: `4000`)
- `OLLAMA_TIMEOUT` (default: `120`)
