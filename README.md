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

## Data Pipeline

### 1. Ingestion

Load the JSON response into a collection. We will use this later for API endpoints and extra augmentation.

```bash
python -m app.data_pipeline.ingest_dataset > logs/ingest_dataset.log 2>&1
```

Run embeddings with 50 entries just to verify it all works well

```bash
python -m app.data_pipeline.create_embeddings > logs/create_embeddings_sample.log 2>&1 --limit 50
```

This will take ~15-30 seconds.

Then connect to Mongo via CLI and query the database:

```bash
> mongosh
> use mtg
> db.cards.find()
{
  "_id": "a471b306-4941-4e46-a0cb-d92895c16f8a",
  "object": "card",
  "id": "a471b306-4941-4e46-a0cb-d92895c16f8a",
  "oracle_id": "00037840-6089-42ec-8c5c-281f9f474504",
  "multiverse_ids": [
    692174
  ],
  "mtgo_id": 137223,
  "tcgplayer_id": 615195,
  "cardmarket_id": 807933,
  "name": "Nissa, Worldsoul Speaker",
  ...
}
> db.card_embeddings.find()
{
  "_id": "a471b306-4941-4e46-a0cb-d92895c16f8a",
  "source_id": "a471b306-4941-4e46-a0cb-d92895c16f8a",
  "summary": "Card Name: Nissa, Worldsoul Speaker. Type: Legendary Creature — Elf Druid. Cost: 3 G (CMC 4.0) (Mana Value 4.0). Abilities: Landfall — Whenever a land you control enters, you get E E (two energy counters). You may pay eight E rather than pay the mana cost for permanent spells you cast.. Flavor: \"Zendikar still seems so far off, but Chandra is my home.\". Current Price: $0.17 USD.",
  "embeddings": [...384 entries here...],
  "created_at": {
    "$date": "2026-02-08T09:40:42.662Z"
  }
}
```

If the above went right, you're ready to run the entire ingestion pipeline:

```bash
> python -m app.data_pipeline.create_embedding > logs/create_embeddings.log 2>&1
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
> python -m app.data_pipeline.query_rag "Which cards care about Phyrexians?"
```
