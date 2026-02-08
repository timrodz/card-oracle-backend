# MTG Oracle backend

## Development Setup

### Environment

- Python `3.12`
- MongoDB `8.2+` (Vector Search)
- Mongo Atlas CLI `1.52`
- Ollama / GPT4All for local models

1. Install `uv`
2. Install Python: `uv python install 3.12`
3. Install MongoDB 8.2: https://www.mongodb.com/docs/manual/administration/install-community/?operating-system=macos&macos-installation-method=homebrew
4. Create a Mongo atlas: `atlas local setup`. Choose `Connection string` as the interaction method.
5. Add the connection string to `MONGODB_URI` in `.env`
6. 

Setup the environment

- `uv venv`
- `source .venv/bin/activate`
- `uv pip install -r requirements.txt`

## Datasets

- Source: [Scryfall](https://scryfall.com/docs/api/bulk-data)

## RAG Setup

### Embeddings

- Model: `mixedbread-ai/mxbai-embed-large-v1` (via Sentence Transformers)

### LLMs

- Model: `Mistral 7B` (via Ollama / GPT4All)
