# MTG Oracle backend

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

#### Installation

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

## Datasets

- Source: [Scryfall](https://scryfall.com/docs/api/bulk-data)
