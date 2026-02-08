# PRD: Create Embeddings for MTG Oracle Text (Local MongoDB 8.2)

## Summary

Create a local, self-hosted embedding pipeline that reads the Scryfall JSON dataset, filters/normalizes card data, chunks `oracle_text`, generates embeddings using `SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')`, and stores the results in MongoDB 8.2 for later vector search. Querying and API exposure are explicitly out of scope for this phase.

## References

- MongoDB Vector Search Local RAG tutorial (local deployment and indexing guidance)

## Goals

- Build a deterministic, repeatable embedding pipeline.
- Store embeddings and metadata in MongoDB 8.2 (self-hosted) for later retrieval.
- Skip and log cards missing `oracle_text`.
- Keep `prices.usd`, `cmc`, `mana_cost`, and `set_name` in metadata.

## Non-Goals (This Phase)

- Vector search query pipeline
- RAG answer synthesis / LLM generation
- FastAPI endpoints
- Ranking, reranking, or evaluation

## Data Requirements

- Input: JSON dataset under `datasets/scryfall` (36,709 records).
- Required field: `oracle_text` (missing values must be logged and skipped).
- Optional field: `prices.usd` (null allowed).
- Additional fields to preserve: `cmc`, `mana_cost`, `set_name`, `id`, `name`, `collector_number`, `type_line`, `rarity`.

## Embedding Model

- Model: `mixedbread-ai/mxbai-embed-large-v1` via `SentenceTransformer`.
- Run locally; use GPU if available, otherwise CPU fallback.
- Use model’s tokenizer to validate chunk sizes and ensure chunks fit within model limits.

## Output Schema (MongoDB)

Each chunk should be stored as a separate document for vector search:

- `_id`: stable unique id (e.g., `${card_id}:${chunk_index}`)
- `embedding`: vector
- `chunk_text`: chunk content
- `oracle_text`: original full oracle text (optional but useful for debugging)
- `price_usd`: optional
- `cmc`, `mana_cost`, `set_name`
- `name`, `type_line`, `rarity`, `collector_number`
- `source_id`: original card `id`

## Step-by-Step Implementation Plan (Embeddings Only)

### 1. Data Loading

**Function:** `load_raw_cards()`

- Read all JSON files from `datasets/scryfall`.
- Yield/return raw card objects.
- Record total card count for reporting.

### 2. Field Selection + Validation

**Function:** `select_and_validate_fields(raw_card)`

- If `oracle_text` is missing/empty: log `card_id` and `name`, increment skipped counter, return `None`.
- Extract:
  - `oracle_text`
  - `prices.usd`
  - `cmc`, `mana_cost`, `set_name`
  - `id`, `name`, `collector_number`, `type_line`, `rarity`
- Return a normalized card payload.

### 3. Text Normalization

**Function:** `normalize_text(card)`

- Normalize whitespace (collapse repeated spaces, trim).
- Preserve MTG symbols like `{G}`, `{T}`.
- Output a clean `oracle_text` with attached metadata.

### 4. Chunking

**Function:** `chunk_oracle_text(doc)`

- Split `oracle_text` into overlapping chunks.
- Chunk sizing should be based on model’s tokenizer max length.
- Recommended starting point:
  - Chunk size: 256–384 tokens
  - Overlap: 40–80 tokens
- If `oracle_text` is short, keep a single chunk.

### 5. Embedding

**Function:** `embed_chunks(chunks)`

- Initialize `SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')` once.
- Embed in batches for performance.
- Use GPU if available; fallback to CPU.
- Return embeddings aligned with chunk metadata.

### 6. Prepare MongoDB Documents

**Function:** `prepare_mongo_docs(embedded_chunks)`

- Construct the final document schema (see Output Schema).
- Ensure vector field is a list/array of floats.

### 7. Upsert into MongoDB

**Function:** `upsert_embeddings(docs)`

- Upsert by `_id` to avoid duplicates.
- Record counts: inserted vs updated.

### 8. Reporting

**Function:** `report_stats()`

- Total cards read
- Total cards skipped (missing `oracle_text`)
- Total chunks created
- Total embeddings inserted/updated

## Logging & Observability

- Log skipped cards with missing `oracle_text` (include `id` and `name`).
- Log batch progress every N records (e.g., 1,000).
- Capture timing for chunking + embedding to estimate throughput.

## Acceptance Criteria

- Pipeline runs end-to-end locally.
- Missing `oracle_text` cards are skipped and logged.
- Embeddings are stored in MongoDB with correct metadata and stable `_id`.
- A report is produced with counts and timings.

## Risks / Open Questions

- Validate model memory usage on the available GPU; if too slow, consider smaller local model fallback.
- Confirm final chunk sizes using the model tokenizer to avoid truncation.
- Ensure MongoDB 8.2 vector search setup (`mongot`) is installed and running (querying will be verified later).
