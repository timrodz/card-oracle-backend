# Embeddings Pipeline Overview

This document summarizes how `scripts/create_embeddings.py` works, including key stages, data flow, and storage behavior.

## High-Level Summary
The script loads Scryfall card JSON, normalizes and chunks `oracle_text`, creates embeddings using a local SentenceTransformer model, and upserts the resulting records into MongoDB. Cards missing `oracle_text` are still stored as metadata-only records (no embedding) so the dataset remains complete.

## Architecture Diagram

```mermaid
flowchart TD
    A[Dataset Path] --> B[Load JSON Cards]
    B --> C[Select Fields]
    C --> D{oracle_text present?}
    D -- no --> E[Create Empty Record]
    E --> F[Upsert Empty Records]
    D -- yes --> G[Normalize Text]
    G --> H[Chunk Oracle Text]
    H --> I[Create Chunk Records]
    I --> J[Embed Chunks]
    J --> K[Upsert Chunk Records]
    F --> L[MongoDB Collection]
    K --> L[MongoDB Collection]
```

## Detailed Flow

```mermaid
sequenceDiagram
    autonumber
    participant CLI as CLI/Env
    participant Loader as load_raw_cards
    participant Select as select_and_validate_fields
    participant Normalize as normalize_text
    participant Chunk as chunk_oracle_text
    participant Build as build_chunk_records/build_empty_record
    participant Embed as embed_chunks
    participant Mongo as MongoDB

    CLI->>Loader: Read JSON files
    Loader->>Select: Yield raw card
    Select-->>Normalize: Card with oracle_text
    Select-->>Build: Card without oracle_text
    Normalize->>Chunk: Cleaned oracle_text
    Chunk-->>Build: Text chunks
    Build-->>Embed: Chunk records
    Embed-->>Mongo: Upsert embeddings
    Build-->>Mongo: Upsert empty records
```

## Key Functions

- `load_raw_cards()`
  - Reads JSON files from the dataset path and yields card dicts.
- `select_and_validate_fields()`
  - Extracts `oracle_text`, `prices.usd`, `cmc`, `mana_cost`, `set_name`, and identifiers.
  - Treats missing or blank `oracle_text` as `None` (no longer skipped).
- `normalize_text()`
  - Collapses whitespace, preserves MTG symbols.
- `chunk_oracle_text()`
  - Splits `oracle_text` using the model tokenizer into overlapping token windows.
- `build_chunk_records()`
  - Builds per-chunk MongoDB docs with `has_oracle_text=true`.
- `build_empty_record()`
  - Creates metadata-only docs for cards with `oracle_text=None`.
- `embed_chunks()`
  - Generates embeddings in batches and attaches metadata.
- `upsert_embeddings()`
  - Bulk upserts into MongoDB by `_id`.

## Storage Behavior

- Chunk documents use `_id` pattern: `${card_id}:${chunk_index}`
- Empty records use `_id` pattern: `${card_id}:-1`
- `has_oracle_text` indicates whether the card produced embeddings.
- Empty records store `embedding=null`, `embedding_dim=0`.

## Operational Notes

- Uses `.env` for configuration (`MONGODB_URI`, model path, chunk sizes).
- If GPU is available, uses `cuda`; otherwise falls back to CPU.
- Logs batch progress and totals to stdout.
