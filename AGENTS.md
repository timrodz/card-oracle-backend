# Repository Guidelines

## Project Structure & Module Organization
Core backend code lives in `app/`.
- `app/main.py`: FastAPI entrypoint and routes.
- `app/core.py`: API orchestration for card lookup and RAG search.
- `app/models/`: Pydantic models for API, dataset, embeddings, and LLM payloads.
- `app/data_pipeline/`: ingestion, embedding creation, and query-time retrieval scripts.
- `app/settings.py`: environment-based configuration.

Supporting resources:
- `datasets/`: local Scryfall inputs.
- `models/`: local embedding model artifacts.
- `docs/`: architecture diagrams, RAG notes, and index specs.
- `logs/`: pipeline run outputs.

## Build, Test, and Development Commands
Use Python 3.12 and `uv`.
- `uv venv && source .venv/bin/activate`: create and activate local environment.
- `uv sync` (or `uv pip install -r requirements.txt`): install dependencies.
- `fastapi dev app/main.py`: run the API locally.
- `python -m app.data_pipeline.ingest_dataset`: load dataset into MongoDB.
- `python -m app.data_pipeline.create_embeddings --limit 50`: quick embedding smoke test.
- `python -m app.data_pipeline.query_rag "Which cards care about Phyrexians?"`: run retrieval flow.

## Coding Style & Naming Conventions
- Follow PEP 8, 4-space indentation, and explicit type hints.
- Use `snake_case` for functions/modules, `PascalCase` for classes, and descriptive config names.
- Keep FastAPI route handlers thin; move logic into `app/core.py` or pipeline modules.
- Validate and serialize data through Pydantic models in `app/models/`.
- Run `uv run ruff check .` and `uv run mypy app` before opening a PR.
- Use `pydantic` utilities anywhere you can. Create custom types often.

## Testing Guidelines
`pytest` is the expected test runner.
- Place tests under `tests/` mirroring `app/` paths (example: `tests/data_pipeline/test_create_embeddings.py`).
- Name tests `test_*.py` and prefer behavior-focused test names.
- Run `uv run pytest` locally; add regression tests for parser, embedding, and query logic changes.

## Commit & Pull Request Guidelines
Repository history uses Conventional Commits (`feat:`, `refactor:`, `chore:`, `docs:`).
- Keep commit messages imperative and scoped (example: `feat: add pydantic search response model`).
- In PRs, include: summary, key changes, validation steps run (`ruff`, `mypy`, `pytest`), and linked issue/ticket.
- For API or pipeline behavior changes, include sample request/response or log snippets.

## Security & Configuration Tips
- Copy `.env.example` to `.env`; never commit secrets.
- Keep `MONGODB_URI`, model paths, and provider settings environment-driven via `app/settings.py`.
- Large local artifacts (`datasets/`, `models/`, `logs/`) should stay out of git.
