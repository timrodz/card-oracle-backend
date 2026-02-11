# Repository Guidelines

## Project Structure & Module Organization

- `main.py` is a minimal entry point that loads env vars and connects to MongoDB.
- `scripts/create_embeddings.py` contains the core embeddings pipeline (dataset parsing, chunking, embedding, Mongo upserts).
- `datasets/` stores Scryfall bulk data; see `datasets/scryfall/README.md` for source notes.
- `env.example` documents required environment variables.
- `requirements.txt` pins Python dependencies.

## Build, Test, and Development Commands

- `uv venv` creates the local virtual environment.
- `source .venv/bin/activate` activates the venv. Run this once per session.
- `uv pip install -r requirements.txt` installs dependencies. When adding a new dependency run `--upgrade` so you grab the latest version.
- Always run Python via `.venv/bin/python` (even if the venv is activated) to ensure the correct interpreter is used.
- `.venv/bin/python main.py` verifies Mongo connectivity (prints the URI and a hello message).
- `.venv/bin/python scripts/create_embeddings.py --dataset-path datasets/scryfall` generates and upserts embeddings for Scryfall JSON data.

## Coding Style & Naming Conventions

- Python 3.12, 4-space indentation, PEP 8–style naming (snake_case for functions/vars, CapWords for classes).
- Prefer small, pure helpers (see `select_and_validate_fields`, `chunk_oracle_text`).
- Formatting/linting tools are not configured yet; keep changes consistent with existing style.

## Testing Guidelines

- No automated test suite is present in this repo yet.
- When adding tests, follow standard `pytest` conventions: `tests/` folder, `test_*.py` files, and descriptive test names.
- Document new test commands here when introduced.

## Commit & Pull Request Guidelines

- Commit history uses Conventional Commits (e.g., `feat: ...`).
- Keep commits scoped to a single change and mention dataset or schema impacts.
- PRs should include: a concise summary, any new env vars, and notes on required dataset files.

## Configuration & Data Notes

- Dataset files are large JSON arrays; validate structure before running embeddings.
