# Contributing to AutoForge

Thanks for contributing.

## Getting Started

1. Fork the repository.
2. Create a feature branch:
   - `git checkout -b feat/short-description`
3. Set up your environment:
   - `python3 -m venv venv`
   - `source venv/bin/activate`
   - `pip install -r requirements.txt`

## Development Guidelines

- Keep changes focused and small when possible.
- Follow Python style conventions (PEP 8, max line length 100).
- Add type hints to function signatures.
- Avoid introducing breaking API changes without discussion.

## Testing Checklist

Before opening a pull request:

- Run the backend locally:
  - `uvicorn backend.main:app --reload --port 8000`
- Verify frontend flow:
  - upload -> train -> download model
- Confirm no local-only files are included in the commit (venv, models, caches).

## Commit and Pull Request

- Use clear commit messages describing the intent of the change.
- Include a short PR description with:
  - what changed
  - why it changed
  - how it was tested

## Code of Conduct

By participating, you agree to follow `CODE_OF_CONDUCT.md`.
