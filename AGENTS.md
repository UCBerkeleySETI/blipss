# AGENTS.md

Guidance for developers and AI coding agents working with `blipss`
(Breakthrough Listen Investigation for Periodic Spectral Signals)

## Project Overview
- Radio-astronomy package that uses the Fast Folding Algorithm (via `riptide-ffa`) to
  search dynamic spectra for narrowband periodic signals, i.e. technosignature candidates.
- Layout: `blipss/cli` (Typer CLI entry points), `blipss/core` (FFA and harmonic
  detection algorithms), `blipss/io` (filterbank/HDF5 I/O, YAML config parsing),
  `blipss/models` (Pydantic models), `blipss/plotting`, `blipss/utils`.
- `tests/` mirrors the `blipss/` package structure 1:1.
- `config/` holds sample YAML configs, one per CLI command.
- Six CLI tools (see README for details): `run-ffa-search`, `compare-cands`,
  `plot-cands`, `compute-phase-resolved-ds`, `inject-signal`, `simulate-data`.

## Setup
- Requires Python 3.12+ and `uv` as the dependency/build manager.
- `make install` creates the venv (`uv sync`) and installs pre-commit hooks.
- Never edit `uv.lock` by hand. Run `uv add`, `uv remove`, or `uv sync` and commit the
  resulting lockfile.

## Running Checks Before Pushing
- `make check` runs lockfile consistency (`uv lock --locked`), pre-commit hooks
  (ruff lint and format, whitespace/EOF/case-conflict checks), `ty check` for static
  type checking, and `deptry .` for unused or missing dependencies.
- `make test` runs `uv run pytest --doctest-modules`, which also exercises doctests.
- Run both before opening a PR. CI runs the same two checks as separate jobs
  (`quality`, `tests`) on every push to `main` and every PR.
- Run a single test file or case the same way, e.g. `uv run pytest tests/path/to/test_x.py -k name`.
- Do not use `python -m pytest` directly. Always go through `uv run` so the correct
  locked environment is used.

## Code Style
- Formatting and linting are ruff-driven (`pyproject.toml` `[tool.ruff]`, line length
  120, preview formatter) and enforced via pre-commit. Do not hand-format against a
  different style.
- Type checking uses `ty` (Astral's checker), not mypy.
- Follow existing patterns in the target module or package rather than introducing
  new ones, and keep changes scoped to what is needed.

## Git and PR Conventions
- Work on a branch off `main`, named `<type>/<short-description>`, e.g. `feature/...`,
  `fix/...`, `refactor/...`, `docs/...`, matching existing branch history.
- Commit messages should be short, imperative, capitalized subject lines, e.g.
  "Add version bump checklist item to PR template."
- All changes land via pull request. There are no direct pushes to `main`, and CI must
  be green before merge.
- PR checklist (from `.github/PULL_REQUEST_TEMPLATE.md`). Confirm each is actually
  true before requesting review:
  - Tests added or updated for the change.
  - `make check` and `make test` pass locally.
  - `README.md` updated if the PR adds or changes CLI functionality.
  - Docstrings and documentation updated.
  - Version bumped in `pyproject.toml` (semver) for any code change.

## Testing Conventions
- New functionality needs tests under `tests/<mirrors-blipss-subpackage>/`.
- Coverage is tracked (`--cov --cov-report=term-missing` locally, `--cov-report=xml`
  plus Codecov upload in CI). Do not add untested code paths.

## Reference Docs
- `CONTRIBUTING.md`: full human-oriented contributor walkthrough.
- `.github/PULL_REQUEST_TEMPLATE.md`: PR checklist template.
- `README.md`: CLI usage and repo organization details.
- `docs/` + `mkdocs.yml`: MkDocs site combining `README.md` (transcluded verbatim into `docs/index.md`) with an API reference generated from docstrings via `mkdocstrings`. Preview locally with `make docs`, strict build check with `make docs-test`. Published to GitHub Pages on push to `main`.
