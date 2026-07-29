# Dev setup: running notebooks in VS Code

## Jupyter kernel: "Python Environment", not "Existing Jupyter Server"

When VS Code opens a notebook (`notebooks/weekN/*.ipynb`) and asks you to pick a kernel, choose
**"Python Environment"** and select `.venv` (created by `uv sync`; shown as `moai-zero-to-rag` —
that's just the display name uv derives from `pyproject.toml`, not a separate environment).

Don't pick "Existing Jupyter Server" — this repo has no standalone Jupyter server or Docker
service for notebooks (`compose.yml` only runs the backend services: Postgres, OpenSearch,
Airflow, Ollama, Redis, Langfuse). "Python Environment" lets VS Code run the kernel directly
against `.venv`, matching the README's `uv run jupyter notebook ...` workflow, with one fewer
process to manage.

If a notebook's kernel hangs, restart VS Code and rerun the cells — notebook data in this course
is small and reproduces quickly, so a running server kept alive outside VS Code (which would
survive a VS Code restart) isn't worth the added complexity here.
