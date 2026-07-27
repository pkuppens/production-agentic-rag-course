# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

Infer the repo from `git remote -v` — `gh` does this automatically when run inside a clone.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --comments` and `gh pr diff <number>` for the diff.
- **List external PRs for triage**: `gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments` then keep only `authorAssociation` of `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, or `NONE` (drop `OWNER`/`MEMBER`/`COLLABORATOR`).
- **Comment / label / close**: `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close`.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either — resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (`gh api` on the sub-issues endpoint). Where sub-issues aren't enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's **native issue dependencies** — the canonical, UI-visible representation. Add an edge with `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/<owner>/<repo>/issues/<n> --jq .id`, _not_ the `#number` or `node_id`). GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only — the live gate). Where dependencies aren't available, fall back to a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children (`gh issue list --state open`, scoped to the map's sub-issues / task list), drop any with an open blocker (`issue_dependencies_summary.blocked_by > 0`, or an open issue in the `Blocked by` line) or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me` — the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.

## Course workflow (weekly branches)

This repo is a **7-week course** ("The arXiv Paper Curator"), where each week adds one architectural layer on top of the previous week's system. The weeks are tagged in git history (`week1.0` … `week7.0`) and each has a companion blog post and a `notebooks/weekN/` walkthrough:

| Week | Topic |
|------|-------|
| 1 | Infrastructure Foundation — Docker, FastAPI, PostgreSQL, OpenSearch, Airflow |
| 2 | Data Ingestion Pipeline — arXiv API fetching + Docling PDF parsing |
| 3 | Keyword Search Foundation — OpenSearch BM25 |
| 4 | Chunking & Hybrid Search — section-based chunking, embeddings, RRF fusion |
| 5 | Complete RAG Pipeline — local LLM (Ollama), streaming, Gradio UI |
| 6 | Production Monitoring & Caching — Langfuse tracing, Redis caching |
| 7 | Agentic RAG & Telegram Bot — LangGraph agent, Telegram integration |

**Expect one branch per week.** New work for a given week (fixes, extensions, or catching up an unfinished week) belongs on its own branch, e.g. `week3/opensearch-filters` or `feature/week3-NNN-description` — don't mix work belonging to different weeks on one branch.

**Expect multiple commits per week, not one.** A week's branch should accumulate separate commits for its distinct pieces of work (e.g. "add OpenSearch client", "add BM25 query builder", "add search router + tests" rather than one squashed "week 3" commit) so the history documents the incremental build-up the course teaches. Reference the issue in each commit per the standard convention (`#NNN: type: description`).

When triaging or scoping an issue, tag which week it belongs to (either in the title/body or as a label) so it's clear which branch it should land on.
