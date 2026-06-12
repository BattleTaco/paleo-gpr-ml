---
name: reference-docs-layout-and-obsidian
description: How paleo-gpr-ml docs are organized and where the Obsidian mirror lives
metadata: 
  node_type: memory
  type: reference
  originSessionId: e6c235d4-3a76-4721-a696-dfcdc468b126
---

paleo-gpr-ml docs were reorganized (2026-06-10). Layout under `docs/`:
- `docs/README.md` index of the whole docs folder.
- `docs/notes/` the dated research journal (entries 00–11), with `docs/notes/README.md` as the chronological index. This is the trace Michael reads to relearn the work.
- `docs/experiments/` experiment specs (`experiment_01_baseline.md`, `experiment_02_detection.md`, `experiment_03_transfer.md`).
- `docs/planning/` `roadmap.md`, `next_steps_todo.md` (active checklist), `immediate_research_todo.md`, `project_scope.md`, `project_ideas.md`, `synthetic_data_plan.md`.
- `docs/reference/` `data_understanding.md`, `literature_matrix.md`.
- `docs/research_guidelines.md` and `docs/research_log.md` stay at the docs root.

**Obsidian mirror:** a snapshot of the journal and key docs is copied to Michael's vault at `~/Documents/Obsidian Vault/Personal Research/paleo-gpr-ml/` (subfolders `journal/`, `experiments/`, `reference/`, plus an index note `paleo-gpr-ml.md` with [[wikilinks]]). The **repo is the source of truth**; the Obsidian copy is a snapshot that drifts. When journal notes or key docs change meaningfully, re-copy them into the vault so Michael has a current trace to learn from. See [[feedback-log-everything-we-do]] and [[reference-sources-library]].
