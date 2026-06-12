# docs

This is where I keep all the writing for the project. Code lives in `src/`, results in
`results/`, papers in `papers/`. This folder is the thinking and the record.

If I am coming back to this after a break, read in this order: `research_guidelines.md`, then
`notes/00_vision_and_goals.md`, then `notes/09_research_hypothesis.md`, then the latest entry
in `notes/`.

## Layout

- `research_guidelines.md` how I work, so the research stays publishable. Standing rules plus a
  pre-result integrity checklist.
- `research_log.md` running log of the project at a high level.
- `notes/` the research journal. Numbered entries in the order I did the work, from the vision
  through each experiment. This is the trace I read to remember what I did and why. See
  `notes/README.md` for the list.
- `experiments/` the formal experiment specs.
  - `experiment_01_baseline.md` classification baseline.
  - `experiment_02_detection.md` detection baseline.
  - `experiment_03_transfer.md` the pre-registered transfer study (H1). This is the spec that
    governs the current work.
- `planning/` plans and to-do lists.
  - `roadmap.md` the full research roadmap.
  - `next_steps_todo.md` the active checklist. Start here to see what is done and what is next.
  - `immediate_research_todo.md` an earlier execution checklist.
  - `project_scope.md`, `project_ideas.md` scope and idea backlog.
  - `synthetic_data_plan.md` the gprMax synthetic data plan.
- `reference/` background I look things up in.
  - `data_understanding.md` the GPR dataset, formats, and gotchas.
  - `literature_matrix.md` the paper comparison and gap analysis.

## A copy lives in Obsidian

I keep a snapshot of the journal and the key docs in my Obsidian vault under
`Personal Research/paleo-gpr-ml/` so I can read and link them while I learn. The repo is the
source of truth. The Obsidian copy is a snapshot, so it can drift. I re-copy when I want it
current.
