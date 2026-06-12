---
name: paleo-research-log
description: Log a paleo-gpr-ml research finding, experiment, or idea in Michael's voice into docs/notes or the Obsidian vault, and keep the session aligned with the project's research priorities. Use when Michael says "log this", "write this up", "note this", finishes an experiment, or wants to capture a finding/idea for the GPR-fossil research.
---

# Paleo Research Log

Capture research progress for **paleo-gpr-ml** (first ML approach to dinosaur/fossil prospecting from GPR) the way Michael would write it himself, and keep work pointed at the goal: a novel, publishable, sole-author paper.

## Standing priorities (apply to every session, not just logging)

1. **Novelty first** — move toward the gap (ML on GPR for fossils/dinosaurs), not a leaderboard.
2. **Publishable rigor** — honest splits (always split by original image ID; augmented files are `{id}_aug_{n}.jpg`, ~285 unique scenes), honest metrics, honest writing.
3. **Write everything down, in his voice.**
4. **Master the architectures** — given a real choice, prefer what teaches the most about modern DL while still serving the research.
5. **Sole-author discipline** — make sure Michael understands every piece.

## How to write it (his voice)

- First person, practitioner tone. Warm but grounded. Blunt about tradeoffs and risks. No hype, no fluff.
- Clear `##` headers, **bold** for emphasis, "--" dashes. Follow `ml_report_rules.md`: report weak/mixed/negative results plainly; a credible negative beats an oversold positive.
- Match the register of existing `docs/notes/*.md` (these are the gold standard).

## Where to put it

- **Repo research notes** → `docs/notes/0X_<topic>.md`. Use for experiment write-ups, EDA, decisions, anything tied to the codebase/paper. Number sequentially; lead with `**Date**` and the relevant notebook/script.
- **Obsidian vault** (`~/Documents/Obsidian Vault`) → use for half-formed ideas, the personal research journal, cooler tangents, or cross-project thinking. Templates exist under `Templates/` (Skill Session, Daily Diary) and research lives under `Personal Research/`.
- When in doubt, ask which destination — or put the rigorous version in the repo and a short pointer in the vault.

## Steps

1. Confirm what to capture (finding / experiment result / decision / idea) and the destination (repo vs vault).
2. Pull the relevant numbers/figures from `results/` (tables in `results/tables/`, figures in `results/figures/`, detection runs in `results/detection/`) so the note is grounded in real outputs, not memory.
3. Draft the note in his voice with the structure above. Be honest about what worked, what didn't, and what's still open.
4. End research notes with **Decisions** and **Open questions / next steps**, and link related notes/files.
5. If the finding changes the plan or the gap, also reflect it in `docs/notes/00_vision_and_goals.md`, `docs/planning/immediate_research_todo.md`, or `docs/notes/05_research_contribution.md` as appropriate.
