---
name: research-priorities-paleo-gpr
description: "Standing decision priorities for paleo-gpr-ml — novelty, publishable rigor, logging, mastery"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e6c235d4-3a76-4721-a696-dfcdc468b126
---

Standing priorities for paleo-gpr-ml. When a decision is ambiguous, default to whatever serves these (in order):

1. **Novelty first.** Every experiment should move toward the gap (ML on GPR for fossils/dinosaurs), not toward a leaderboard. Beating SOTA on pipes/infrastructure is not the point.
2. **Publishable rigor.** Honest splits — **always split by original image ID to avoid augmentation leakage** (augmented files are `{id}_aug_{n}.jpg`; effective unique scenes ≈285, not 2,524). Honest metrics, honest writing.
3. **Write everything down, in his voice.** See [[feedback-write-in-michaels-voice]].
4. **Master the architectures.** Given a real choice, prefer the approach that teaches the most about modern deep learning, as long as it still serves the research. Depth over convenience. Default to current/SOTA-capable architectures.
5. **Sole-author discipline.** Explain enough that Michael understands and can defend every piece himself.

**Standing practices** are written up in `docs/research_guidelines.md` (pre-register before modeling; controlled comparisons / one variable; no leakage; quantify uncertainty + don't oversell; ground claims in physics + cite honestly; scope claims to domain-shift not deployment; reproducibility; log everything; test the science; let the data pick the story). Before claiming any experimental result, run the **`research-integrity-check`** skill (the pre-result checklist). Experiments are pre-registered in `docs/experiments/experiment_03_transfer.md` — don't move goalposts; log + date any deviation.

**Why:** These come straight from his goal — a novel, credible, sole-author paper that also makes him a stronger DL engineer. See [[project-novel-gpr-fossil-research]] and [[feedback-log-everything-we-do]].
