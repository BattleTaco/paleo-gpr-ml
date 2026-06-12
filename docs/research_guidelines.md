# Research Guidelines, How I Do Research Here

**Date**: 2026-06-10
**Purpose**: The standing practices that keep this project publishable and defensible. Every
session (me or any agent) follows these. If a guideline ever conflicts with moving fast, the
guideline wins, the whole point is a paper that survives review.

These are distilled from how the project already works (`ml_report_rules.md`, the leakage
control in `src/data/build_splits.py`, the pre-registration in `docs/experiments/experiment_03_transfer.md`)
plus standard open-science practice. They are the bar I hold every result to.

---

## 1. Pre-register before you model
Conditions, metrics, and decision rules are written down *before* running the experiment
(`docs/experiments/experiment_03_transfer.md`). No inventing a metric after seeing results. Any post-hoc
change is logged with date + reason and disclosed in the paper. This is my #1 defense against
"cherry-picking."

## 2. Controlled comparisons, one variable at a time
When I claim X causes Y, the design must vary X and hold everything else constant (e.g. the
`anti_bone` control varies only dielectric *sign*). A difference I can't attribute to a single
manipulated variable is a correlation, not a finding.

## 3. No data leakage, ever
Split by original scene id so augmented/paired variants never straddle train/test
(`src/data/build_splits.py`). State the split protocol in every write-up. Leakage is the
fastest way to an inflated, indefensible result.

## 4. Quantify uncertainty; don't oversell
Report >=5 seeds, bootstrap CIs, and effect sizes, not single numbers. Avoid "SOTA" / "proves"
language given small effective N. Weak, mixed, or negative results are reported plainly and are
just as valuable (`ml_report_rules.md`). A credible negative beats an oversold positive.

## 5. Ground claims in physics and cite honestly
Synthetic data must be validated against published physics before use (the fidelity gate).
Cite the real sources for every dielectric value, dataset, and method (`papers/sources.md`).
Acknowledge prior work generously; my novelty is specific, not "first at everything."

## 6. Separate what I showed from what I hope
Every quantitative claim states its scope. I characterize a *domain shift* under controlled
conditions with a real-data anchor; I do **not** claim a deployable field system. Aspirational
directions are labeled aspirational.

## 7. Reproducibility is part of the result
One environment (`paleo-gpr-ml`, pinned in `environment.lock.yml`), fixed+recorded seeds,
versioned configs/manifests, code-not-notebooks for anything load-bearing, tests for any physics
or data logic. Someone else should be able to rerun me from the repo.

## 8. Log everything I do
The process, decisions, alternatives considered, blockers, dead ends, goes in `docs/notes/`
or the Obsidian vault, in Michael's voice. A result I can't explain how I got is not a result.
(See the memory rule "log everything I do.")

## 9. Test the science, not just the code
Physics invariants and data-pipeline contracts get unit tests (`tests/`). If a physical
assumption is wrong, the synthetic data built on it is wrong, catch it early.

## 10. Let the data pick the story
If the evidence refutes a hypothesis, I report the refutation and pivot the narrative honestly.
The controls are designed so I learn something either way. I never bend the result to fit the
intended story.

---

## Pre-result integrity checklist

Run this before claiming any experimental result or writing it into the paper:

- [ ] Was this comparison pre-registered? If it deviates, is the deviation logged + dated?
- [ ] Is it a controlled comparison (only the intended variable changed)?
- [ ] Splits leak-free and stated?
- [ ] >=5 seeds, CIs, effect sizes reported? Language matched to the evidence (no overclaiming)?
- [ ] Synthetic data passed the physics-fidelity gate?
- [ ] Claims scoped (domain-shift, not field deployment)? Aspirations labeled?
- [ ] Reproducible (env, seeds, configs, manifests) and logged in his voice?
- [ ] If the result is negative/mixed, is it reported as a finding rather than buried?

If any box is unchecked, fix it before the claim leaves the repo.
