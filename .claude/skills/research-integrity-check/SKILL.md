---
name: research-integrity-check
description: Run the pre-result integrity checklist for paleo-gpr-ml before claiming an experimental result, writing a finding into the paper/notes, or reporting numbers. Use when an experiment finishes, before stating a result as established, or when Michael asks "is this defensible / are we following the guidelines / can we report this".
---

# Research Integrity Check

A gate to keep paleo-gpr-ml publishable and defensible. Run this BEFORE any experimental
result is stated as established or written toward the paper. Source of truth:
`docs/research_guidelines.md` and the pre-registration in `docs/experiments/experiment_03_transfer.md`.

## How to run it

Go through the checklist against the actual result in front of you. For each item, state
PASS / FAIL / N-A with one line of evidence (a file, a number, a config). Don't rubber-stamp --
if you can't point to evidence, it's a FAIL.

1. **Pre-registered?** Was this comparison/metric/decision-rule fixed in `experiment_03_transfer.md` before running? If it deviates, is the deviation logged with date + reason?
2. **Controlled?** Did only the intended variable change? (e.g. the `anti_bone` control varies only dielectric sign.) Any uncontrolled confound = not a causal claim.
3. **Leak-free?** Split by original scene id; protocol stated. No augmented/paired variant straddles train/test.
4. **Uncertainty?** >=5 seeds, bootstrap CIs, effect sizes reported. Language matches the evidence (no "SOTA"/"proves" on small N).
5. **Physics-grounded?** If synthetic data is involved, did it pass the fidelity gate (reproduce Peredo triplet / Catanzariti hyperbola)? Dielectric values + datasets + methods cited?
6. **Scoped?** Claims framed as domain-shift characterization, not field deployment. Aspirations labeled aspirational.
7. **Reproducible?** Run in the `paleo-gpr-ml` env; seeds, configs, manifests versioned; load-bearing logic in code + tested.
8. **Logged?** The process (decisions, alternatives, blockers) recorded in `docs/notes/` or the vault, in Michael's voice.
9. **Honest about negatives?** If the result is weak/mixed/refuting, is it reported as a finding rather than buried? Did we let the data pick the story?
10. **Written in Michael's voice, no AI tells?** Any code, comments, notes, or text headed for the repo or paper must read like he wrote it. No em dashes (and no "--" used as a dash), no AI filler or formulaic openers. Match his Obsidian Diary/Skill-Session voice, professional for research notes. See the `feedback-write-in-michaels-voice` memory.

## Output

- A short PASS/FAIL table with evidence per item.
- If anything FAILs: list exactly what to fix before the claim leaves the repo, and don't
  green-light the result until fixed.
- If all PASS: say so plainly, and note which paper figure/table (T1, F1-F6 in
  experiment_03_transfer.md) the result feeds.

Keep it honest -- the job of this skill is to catch problems a reviewer would, while they're
still cheap to fix.
