# Research Hypothesis & Defensible Contribution

**Date**: 2026-06-10
**Status**: This is the spine of the paper. Everything I build from here serves this.
**Supersedes the framing in**: `docs/notes/05_research_contribution.md` (still valid on "the gap"; this sharpens it into a testable claim).

---

## Why I rewrote the framing

My original pitch was "first ML approach to fossil GPR prospecting." After a literature pass (June 2026) I'm retiring "first" as the headline. Two reasons:

1. **"First to do X" is fragile.** If a reviewer finds one paper I missed, the claim is gone. A contribution should still be worth something even if someone else has touched the space.
2. **My infrastructure dataset is now a published open benchmark** (the 2,239-image Morocco utilities/voids/intact set, Bestagini/El Mahdaoui et al., *Data in Brief*, via PMC11847285). So detecting voids/utilities is a solved, public problem. I can't claim novelty there.

What the literature *does* tell me: **sim-to-real for GPR (gprMax synthetic data + domain adaptation) is an accepted, active research paradigm** (physics-guided DA, adversarial DA such as DDA-GPR for cavities, multi-source DA for IEDs). So my *method* is community-valid. The empty cell is the **fossil/high-permittivity target domain**, and, more sharply, a **characterization of that specific domain shift**. That's where the defensible science is.

## The bar I'm holding this to

A hypothesis worth a sole-author paper must be: **falsifiable**, **specific & measurable**, **mechanistically grounded** (not hand-wavy), **novel vs. the literature**, **methodologically accepted**, **actionable for the community**, and **defensible against the obvious reviewer attacks**. I check H1 against all of these at the bottom.

---

## H1, the central hypothesis

> **A subsurface-anomaly detector trained on void/utility GPR data does not transfer to high-permittivity fossil-like targets out of the box, because the dominant reflection polarity inverts (high-ε bone -> negative top reflection; low-ε void -> positive) and the bone contrast is weaker (~1.7x in dry sand). Physics-grounded synthetic bone data plus polarity-aware adaptation measurably closes this gap.**

The mechanism is not speculation, the 1D forward model already demonstrated the polarity inversion and the contrast ratio (`docs/notes/08_forward_model_1d_validation.md`). H1 promotes that physical fact into a claim about **machine-learning transfer**, which is the novel and useful part.

### Sub-hypotheses (each is independently falsifiable)

- **H1a, the gap exists.** A void-trained detector's AP on fossil-like (bone) targets is substantially lower than the matched-domain ceiling (a bone-trained detector on bone). *Prediction:* large relative AP drop. *Falsified if* void-trained transfers to bone with no meaningful drop.
- **H1b, polarity is the cause (the causal control).** Holding target geometry, depth, and |contrast| fixed and flipping only the *sign* of the dielectric contrast (bone-like high-ε vs an "anti-bone" low-ε target of equal |contrast|), a void-trained detector detects the void-polarity target but not the bone-polarity one. *This isolates polarity from "synthetic just looks different" and from contrast magnitude.* *Falsified if* the drop persists regardless of polarity (then the gap is something else, still publishable, just a different story).
- **H1c, the gap is closable.** Training on physics-grounded synthetic bone, and/or polarity-aware augmentation (random amplitude-sign flip), recovers AP to within Δ of the matched-domain ceiling. *Falsified if* adaptation doesn't help.
- **H1d, it survives contact with reality (the anchor).** A bone-adapted detector outperforms the void-trained baseline on a small set of **real fossil GPR B-scans digitized from the literature** (Catanzariti theropod, Peredo whale, Bargiano). *Falsified if* it does no better than baseline on real data, which would itself be an important, honest negative result about sim-to-real for fossils.

---

## Experimental design (controlled and fair)

The whole point is to **isolate the variable**. I build a controlled synthetic benchmark where scenes are identical except for the thing under test.

1. **Controlled synthetic set (gprMax):** fixed scene generator; vary only target dielectric (void ε=1 / bone ε≈7-12 / null rock ε≈host), plus host medium (dry sand, wet sand, clay, limestone), depth, and frequency (400 MHz, 800 MHz, 2 GHz). Geometry and clutter distribution held constant across target types so polarity/contrast are the only systematic difference.
2. **Training conditions (the matrix):**
   - *Ceiling:* train & test same target domain (void→void, bone→bone).
   - *Naive transfer:* train void (incl. the real public infrastructure data), test bone.
   - *Polarity control (H1b):* the equal-|contrast| sign-flip comparison.
   - *Adaptation (H1c):* synthetic-bone training, polarity augmentation, and one accepted DA method (e.g. adversarial DA) for comparison to the established toolkit.
3. **Real anchor (H1d):** digitized real-fossil B-scans as a held-out *test-only* set. Small N, treated honestly (see threats).
4. **Physics baseline:** a non-ML polarity-triplet matched filter / attribute detector (per Peredo/Catanzariti), so I can answer "why ML over existing interpretation?" by comparison, not assertion.
5. **Metrics:** detection AP / mAP@0.5, recall at fixed false-positive rate, and a **polarity-confusion analysis** (does the model mis-score targets whose polarity it never saw?). Report bootstrap CIs and multiple seeds, effective N is small, so I quantify uncertainty and avoid SOTA language.
6. **Leakage control:** split by original scene id (already enforced, `src/data/build_splits.py`).

### Decision rule (pre-registered in spirit)

I commit *now* to what counts as support vs. refutation, so I can't move the goalposts:
- H1a supported if naive-transfer AP is below the bone-bone ceiling by a margin whose bootstrap CI excludes zero.
- H1b supported if the void-trained detector's AP on the void-polarity target exceeds its AP on the equal-|contrast| bone-polarity target, CI excluding zero.
- H1c supported if an adaptation condition recovers AP to within a pre-set Δ of the ceiling.
- H1d supported if adapted > baseline on the real digitized set (directional, given small N).

A clean refutation of any sub-hypothesis is a result I report, not a failure. That is the reason for running the controls.

---

## Threats to validity, and how I defend each

1. **Synthetic→synthetic circularity ("says nothing about real fossils").** *Defense:* H1d real digitized anchor + framing every quantitative claim as being about the *domain shift*, not absolute field performance. I never claim a deployable fossil finder; I claim a characterized, mitigated transfer gap.
2. **"Polarity inversion is textbook physics."** *Defense:* agreed, and I say so. The novelty is its *measured consequence for ML transfer* and a mitigation, not the physics. The causal control (H1b) is the new knowledge.
3. **"Why ML over physics-based interpretation?"** *Defense:* the physics baseline (#4). I show ML matches/exceeds it while removing the per-B-scan expert, which is the actual value proposition (scalability/automation over large surveys).
4. **Synthetic fidelity.** *Defense:* validate sims against the published forward models, reproduce Peredo's polarity triplet and Catanzariti's hyperbola, and report fidelity honestly instead of assuming it.
5. **Digitized real data is imperfect** (JPEG figures, no raw traces, copyright). *Defense:* use it as a *qualitative-to-semi-quantitative* test-only anchor, cite sources, seek author permission where needed, and be explicit about its limits. It is a sanity check on reality, not a benchmark.
6. **Small effective N.** *Defense:* uncertainty quantification, multiple seeds, no overclaiming.

---

## The contribution (what the community actually gets)

Primary claim is the **mechanistic transfer study**; the rest are supporting artifacts:

1. **(Primary, knowledge)** First characterization, *with a causal control*, of the void→fossil GPR domain shift: polarity inversion + weaker contrast as a concrete, measured reason infrastructure-trained detectors don't directly serve paleontology.
2. **(Supporting, method)** A polarity-aware adaptation recipe that closes the gap, benchmarked against an accepted DA method.
3. **(Supporting, resource)** An open, physics-grounded synthetic fossil-GPR benchmark + baseline detectors + protocol, reproducible from this repo.
4. **(Supporting, positioning)** An honest sim-to-real roadmap for the paleontology-GPR community, including a physics baseline comparison.

This is defensible because it's true even if someone else enters the space: the *finding* and the *causal isolation* stand on their own, and the artifacts are reusable regardless.

## Venue fit

*Remote Sensing*, *Near Surface Geophysics*, *Computers & Geosciences*, or *Scientific Reports*. A controlled study + open benchmark + honest scope is exactly what these reward; none require a deployed field system.

## Does H1 clear the bar?

| Criterion | H1 |
|---|---|
| Falsifiable | Yes, 4 sub-hypotheses with pre-set decision rules |
| Specific & measurable | Yes, AP/recall, polarity-confusion, defined deltas |
| Mechanistically grounded | Yes, forward-model physics (note 08), causal control |
| Novel vs. literature | Yes, fossil target domain + polarity-driven shift is unstudied |
| Methodologically accepted | Yes, sim-to-real + DA is established for GPR |
| Actionable for community | Yes, recipe + open benchmark + physics baseline |
| Defensible | Yes, real anchor + controls + honest framing answer the standard attacks |

---

## Next steps (this reorders `docs/planning/next_steps_todo.md`)

1. Write `docs/experiments/experiment_03_transfer.md`, the formal experiment spec for H1a-d (conditions, metrics, decision rules) before writing model code. Pre-registering keeps me honest.
2. Build `src/data/generate_gprmax_models.py`, now explicitly producing the **controlled** void / bone / anti-bone / null sets the design needs (not just "fossil-like targets").
3. Start the **real-fossil digitization** task: collect B-scan figures from Catanzariti/Peredo/Bargiano, extract test crops, log provenance + permissions in `papers/sources.md`.
4. Implement the **physics baseline** (polarity-triplet matched filter), I already have `instantaneous_amplitude` in `src/features/build_features.py` to build on.
