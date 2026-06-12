# Vision & Goals: Why I'm Doing This

**Date**: 2026-06-10
**Status**: North star. Everything in this repo should ladder up to what's written here.

---

## The one-sentence version

I'm building the first machine-learning approach to **fossil prospecting from GPR (ground-penetrating radar) data**, with a long-term focus on **dinosaurs**, because it sits exactly at the intersection of the two things I care about most: deep learning and dinosaurs.

This is the project where my two passions finally meet, and I want it to produce something real, a novel, publishable research paper with my name on it as the sole author.

---

## Why this project exists (the personal part)

I've wanted to be in the dinosaur space for as long as I can remember. I also love machine learning, and I'm building a career in it, I want to become a master of deep learning architectures and the hard problems that come with them. For a long time those felt like two separate lives. This project is how I stop choosing between them.

The honest motivations, written down so I don't lose sight of them:

- **Build something truly novel** that's actually worthwhile for paleontology, using computer vision and modern architectures.
- **Publish a real research paper**, sole author, so I can contribute to the field, know what that feels like, and have my name on the record.
- **Become a master in deep learning.** Earn more, be more, feel fulfilled. I want depth, not surface-level familiarity.
- **Break into paleontology through dinosaurs specifically.** This is the bridge.

I'm writing this here because when the work gets hard or slow, this is the why.

---

## Why this project is worth doing (the research part)

I validated the gap (literature + web search, June 2026). It holds. Three things exist independently and **nobody has connected them**:

1. **GPR can physically detect buried fossils**, but only through physics-based / manual interpretation (FDTD modeling, attribute analysis, Kirchhoff migration). Real dinosaur-relevant cases exist: the Catanzariti 2023 theropod in Sicily, a sauropod quarry in Cretaceous Texas, sirenian bones in Tuscany. One of the GPR-fossil papers (Bargiano, Italy) literally says GPR fossil prospecting is *"currently poorly exploited in paleontology."* None of these used ML.

2. **Deep learning can detect objects in GPR B-scans**, but the targets are buried threats, utility pipes, roads, archaeology, forensic burials. Never fossils.

3. **Deep learning is useful in paleontology**, but for CT segmentation, fossil image classification, morphometrics. Never GPR / subsurface prospecting.

I searched specifically for a 2025-2026 paper combining deep learning + GPR + fossils + synthetic data and found nothing. The intersection is empty. This is a genuine **novel-application** gap, not "incremental improvement on an existing method" but "first application of an existing method to an unstudied problem." That's the kind of contribution a single author can own.

(Full breakdown lives in `docs/reference/literature_matrix.md` and `docs/notes/05_research_contribution.md`. This file is the why; those are the how-do-I-know.)

---

## The dinosaur angle (how I keep it honest)

I want dinosaurs front and center, but I'm not going to pretend the data is something it isn't. Here's the honest framing:

- The dataset I have **right now** is infrastructure GPR (intact / cavities / utilities). That's where annotations exist, so that's where I learn the *mechanics* of anomaly detection in B-scans.
- The dinosaur focus enters through the **synthetic data + transfer** phase: generate GPR B-scans of dinosaur-bone-like targets in gprMax, using bone/matrix dielectric properties grounded in the paleontological GPR papers (size, geometry, permineralized-bone contrast). Then test whether a detector trained on real anomalies transfers to fossil-like targets.
- The published dinosaur GPR cases (Sicily theropod, Texas sauropod) are my physics anchor and my "this is real" justification. Large bones actually help, bigger targets are more GPR-detectable.

So the throughline is: **learn anomaly detection where labels exist -> synthesize dinosaur-bone targets with real physics -> measure transfer -> characterize the domain gap.** Dinosaurs are the destination; infrastructure is the on-ramp.

---

## What "done" looks like (the paper)

**Working title**: *Toward ML-Based Dinosaur Prospecting: Detecting Subsurface Fossil-Like Anomalies in GPR Radargrams with Physics-Based Synthetic Training Data*

To make the contribution credible I need to show:

1. **Baseline anomaly detection works** on the real GPR data (classification done, ResNet18 ~99.5%; detection done, YOLOv8 mAP@0.5 ~0.58-0.60). Proves the method is sound.
2. **Synthetic fossil/dino data is physically realistic** (gprMax validation against the forward-modeling signatures in the paleo GPR papers).
3. **Some degree of transfer** from real anomalies to fossil-like targets. Even partial transfer is interesting; total failure is *also* a result if I characterize why.
4. **The model uses physically meaningful features** (Grad-CAM, feature analysis), not dataset artifacts like "anomaly = bottom of the image."

What I explicitly **don't** need: SOTA infrastructure detection (others did that), perfect fossil detection (this is a first attempt), or real field excavation (aspirational, not required for paper #1).

**Venue candidates**: *Remote Sensing*, *Near Surface Geophysics*, *Computers & Geosciences*, *Frontiers in Earth Science*, *Scientific Reports*. (Interdisciplinary venues are friendliest to a novel-application paper.)

---

## How I want to work (so future-me and any agent stays aligned)

These are the standing priorities. If a decision is unclear, default to whatever serves these:

1. **Novelty first.** Every experiment should move toward the gap, not toward a leaderboard. If something is well-trodden (beating SOTA on pipes), it's not the point.
2. **Publishable rigor.** Honest splits (split by original ID, no leakage), honest metrics, honest writing. Weak or mixed results get reported plainly. I'd rather have a credible negative result than an oversold positive one. (See `ml_report_rules.md`.)
3. **Write everything down, in my voice.** Every meaningful or genuinely cool finding gets logged, either in `docs/notes/` (repo) or my Obsidian vault. Code, comments, and notes should read like *I* wrote them: practitioner tone, first person, blunt about tradeoffs and risks, no fluff. I'm learning as I build, so the notes are half the value.
4. **Master the architectures.** When I have a real choice of model/approach, prefer the one that teaches me the most about modern deep learning, as long as it still serves the research. Depth over convenience.
5. **Sole-author discipline.** This is my contribution. I want to understand every piece well enough to defend it.

---

## Where things live

- **Active checklist**: `docs/planning/immediate_research_todo.md`
- **Long-term plan**: `docs/planning/roadmap.md`
- **Why it's novel**: `docs/notes/05_research_contribution.md` + `docs/reference/literature_matrix.md`
- **Per-experiment notes**: `docs/notes/0X_*.md`
- **Writing style for reports**: `ml_report_rules.md`
- **Repo orientation for agents**: `CLAUDE.md`
- **Personal research journal / cooler half-formed ideas**: Obsidian vault (`~/Documents/Obsidian Vault`)
