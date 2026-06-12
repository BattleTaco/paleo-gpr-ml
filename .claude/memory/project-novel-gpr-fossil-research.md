---
name: project-novel-gpr-fossil-research
description: "The core goal of paleo-gpr-ml — first ML approach to dinosaur/fossil prospecting from GPR, aimed at a sole-author paper"
metadata: 
  node_type: memory
  type: project
  originSessionId: e6c235d4-3a76-4721-a696-dfcdc468b126
---

The primary goal of paleo-gpr-ml is to build something **truly novel and publishable**: the **first machine-learning approach to fossil prospecting from GPR (ground-penetrating radar) radargrams**, with a long-term **dinosaur** focus. Michael's target is a **novel research paper, sole author**, to contribute to the field and advance his ML career.

**The validated gap (checked via literature + web search, June 2026 — it holds):** Three things exist independently and nobody has connected them: (1) GPR physically detects fossils but only via physics/manual interpretation (FDTD, Kirchhoff migration; e.g. Catanzariti 2023 Sicily theropod, Texas sauropod quarry); (2) deep learning detects GPR objects but for infrastructure/threats/archaeology, never fossils; (3) deep learning is used in paleontology but for CT segmentation / image classification, never GPR. No 2025–2026 paper combines DL + GPR + fossils + synthetic data. This is a novel-*application* gap, not an incremental one.

**The defensible thesis (sharpened 2026-06-10, full spec in `docs/notes/09_research_hypothesis.md`):** Retired "first ML approach" as the headline (novelty-by-absence is fragile; our infrastructure dataset is a *published* public benchmark — PMC11847285 — so detecting voids/utilities isn't novel). The paper is a **mechanistic transfer study**, central hypothesis **H1**: *a void/utility-trained detector fails to transfer to high-permittivity fossil-like targets because reflection polarity inverts (bone high-ε → negative top reflection; void low-ε → positive) and bone contrast is ~1.7× weaker; physics-grounded synthetic bone + polarity-aware adaptation closes the gap.* Falsifiable sub-hypotheses H1a–d with pre-set decision rules, a **causal polarity control** (equal-|contrast| sign flip), a **physics baseline** (polarity-triplet matched filter), and a **real-fossil anchor: digitize B-scans from Catanzariti/Peredo/Bargiano** (user-chosen — defends against the synthetic-only critique). Method (sim-to-real + domain adaptation) is community-accepted; novelty is the fossil target domain + the polarity mechanism.

**The throughline:** learn detection on the real infrastructure GPR (on-ramp) → generate **controlled** synthetic void/bone/anti-bone/null targets in gprMax → measure + causally isolate the domain shift → close it → sanity-check on digitized real fossil B-scans. Dinosaurs are the destination; infrastructure is the on-ramp. Stay honest the *current* dataset is infrastructure, not dinosaurs.

**Done so far:** EDA, classification baseline (ResNet18 ~99.5%), Grad-CAM analysis, YOLOv8 detection (mAP@0.5 ~0.58–0.60). **Next major phase:** synthetic data (gprMax) + transfer.

Full north star: `docs/notes/00_vision_and_goals.md`. Novelty/lit detail: `docs/notes/05_research_contribution.md`, `docs/reference/literature_matrix.md`. See [[research-priorities-paleo-gpr]] for standing priorities and [[user-michael-ml-paleo]].
