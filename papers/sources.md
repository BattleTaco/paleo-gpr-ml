# Sources & Reference Library

**Last updated**: 2026-06-10

This is my master list of every paper and source I'm using for this project. It exists for two reasons:

1. **References for the paper**, so I can cite cleanly later and never lose track of where a claim came from.
2. **My own learning**, this whole project is as much about me becoming a master of deep learning (and learning the GPR + paleontology side from scratch) as it is about the research output. So I'm keeping notes on *what each source teaches me*, not just its bibliographic info.

Everything here is for both my learning and the benefit of the research. When I read something new, it goes in here with a one-liner on why it matters and what I got out of it.

**Where the files live:**
- `papers/references/`, the 6 core PDFs I curated at the start (each has a reading note).
- `papers/references/external/`, PDFs I pulled while researching the gap (open-access only; the rest are links).
- `papers/reading_notes/`, my detailed notes, one per core paper.

---

## Core papers (curated, in repo, with reading notes)

These are the backbone of the literature review. Reading notes are in `papers/reading_notes/`.

| # | Paper | PDF | Note | Why it matters to me |
|---|---|---|---|---|
| 1 | Yu et al., *AI in Paleontology* (review) | `references/AI_in_Paleontology.pdf` | `reading_notes/01_ai_in_paleontology.md` | The big-picture map of where ML is (and isn't) used in paleo. Shows data scarcity is the field's bottleneck. |
| 2 | Peredo et al., *GPR Vertebrate Skeleton Detection* (2024) | `references/GPR_Vertebrate_Skeleton_Detection_2024.pdf` | `reading_notes/02_gpr_vertebrate_skeleton_detection.md` | Proof that GPR physically detects large vertebrate bones. Forward modeling + polarity triplet analysis, my physics anchor for synthetic targets. |
| 3 | Catanzariti et al., *GPR Dinosaur Bones, Sicily* (2023) | `references/GPR_Dinosaur_Bones_Sicily_2023.pdf` | `reading_notes/03_gpr_dinosaur_bones_sicily.md` | **The dinosaur one.** Theropod bones found with 2 GHz GPR + Kirchhoff migration + Hilbert transform. My "this is real for dinosaurs" justification. |
| 4 | Kücükdemirci & Sarris, *GPR AI Review* (2022) | `references/GPR_AI_Review_2022.pdf` | `reading_notes/04_gpr_ai_review_2022.md` | How ML is actually applied to GPR (R-CNN, U-Net, ResNet, gprMax synthetic data), but for archaeology/infrastructure, not fossils. Methods menu. |
| 5 | Yu et al., *Fossil CT Segmentation* (2022) | `references/Fossil_CT_Segmentation_2022.pdf` | `reading_notes/05_fossil_ct_segmentation_2022.md` | DeepLab v3+ for fossil-vs-rock segmentation. Architecture reference + shows the CV-in-paleo side. |
| 6 | Knutsen & Konovalov, *Fossil CT Acceleration* (2024) | `references/Fossil_CT_Acceleration_2024.pdf` | `reading_notes/06_fossil_ct_acceleration_2024.md` | UNet + EfficientNet-V2-XL, 0.96 Dice from only 18 slices. Lessons on training under extreme data scarcity (my situation too). |

---

## External sources found during gap validation (June 2026)

Pulled while confirming the gap is still open. PDFs marked saved are in `papers/references/external/`; the rest are paywalled or bot-blocked, so they're links.

### Confirms the gap (most important)

- **GPR Detection of Fossil Structures in Conductive Media, FDTD + Attributes (Bargiano, Italy)**, MDPI Geosciences 2021. *Link only.*
  https://www.mdpi.com/2076-3263/11/9/386
  Does GPR fossil detection with **FDTD modeling + attribute analysis, NOT machine learning**, and literally says GPR fossil prospecting is *"currently poorly exploited in paleontology."* This is my single best citation that the gap is real. Also a model for how to set up FDTD/gprMax for fossil-like targets.

- **A Review of ML Applications for Identification & Classification in Paleontology**, ScienceDirect 2025. *Link only (paywall).*
  https://www.sciencedirect.com/science/article/pii/S1574954125003383
  Recent survey of ML in paleo. Useful to confirm GPR/subsurface prospecting is absent from the field's ML work, and to position my contribution.

- **Bridging Theory and Practice: AI-Driven Techniques for GPR Interpretation**, MDPI Applied Sciences 2025. *Link only.*
  https://doi.org/10.3390/app15158177
  State-of-the-art review of AI on GPR (CNNs, hybrid physics-informed, multimodal fusion). My reference for "what's current in DL+GPR", none of it is fossils.

- **Advancing Paleontology: Deep Learning in Fossil Image Analysis (survey)**, Springer AI Review 2024. *Link only.*
  https://link.springer.com/article/10.1007/s10462-024-11080-y
  Survey of DL on fossil *images*. Confirms the paleo-DL world is image/CT-focused, not GPR.

### Dinosaur / fossil GPR precedent (physics-based, no ML)

- **Use of GPR in Detecting Fossilized Dinosaur Bones (Texas sauropod quarry)**, ResearchGate. *Link only.*
  https://www.researchgate.net/publication/252789955_Use_of_ground_penetrating_radar_in_detecting_fossilized_dinosaur_bones
  Sauropod bones located + excavated with GPR in Cretaceous Texas. More dinosaur precedent for the intro.

- **GPR Detection of Sirenian Fossil Bones under a Sunflower Field, Tuscany**, ScienceDirect. *Link only.*
  https://www.sciencedirect.com/science/article/pii/S1631068312000796
  Another real fossil-GPR success (sea cow). Breadth for "GPR finds fossils, manually."

### DL + GPR methods I can borrow (infrastructure/threats, transferable techniques)

- **A Deep Learning-Based GPR Forward Solver for Predicting B-Scans**, arXiv 2207.06527. *Saved:* `external/GPR_DL_Forward_Solver_arxiv_2207.06527.pdf`
  Bimodal encoder-decoder that predicts B-scans of buried objects in heterogeneous soil. Directly relevant to fast synthetic-data generation as an alternative/complement to gprMax.

- **GPR Subsurface Object Detection & Reconstruction (DepthNet)**, arXiv 2008.08731. *Saved:* `external/GPR_DepthNet_arxiv_2008.08731.pdf`
  Detects B-scan hyperbolas, denoises, predicts dielectric to get depth. Good architecture reference for the detection side.

- **Automatic Road Subsurface Distress Recognition via DL Cross-Verification**, arXiv 2507.11081. *Saved:* `external/GPR_Road_Subsurface_DL_arxiv_2507.11081.pdf`
  Recent (2025) DL-on-GPR detection pipeline. Cross-verification idea could reduce false positives on rare targets like fossils.

- **DL + Geometric Modeling for 3D Reconstruction of Subsurface Utilities from GPR**, PMC12567710. *Link only.*
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12567710/
  3D from GPR. Aspirational direction if I ever go from 2D B-scan detection to 3D fossil reconstruction.

---

### Sim-to-real / domain adaptation for GPR (method grounding for H1)

Found while pressure-testing the hypothesis (2026-06-10). These establish that my *method* (synthetic training + domain adaptation) is an accepted, active paradigm, so reviewers won't reject the approach, only the execution. All *link only*.

- **Physics-guided hierarchical domain adaptation with deep adversarial learning (subsurface radar)**, arXiv 2512.17831 (2025).
  https://arxiv.org/abs/2512.17831
  Closest methodological cousin: bridges simulated->real GPR with physics-guided adversarial DA. Cite as the state of the art my transfer recipe sits next to. Differentiator: they do material-property estimation on infrastructure; I do detection on the fossil target domain with a polarity mechanism.

- **DDA-GPR: symmetric adversarial domain adaptation for few-shot underground cavity identification**, ScienceDirect 2025.
  https://www.sciencedirect.com/science/article/abs/pii/S0263224125026272
  Adversarial DA for *cavities* specifically. Direct related work and a baseline DA method to compare my polarity-aware adaptation against (H1c).

- **Multi-source domain adaptation of GPR data for IED detection**, Springer SIViP 2022.
  https://link.springer.com/article/10.1007/s11760-022-02394-x
  Establishes cross-domain DA between GPR target types. Supports that "transfer between GPR domains" is a recognized problem; my novelty is the *fossil* domain + the polarity cause.

- **Public infrastructure GPR dataset (utilities / voids / intact, Morocco, 400 & 200 MHz)**, Data in Brief, PMC11847285.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11847285/
  **This is (the published version of) my training dataset.** Cite it as the source; note that it means infrastructure detection is a solved/public task and cannot be my novelty.

### Real-fossil GPR figures to digitize (H1d test anchor, TODO)

Per the plan, build a small *real* fossil-GPR test set by digitizing B-scan figures from these (cite + seek permission; log crops + provenance here as I extract them):
- Catanzariti et al. 2023 (theropod, Sicily), `references/GPR_Dinosaur_Bones_Sicily_2023.pdf`
- Peredo et al. 2024 (whale skeleton, Peru), `references/GPR_Vertebrate_Skeleton_Detection_2024.pdf`
- Bargiano fossil-GPR FDTD paper (Italy), link in the gap-validation section above.

## How I add to this list

When I read or find something new:
1. Add a row/bullet here with the link, where the PDF lives (if saved), and a one-line "why it matters to me."
2. If it's a core paper I'll cite heavily, save the PDF and write a full reading note in `papers/reading_notes/`.
3. If it changes the gap or the plan, update `docs/notes/05_research_contribution.md` and `docs/reference/literature_matrix.md`.

Keep it honest, note what a source actually shows, and where it stops short of what I'm trying to do.
