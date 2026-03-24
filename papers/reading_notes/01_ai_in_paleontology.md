# Reading Notes: Artificial Intelligence in Paleontology

**Citation**: Yu, C. et al. (2024). Artificial intelligence in paleontology. *Earth-Science Reviews*, 252, 104765.
**Read**: 2026-03-23

---

## What this paper is about

Comprehensive review of 70+ studies using AI in paleontology. Covers the full spectrum: fossil identification, classification, segmentation, morphological analysis, phylogenetics, taphonomy, and paleoecology. The goal is to map where AI has been applied in the field and where it hasn't.

## Key findings that matter for our project

**Fossil image classification is an active area.** Multiple groups have used CNNs for classifying fossil types from photographs or thin sections. Accuracy is generally high (85-98%) when there's enough training data and the classes are visually distinct. This tracks with what we saw in our baseline classification experiment.

**CT segmentation is the most mature DL application in paleo.** U-Net and DeepLab variants are the go-to architectures. The key challenge isn't the model, it's the annotation bottleneck. Manual segmentation of CT slices is brutally time-consuming, and most datasets are small (hundreds to low thousands of slices).

**GPR + paleontology is barely represented.** This is the gap. The review mentions geophysical prospecting as a data source for paleontology but doesn't cite any ML work specifically on GPR for fossil detection. The closest references are to archaeological GPR detection. This is directly relevant because it means our project sits in a genuine gap in the literature.

**Data scarcity is the universal problem.** Every single application area in this review struggles with limited labeled data. Transfer learning, synthetic data, and semi-supervised methods are repeatedly suggested as solutions but rarely implemented well.

**Morphometric analysis via deep learning is growing.** Automated landmarking, shape analysis, and geometric morphometrics using neural networks. Not directly relevant to Stage 1 but could inform Stage 2 fossil classification.

## What I want to steal from this paper

- The framing of paleontological AI as a tool for augmenting expert judgment, not replacing it. Good for positioning our work.
- The taxonomy of tasks (detection, classification, segmentation, morphometrics) maps nicely onto a staged research plan.
- The explicit call-out that geophysical prospecting + ML for paleontology is an open area. This validates our research direction.

## Limitations

- It's a review paper, so the depth on any single method is thin.
- Published 2024, so it covers up through roughly 2023 literature. Recent GPR work may not be included.
- Doesn't really evaluate or compare the methods it surveys. It's more of a catalog.

## Relevance to our project: HIGH

Validates that GPR-based fossil prospecting with ML is an underexplored niche. Useful for framing our contribution in a paper. The CT segmentation sections are relevant to Stage 2 but not the immediate focus.

## 3-2-1

**3 things I learned:**
1. Nobody has published ML-based fossil detection from GPR data specifically. Archaeological GPR detection exists, but the fossil application is genuinely novel.
2. The most common DL architecture in paleo CT work is U-Net, and the best results come from domain-specific fine-tuning rather than training from scratch.
3. Data augmentation in paleontology studies is usually basic (flips, rotations, brightness). More sophisticated synthetic data pipelines are rare.

**2 things I still don't know:**
1. How much of the archaeological GPR detection work (which does exist) transfers to paleontological contexts. The physics is similar but the targets are different.
2. Whether the review missed any GPR+fossil work that was published in geophysics journals rather than paleo journals.

**1 thing I want to investigate:**
Whether our approach (training on infrastructure GPR data, then applying to fossil-like synthetic targets) has any precedent in the transfer learning literature for geophysics.
