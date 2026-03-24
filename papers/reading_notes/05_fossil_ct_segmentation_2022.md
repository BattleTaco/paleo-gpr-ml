# Reading Notes: CT Segmentation of Dinosaur Fossils by Deep Learning (2022)

**Citation**: Yu, C., Qin, F., Li, Y., Qin, Z. & Norell, M. (2022). CT Segmentation of Dinosaur Fossils by Deep Learning. *Frontiers in Earth Science*, 9, 805271.
**Read**: 2026-03-23

---

## What this paper is about

Applies U-Net and modified DeepLab v3+ to segment dinosaur fossils from CT scans. Dataset is CT slices of three protoceratopsian skulls from the Gobi Desert, Mongolia. The goal is to automate the tedious manual segmentation of fossil bone from surrounding rock matrix.

## Key technical details

**Dataset:**
- 3 protoceratopsian skulls (2 embryonic Protoceratops, 1 embryonic Protoceratopsia)
- CT scanned at high resolution (voxel sizes 21-23 microns)
- 7986 training slices, 3329 testing slices
- Manually segmented and cross-validated in axial, sagittal, and coronal planes using Mimics 19.0

**Models tested:**
- Classic U-Net: 98.09% accuracy on test, but high cross-entropy loss (4.94/9.92 for test/val). Accuracy is misleading here because most pixels are background.
- DeepLab v3+ with MobileNet v1 (1 skip connection): mean Dice 0.738, mean IoU 0.612
- DeepLab v3+ with MobileNet v1 (2 skip connections): mean Dice 0.894, mean IoU 0.817 - BEST
- DeepLab v3+ with MobileNet v1 + ASPP concatenated: mean Dice 0.864, mean IoU 0.777
- DeepLab v3+ with ResNet v2_50: mean Dice 0.864, mean IoU 0.773

**Best model**: DeepLab v3+ with MobileNet v1 and 2 skip connections. Mean Dice of 0.894.

**The key challenge**: Fossil bone and surrounding rock matrix often have very similar densities in CT scans, making segmentation hard. Manual segmentation requires expert knowledge of anatomy.

**Generalization test**: Trained model was applied to non-protoceratopsian fossils (Haya, Pinacosaurus) from the same Gobi Desert environment. Results were "extremely poorly segmented." The model does not generalize across taxa.

## What matters for our project

**The generalization failure is the headline result.** A model trained on 3 protoceratopsian skulls cannot segment different dinosaur genera. This is directly analogous to our concern about whether a model trained on infrastructure GPR data will detect fossil-like anomalies.

**Dice and IoU are the right metrics for segmentation.** Not accuracy (which is inflated by background pixels). If we move to segmentation for GPR anomaly detection, these are what we should report.

**Skip connections matter a lot.** Going from 1 to 2 skip connections jumped Dice from 0.738 to 0.894. The coarse-to-fine feature propagation is critical when targets have complex boundaries.

**Small dataset, reasonable results.** ~8000 annotated slices from 3 specimens produced a model with 0.89 Dice on same-taxon data. This suggests that with even modest amounts of real GPR data labeled for anomaly regions, we could get reasonable segmentation performance.

## What I want to steal

- The evaluation framework: Dice + IoU, not accuracy. Visualize predictions overlaid on ground truth.
- The finding that generalization across targets is hard even within a single domain (CT fossils). We should expect the same for GPR and plan accordingly.
- DeepLab v3+ with skip connections as a candidate architecture if we move to segmentation.

## Limitations

- Very specific taxa. 3 skulls of closely related species. Not representative of fossil diversity.
- Manual segmentation as ground truth introduces human bias and inconsistency.
- No domain adaptation or transfer learning from other imaging domains attempted.
- The generalization failure wasn't analyzed in depth. Why does it fail? Different bone density? Different morphology? Different surrounding matrix?

## Relevance to our project: MODERATE-HIGH

Not directly GPR-related, but the segmentation methodology and generalization analysis are relevant. The lesson that DL models in paleontology are brittle to domain shift is one we need to internalize for our GPR work.

## 3-2-1

**3 things I learned:**
1. DeepLab v3+ with MobileNet v1 and 2 skip connections achieves 0.894 mean Dice on fossil CT segmentation, trained on ~8000 slices from 3 specimens.
2. The model completely fails when applied to different dinosaur genera, even from the same sedimentary environment.
3. Pixel accuracy is meaningless for segmentation when background dominates. Dice and IoU are the correct metrics.

**2 things I still don't know:**
1. Whether the generalization failure is due to morphological differences, density differences, or both.
2. Whether pretraining on a large medical CT segmentation dataset before fine-tuning on fossils would improve cross-taxon generalization.

**1 thing I want to investigate:**
Whether the domain gap between infrastructure GPR targets and fossil-like GPR targets is smaller or larger than the gap between protoceratopsian and non-protoceratopsian CT data.
