# Next Steps TODO

**Last updated**: 2026-03-23
**Context**: EDA, literature review, classification baseline, Grad-CAM analysis, and object detection baseline are all complete. The project is transitioning from "can we detect anomalies in infrastructure GPR?" (yes) to "can we detect fossil-like anomalies?" (the actual research question).

---

## Where we are

### Completed
- [x] EDA on GPR B-scan dataset (`notebooks/01_gpr_data_exploration.ipynb`, 25+ figures)
- [x] Data understanding doc (`docs/data_understanding.md`)
- [x] Leak-proof train/val/test splits by original image ID (`src/data/build_splits.py`)
- [x] Classification baseline: SimpleCNN 96.7%, ResNet18 99.5% (`notebooks/02_baseline_classification.ipynb`)
- [x] Literature review: 6 papers read, reading notes for all in `papers/reading_notes/`
- [x] Literature matrix with gap analysis (`docs/literature_matrix.md`)
- [x] Literature synthesis (`docs/notes/03_literature_synthesis.md`)
- [x] Research contribution framing (`docs/notes/05_research_contribution.md`)
- [x] Grad-CAM analysis (`notebooks/03_gradcam_analysis.ipynb`, 10 figures)
  - Cavities: 51% attention in bbox, 88% bbox covered (model sees the anomaly)
  - Utilities: 18% attention in bbox (model uses broader context, not just the target)
- [x] Object detection baseline (`notebooks/04_object_detection.ipynb`)
  - YOLOv8n 2-class: mAP@0.5 = 0.581
  - YOLOv8n binary: mAP@0.5 = 0.600
  - False positive rate on intact: 0.7%
- [x] Detection data prep script (`src/data/prepare_detection_data.py`, both 2class and binary modes)
- [x] Synthetic data plan written (`docs/synthetic_data_plan.md`)
- [x] Detection experiment spec (`docs/experiment_02_detection.md`)

### Key results so far
- Classification is trivially easy on this dataset (99.5%). Not the interesting problem.
- Detection is meaningfully harder (mAP 0.60). This is the real task.
- Binary anomaly detection (cavity + utility = "anomaly") works as well as 2-class and is the right framing for fossil prospecting.
- The model learns physically meaningful features for cavities but uses whole-image context for utilities.
- False positive rate is near-zero (0.7%), meaning the detector cleanly separates "something" from "nothing."
- Literature confirms: nobody has done ML + GPR + fossils. This is a genuine gap.

---

## What to do next (in priority order)

### 1. Synthetic GPR Data Generation with gprMax

**Why**: We need synthetic fossil-like targets to test whether the detector transfers from infrastructure anomalies to paleontological targets. This is the core research question.

**Plan is written**: `docs/synthetic_data_plan.md` has full details. Key parameters:

| Parameter | Values |
|---|---|
| Bone permittivity | 7-12 (from Peredo et al. 2024) |
| Sediment permittivity | 3-5 (dry sand), 10-25 (wet) |
| Target shapes | Elliptical bone fragment, bone cluster, articulated skeleton approx |
| Depths | 0.3-1.5 m |
| Frequencies | 400 MHz, 800 MHz |
| Host media | Dry sand, wet sand, dry clay, loam |
| Target dataset size | ~475 synthetic B-scans |

**Steps**:
1. Install gprMax (`pip install gprMax` or build from source). Test with a simple model.
2. Create `src/data/generate_gprmax_models.py` -- script that programmatically generates gprMax input files with parameterized target geometry, depth, host medium, and antenna frequency.
3. Run simulations. Each model produces one synthetic B-scan. Start with a small batch (10-20) to validate, then scale up.
4. Create `src/data/process_gprmax_output.py` -- convert gprMax HDF5 output to 224x224 JPEG images matching our real data format. Generate corresponding YOLO annotation files from known target positions.
5. **Validation checks before training**:
   - Visual plausibility: do synthetic B-scans look like real GPR data?
   - Polarity triplet: verify bone targets produce the positive-negative-positive pattern from Peredo et al.
   - Hyperbola shape: compare synthetic bone hyperbolas with real utility hyperbolas
   - Add realistic noise to synthetic data
6. Create `notebooks/05_synthetic_data_exploration.ipynb` -- visualize and validate the synthetic data, compare with real data side by side.
7. Write notes in `docs/notes/07_synthetic_data_notes.md`.

**Risk**: gprMax simulations can be slow. If compute is a bottleneck, start with 2D models (faster) and a smaller dataset. The plan calls for ~475 B-scans, but even 50-100 would be enough for initial transfer experiments.

### 2. Transfer Learning Experiment (Synthetic -> Detection)

**Why**: This is the experiment that answers the research question. Can a detector trained on infrastructure GPR data find fossil-like targets?

**Steps**:
1. **Phase 1: Infrastructure-only baseline** (already done). YOLOv8n binary detector, mAP@0.5 = 0.600.
2. **Phase 2: Synthetic pretraining**. Pretrain YOLOv8n on synthetic data (all target types including fossil-like), then fine-tune on real infrastructure data. Compare with Phase 1.
3. **Phase 3: Transfer to fossil targets**. Take the Phase 2 model and test on held-out synthetic fossil-only B-scans. Can it detect fossil-like targets it was pretrained on but never saw during fine-tuning?
4. **Phase 4 (aspirational)**: If we can find any published raw GPR data from a paleontological site, test on that.
5. Create `notebooks/06_transfer_experiment.ipynb`.
6. Write up in `docs/notes/08_transfer_experiment_notes.md`.

**Metrics to report**: mAP@0.5 at each phase, per-target-type AP, false positive rate, visual examples of detections on synthetic fossil data.

### 3. Experiment with Additional Detection Architectures

**Why**: The literature uses both YOLO and Faster R-CNN for GPR detection. Comparing architectures strengthens the paper.

**Steps**:
1. Train Faster R-CNN (torchvision, ResNet50-FPN backbone) on same detection data.
2. Compare mAP with YOLOv8n.
3. Optionally try YOLOv8s (small) to see if a larger model helps.
4. Add results to `results/tables/detection_results.csv`.

**Lower priority than synthetic data. Do this if time permits or if YOLO results are weak.**

### 4. Deeper Grad-CAM / Feature Analysis

**Why**: The Grad-CAM results for utilities were surprising (model doesn't focus on the bbox). Understanding this better could inform the paper narrative.

**Steps**:
1. Run Grad-CAM on intermediate layers (not just layer4) to see where the model starts distinguishing classes.
2. Try Grad-CAM++ or Score-CAM for comparison.
3. Compute feature similarity between cavity regions and utility regions in the model's embedding space.
4. Test whether the model's features cluster by class in a meaningful way (t-SNE or UMAP on penultimate layer features).
5. Add to `notebooks/03_gradcam_analysis.ipynb` or create a new notebook.

**Lower priority. Do after synthetic data experiment.**

### 5. Paper Writing Preparation

**Why**: The project has enough results for at least a workshop paper or short conference paper.

**Steps**:
1. Create `docs/paper_outline.md` with the narrative structure:
   - Problem: fossil prospecting is slow, GPR can detect bones, ML can detect GPR anomalies, nobody has combined them
   - Method: train on infrastructure GPR, generate synthetic fossil targets, test transfer
   - Results: classification baseline, detection baseline, Grad-CAM analysis, synthetic data quality, transfer results
   - Discussion: what worked, what didn't, implications for fieldwork
2. Collect all figures into a coherent set. Current inventory:
   - EDA: 16 figures in `results/figures/`
   - Grad-CAM: 10 figures
   - Detection: 3 figures + YOLO training curves
   - Synthetic data: TBD
3. Draft the methods section first (easiest, most concrete).
4. Identify target venue (see `docs/notes/05_research_contribution.md` for options).

**Start this after the synthetic data and transfer experiments are done.**

---

## File inventory (what exists where)

### Notebooks
| File | Status | Description |
|---|---|---|
| `notebooks/01_gpr_data_exploration.ipynb` | Complete | EDA, 25+ figures |
| `notebooks/02_baseline_classification.ipynb` | Complete | SimpleCNN + ResNet18, 99.5% acc |
| `notebooks/03_gradcam_analysis.ipynb` | Complete | Grad-CAM on both models, bbox overlap |
| `notebooks/04_object_detection.ipynb` | Complete | YOLOv8n 2-class + binary, mAP 0.60 |
| `notebooks/05_synthetic_data_exploration.ipynb` | **Not created** | Validate synthetic gprMax data |
| `notebooks/06_transfer_experiment.ipynb` | **Not created** | Synthetic pretrain -> fossil detection |

### Scripts
| File | Status | Description |
|---|---|---|
| `src/data/build_splits.py` | Complete | By-ID train/val/test splits |
| `src/data/prepare_detection_data.py` | Complete | YOLO-format detection data |
| `src/data/generate_gprmax_models.py` | **Not created** | Generate gprMax input files |
| `src/data/process_gprmax_output.py` | **Not created** | Convert gprMax output to images |

### Documentation
| File | Status | Description |
|---|---|---|
| `docs/research_log.md` | Up to date | 4 dated entries |
| `docs/data_understanding.md` | Complete | Dataset documentation |
| `docs/literature_matrix.md` | Complete | 6 papers + gap analysis |
| `docs/synthetic_data_plan.md` | Complete | gprMax parameters and strategy |
| `docs/experiment_01_baseline.md` | Complete | Classification spec |
| `docs/experiment_02_detection.md` | Complete | Detection spec |
| `docs/immediate_research_todo.md` | Phase 1 complete | Updated with Phase 2 items |

### Notes
| File | Description |
|---|---|
| `docs/notes/01_gpr_eda_notes.md` | EDA findings |
| `docs/notes/02_baseline_experiment_notes.md` | Classification results |
| `docs/notes/03_literature_synthesis.md` | Cross-paper synthesis |
| `docs/notes/04_gradcam_notes.md` | Grad-CAM findings |
| `docs/notes/05_research_contribution.md` | Novelty framing |
| `docs/notes/06_detection_experiment_notes.md` | Detection results |

### Reading notes
All 6 in `papers/reading_notes/01_*.md` through `06_*.md`.

### Models
| File | Description |
|---|---|
| `results/models/simple_cnn_best.pt` | SimpleCNN 96.7% |
| `results/models/resnet18_best.pt` | ResNet18 99.5% |
| `results/detection/yolov8n_2class/weights/best.pt` | YOLOv8n 2-class mAP 0.581 |
| `results/detection/yolov8n_binary/weights/best.pt` | YOLOv8n binary mAP 0.600 |

---

## Quick-start for next session

```bash
conda activate paleo-gpr-ml
cd /home/michael-ramirez/GitHub/personal/paleo-gpr-ml

# Check what's new
git status
cat docs/next_steps_todo.md

# Next action: install gprMax and start synthetic data generation
# See docs/synthetic_data_plan.md for full details
```

**First thing to do**: Install gprMax, create a simple test model, verify it runs, then start building the parameterized model generation script.
