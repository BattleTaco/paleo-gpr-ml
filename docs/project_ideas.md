# Project Ideas

## Core Research Questions
1. Can ML models trained on synthetic GPR data generalise to real field data?
2. What feature representations best capture bone / fossil hyperbolic reflections in GPR B-scans?
3. How does GPR-based detection performance vary with soil type, moisture, and burial depth?

## Potential Extensions
- **Multi-modal fusion**: combine GPR with surface imagery or LiDAR
- **Active learning**: prioritise field scans that maximise information gain
- **3D GPR reconstruction**: volumetric representation of subsurface anomalies
- **Transfer learning**: leverage pre-trained weights from construction / utility GPR domains

## Dataset Ideas
- Synthetic GPR using gprMax or similar FDTD simulators
- Museum CT scans of bones as proxy labels
- Publicly available archaeological GPR surveys

## Open Questions
- Ground-truth acquisition strategy in the field
- Annotation methodology for GPR B-scans (bounding box vs. pixel-level mask)
- Evaluation metrics appropriate for sparse anomaly detection
