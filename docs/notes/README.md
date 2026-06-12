# Research journal

These are my notes in the order I did the work. Each one is a record of what I set out to do,
what I found, what broke, and what I decided. I read these to remember the path, not just the
result.

- `00_vision_and_goals.md` why this project exists and what done looks like. The north star.
- `01_gpr_eda_notes.md` first look at the GPR dataset, what is in it and what would bite me later.
- `02_baseline_experiment_notes.md` the classification baseline (SimpleCNN, ResNet18).
- `03_literature_synthesis.md` what the papers tell me, pulled together.
- `04_gradcam_notes.md` what the classifier actually looks at.
- `05_research_contribution.md` the early gap framing. Superseded by 09, kept for the record.
- `06_detection_experiment_notes.md` the YOLOv8 detection baseline.
- `07_environment_setup.md` the dedicated conda env, and the OpenMP fix.
- `08_forward_model_1d_validation.md` the 1D physics check. Where the bone vs cavity polarity
  inversion shows up.
- `09_research_hypothesis.md` the spine of the paper. H1 and why it is defensible.
- `10_synthetic_generator.md` the controlled gprMax generator (void, bone, anti_bone, null).
- `11_polarity_baseline.md` the non-ML physics baseline (polarity matched filter).
- `12_cuda_runbook.md` how to set up the RTX 5090 / Blackwell machine and run the synthetic data step.

The experiment specs these notes refer to live in `../experiments/`. The active to-do list is
`../planning/next_steps_todo.md`. The runnable notebook for the synthetic step is
`notebooks/05_synthetic_data_generation.ipynb`.
