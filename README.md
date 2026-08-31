# paleo-gpr-ml

Testing whether subsurface anomaly detectors transfer from voids and utilities to
fossil-like targets, and if not, characterizing exactly why.

**Author:** Michael Ramirez (sole author)
**Status:** active. Physics validated, controlled synthetic benchmark built, physics
baseline and learned detectors evaluated. Everything below is synthetic; the
real-data anchor is the open question.
**License:** MIT

## The claim being tested

A detector trained on voids and utilities should fail on fossil-like targets
because the dominant reflection polarity inverts. High-permittivity bone reflects
negative at the top interface where a low-permittivity void reflects positive, and
the bone contrast is roughly 1.7x weaker in dry sand.

That mechanism is not speculation. It came out of a 1D forward model before any
deep learning was involved (`docs/notes/08_forward_model_1d_validation.md`). The
hypothesis promotes it into a claim about machine learning transfer, which is the
part that is actually novel.

Four sub-hypotheses, each with its falsification condition written down before
anything was run, are in `docs/notes/09_research_hypothesis.md`. One of them treats
a null result on real data as a publishable finding.

I retired my original "first ML approach to fossil GPR prospecting" framing after a
literature pass. Void and utility detection is already a published open benchmark,
and "first to do X" dies the moment a reviewer finds one paper I missed. The
defensible contribution is characterizing the domain shift.

## Results so far

### Physics baseline, zero learned parameters

A Ricker-pulse matched filter that reads reflection polarity from normalized
cross-correlation. Built deliberately before any deep learning: if a CNN cannot
beat a plain matched filter, the extra complexity has not earned its place.

| target | eps | detected | polarity | expected |
|---|---|---|---|---|
| void | 1.00 | yes | +1 (void-like) | +1 |
| bone | 9.00 | yes | -1 (bone-like) | -1 |
| anti-bone (control) | 1.78 | yes | +1 (void-like) | +1 |
| null (no contrast) | 4.00 | no | 0 | 0 |

The anti-bone row is the one that matters. It carries bone's contrast magnitude
with a void's sign, and the filter groups it with the void. The baseline keys on
polarity, not amplitude, which is the causal control for the hypothesis.

Noise sweep: 300 trials per level. Detection holds near 1.0 until noise reaches
roughly 0.65 to 0.7 of signal peak, while polarity accuracy stays flat well past
that. Details in `docs/notes/11_polarity_baseline.md`.

### Learned detection on synthetic B-scans

| model | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| YOLOv8n (2-class) | 0.581 | 0.296 |
| YOLOv8n (binary) | 0.600 | 0.312 |

## Controlled synthetic benchmark

`configs/synthetic_controlled.yaml` drives a gprMax generator that holds scene
geometry, depth, host medium, and target geometry constant and varies only the
target dielectric. For every (host, depth, frequency, position) cell it emits the
same scene four times, once per target type, under a shared `scene_id` so the
conditions pair exactly. Four conditions: void, bone, magnitude-matched anti-bone,
and a null with no contrast.

The point is to isolate the variable. If the void-trained detector drops on bone,
the design tells you whether that is polarity, contrast magnitude, or just
"synthetic looks different."

## What is not yet established

- **Everything here is synthetic.** No result in this repo has touched real fossil
  GPR. The anchor experiment is a small set of real B-scans digitized from the
  literature, and it is not done.
- **The early fossil-image classification numbers in `results/tables/baseline_results.csv`
  have not had validity work.** ResNet18 reports 99.5% test accuracy with a
  train-val gap of exactly 0.0, which is the shape of a leak rather than a result.
  Treat that table as unaudited until it is re-run. It is unrelated to the GPR
  hypothesis above.
- No claim is made about beating published void/utility detectors on their own
  domain. That problem is solved and public.

## Layout

```
src/data/          forward model, gprMax generation, preprocessing, splits
src/models/        polarity matched filter, detectors, classifiers, evaluation
src/visualization/ figure generation
configs/           experiment configs, including synthetic_controlled.yaml
docs/notes/        numbered research log, 00 through 12
docs/experiments/  experiment_01 baseline, 02 detection, 03 transfer
results/           tables, figures, detection runs
tests/             physics and generator tests
notebooks/         exploration
```

Start with `docs/notes/09_research_hypothesis.md`. It is the spine of the project
and everything else serves it.

## Reproducing

```bash
conda env create -f environment.lock.yml    # or: pip install -r requirements.txt
pytest tests/                               # physics and generator tests
python -m src.data.run_forward_model_validation
python -m src.models.run_polarity_baseline
```

gprMax scene generation is the expensive step and is driven by
`src/data/generate_gprmax_models.py` with `configs/synthetic_controlled.yaml`.

## References

The infrastructure dataset is the published Morocco utilities/voids/intact set
(2,239 images, Bestagini / El Mahdaoui et al., *Data in Brief*). Bone permittivity
range follows Peredo et al. Full literature matrix in
`docs/reference/literature_matrix.md`.
