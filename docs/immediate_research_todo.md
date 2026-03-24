# Immediate Research TODO: paleo-gpr-ml

## Where this file should live in the repo

Save this file as:

`docs/immediate_research_todo.md`

This is the right place because it is not code, not a paper note, and not a long-term roadmap. It is your **active execution checklist** for the next stage of the project.

---

## Purpose of this document

This file is meant to answer one question clearly:

**What do I need to do right now to start this project correctly and learn as much as possible?**

The goal is not just to rush into modeling. The goal is to:
- understand the problem deeply
- understand the data deeply
- understand how the literature connects to the data
- build the project in a way that is research-grade and future-proof
- make it easy for me or an agent/LLM to continue the work later without confusion

---

## High-level objective for this phase

My current phase is:

**Move from setup into real research execution by understanding the literature, understanding the first GPR dataset, documenting what I observe, and preparing for the first baseline experiment.**

This means I am **not** trying to do everything at once.

I am trying to do four things well:
1. Read and understand the most important papers
2. Explore the GPR dataset carefully
3. Document observations and questions
4. Prepare a clean foundation for synthetic data and baseline modeling

---

## What is already done

The following pieces are already in place:
- repository created
- environment and core libraries installed
- roadmap drafted
- project scope drafted
- literature matrix started
- papers downloaded
- first GPR dataset downloaded and extracted

That means I now need to begin the actual research workflow.

---

## Main workstreams from this point forward

There are four parallel workstreams:

### Workstream A: Literature understanding
Read the papers, understand the methods, and connect them to my project.

### Workstream B: Dataset understanding
Inspect the GPR dataset carefully and learn what the signals, labels, and structures look like.

### Workstream C: Research documentation
Write down what I am learning so that the project becomes reproducible and organized.

### Workstream D: Experiment preparation
Prepare the dataset and notebook structure so that a first baseline model can be built later.

These should happen together, not one giant block at a time.

---

# PHASE 1: Organize the immediate working area

## Task 1.1: Verify the dataset location and structure

### Goal
Make sure the extracted GPR dataset is in the correct place and easy to inspect.

### Action
Put the extracted dataset inside a clean folder such as:

`data/raw/gpr/<dataset_name>/`

### Example
`data/raw/gpr/mendeley_gpr_dataset/`

### What to check
- Are the files images, arrays, text files, or something else?
- Are there labels?
- Is there a metadata file?
- Is there a train/test split already?
- Are there multiple folders by class?
- Are filenames clean and understandable?

### Deliverable
A short note in `docs/data_understanding.md` describing the directory structure.

---

## Task 1.2: Create or verify these working files

### Files to create or confirm
- `docs/immediate_research_todo.md`
- `docs/data_understanding.md`
- `docs/research_log.md`
- `notebooks/01_gpr_data_exploration.ipynb`

### Why
These are your active workspace files:
- `immediate_research_todo.md` = execution plan
- `data_understanding.md` = what the dataset is
- `research_log.md` = daily notes, decisions, confusion, questions
- `01_gpr_data_exploration.ipynb` = actual hands-on analysis

### Deliverable
These files exist in the repo and are ready to use.

---

# PHASE 2: Read the papers in a structured way

## Task 2.1: Do not read passively

### Goal
Read papers like a researcher, not like a student cramming for a quiz.

For each paper, answer:
- What problem is this paper solving?
- What data does it use?
- What exact input representation is used?
- What model or algorithm is used?
- What metrics are used?
- What is the actual contribution?
- What are the weaknesses?
- How does this help my project?
- What part of my problem does this paper not solve?

### Deliverable
For each paper, create one note in:

`papers/reading_notes/<category>/`

---

## Task 2.2: Read the first paper batch in this order

### First reading order
1. AI in Paleontology review
2. Detection of Vertebrate Skeletons by GPR
3. Dinosaur-bone GPR paper in Sicily
4. GPR AI review
5. YOLO/GPR detection paper
6. Fossil CT segmentation paper

### Why this order
This order goes from:
- field overview
- closest problem match
- practical geophysics context
- ML for GPR methods
- actual detection model example
- downstream fossil analysis

That gives me both the big picture and the first technical direction.

### Deliverable
A minimum of 5 completed reading notes.

---

## Task 2.3: Update the literature matrix after every paper

### Goal
Keep the matrix current so I do not lose track of what matters.

For every paper, add:
- title
- year
- link
- file path in repo
- task
- input data
- model/method
- main contribution
- weakness
- relevance to my project

### Deliverable
`docs/literature_matrix.md` stays updated as the central source of truth.

---

## Task 2.4: Create a “Questions Raised by the Literature” section

### Goal
Turn confusion into research direction.

Inside `docs/research_log.md`, make a section called:

`Questions Raised by the Literature`

Whenever a paper makes me wonder something, write it down.

### Example questions
- What radar signatures would fossilized bone produce compared to pipes or voids?
- Will synthetic fossil-like geometry actually transfer to real geology?
- Is object detection better than segmentation for subtle radar anomalies?
- How much preprocessing is standard in GPR ML papers?
- Do most papers use raw radargrams or processed radargrams?

### Deliverable
A growing question bank that guides the project instead of leaving me lost.

---

# PHASE 3: Start the first notebook properly

## Task 3.1: Build the notebook skeleton before doing heavy analysis

### File
`notebooks/01_gpr_data_exploration.ipynb`

### Recommended sections
1. Project context and purpose
2. Imports
3. Locate dataset files
4. Inspect directory structure
5. Load sample files
6. Visualize random samples
7. Inspect labels or metadata
8. Summarize observations
9. Open questions
10. Next preprocessing ideas

### Why
This keeps the notebook readable and makes future iteration much easier.

### Deliverable
A notebook with clean sections and markdown cells before heavy code begins.

---

## Task 3.2: Inspect file formats

### Goal
Understand exactly what data format I am working with.

### Questions to answer
- Are the scans stored as `.png`, `.jpg`, `.npy`, `.mat`, `.csv`, or something else?
- Are labels stored separately?
- Is metadata encoded in filenames?
- Are classes obvious from folder names?
- Is there a README or description file included?

### Deliverable
A section in the notebook and in `docs/data_understanding.md` describing file formats.

---

## Task 3.3: Visualize at least 20 samples

### Goal
Develop visual intuition for radargrams.

### What to do
- randomly sample at least 20 examples
- display them in the notebook
- note similarities and differences

### What to pay attention to
- bright arcs or hyperbolic patterns
- background noise
- horizontal banding
- depth variation
- faint vs strong targets
- whether anomalies are centered or off-center
- whether samples look standardized

### Deliverable
Saved visual examples in:
`results/figures/gpr_exploration/`

Suggested filenames:
- `sample_grid_01.png`
- `sample_grid_02.png`
- `strong_anomaly_examples.png`
- `weak_anomaly_examples.png`

---

## Task 3.4: Learn what a hyperbola looks like in GPR

### Goal
This is one of the most important learning steps.

Buried objects in radargrams often appear as hyperbolic signatures rather than literal object outlines.

### What to investigate
- Which samples clearly show hyperbolic structure?
- Which anomalies are not clean hyperbolas?
- Are there examples where clutter could be mistaken for a target?
- Are some classes easier to recognize than others?

### Deliverable
A short written section in `docs/data_understanding.md` called:

`Visual patterns and likely anomaly signatures`

---

## Task 3.5: Inspect label structure

### Goal
Determine what learning task is actually possible with the dataset.

### Questions
- Are there bounding boxes?
- Are there segmentation masks?
- Are labels only image-level classes?
- Are there multiple classes or binary labels?
- Is there a train/val/test split?

### Why this matters
This directly determines what model family I should use first:
- image-level classes → classifier
- bounding boxes → detector
- masks → segmentation

### Deliverable
A section in `docs/data_understanding.md` titled:

`Label structure and implications for modeling`

---

## Task 3.6: Check class balance and sample counts

### Goal
Understand the learning difficulty.

### What to compute
- total number of samples
- number of classes
- samples per class
- imbalance severity

### Questions
- Is the dataset balanced?
- Are some anomaly types rare?
- Will I need weighted loss or sampling later?

### Deliverable
A summary table or chart saved in:
`results/figures/gpr_exploration/class_distribution.png`

---

## Task 3.7: Identify data quality issues

### Goal
Catch problems early.

### Look for
- corrupt files
- duplicates
- inconsistent sizes
- unreadable images
- mislabeled samples
- very low-contrast examples
- padding/cropping inconsistencies

### Deliverable
A section in `docs/data_understanding.md` called:

`Data quality issues and risks`

---

# PHASE 4: Turn observations into structured understanding

## Task 4.1: Write `docs/data_understanding.md`

### Goal
Convert notebook observations into clear documentation.

### Suggested structure
1. Dataset source
2. Dataset path in repo
3. File structure
4. File formats
5. Number of samples
6. Label structure
7. Example classes
8. Visual patterns
9. Data quality issues
10. Implications for preprocessing
11. Implications for modeling
12. Open questions

### Deliverable
A complete first version of `docs/data_understanding.md`

---

## Task 4.2: Start `docs/research_log.md`

### Goal
Keep a chronological record of decisions.

### What to write after each session
- What I did
- What I learned
- What confused me
- What decisions I made
- What to do next

### Why this is important
This becomes useful for:
- paper writing
- debugging
- experiment tracking
- handing off to an LLM agent later

### Deliverable
At least the first 2 to 3 dated entries.

---

# PHASE 5: Connect the papers to the data

## Task 5.1: Compare what papers assume vs what my dataset actually contains

### Goal
Avoid building a plan that does not match reality.

### Questions
- Do the papers use data similar to mine?
- Do they use stronger preprocessing than I expected?
- Are their labels richer than mine?
- Are their tasks easier or harder than what I have?

### Deliverable
A section in `docs/research_log.md` called:

`Literature-to-data alignment`

---

## Task 5.2: Create a “methods to borrow” list

### Goal
Extract reusable research ideas.

### Create a section in `docs/research_log.md`:
`Methods to borrow from the literature`

### Possible items
- preprocessing techniques
- augmentation ideas
- detection framing
- segmentation framing
- evaluation metrics
- visualization styles
- synthetic data strategies

### Deliverable
A growing list of concrete implementation ideas.

---

# PHASE 6: Prepare for synthetic fossil-like data

## Task 6.1: Write down what a fossil-like target should mean in V1

### Goal
Avoid vague project language.

Inside `docs/synthetic_data_plan.md`, define:
- what counts as bone-like geometry
- what counts as rib-like geometry
- what counts as fossil fragments
- what confounders matter
- what environmental variation matters

### Suggested first synthetic target classes
- elongated bone-like object
- curved rib-like object
- fragmented fossil cluster
- buried pipe
- rock clutter
- void/cavity

### Deliverable
A first draft of `docs/synthetic_data_plan.md`

---

## Task 6.2: Create a list of simulation variables

### Goal
Prepare for gprMax or any synthetic generation tool later.

### Variables to define
- object depth
- object size
- object orientation
- material properties
- antenna frequency
- noise level
- soil background type
- number of objects in scene

### Deliverable
Simulation variable table in `docs/synthetic_data_plan.md`

---

# PHASE 7: Prepare the first baseline experiment without training yet

## Task 7.1: Decide what the first ML task really is

### Goal
Make the first experiment small and executable.

Possible first tasks:
- image classification of anomaly vs non-anomaly
- detection of buried target locations
- segmentation of anomaly regions

### Best likely first choice
This depends on the labels, but after inspection:
- if image-level labels only → start with classification
- if boxes exist → start with object detection
- if masks exist → segmentation is possible

### Deliverable
A short decision note in `docs/experiment_01_baseline.md`

---

## Task 7.2: Write the baseline experiment spec

### File
`docs/experiment_01_baseline.md`

### Include
- objective
- dataset used
- task type
- preprocessing
- train/val/test split
- baseline model
- metrics
- risks
- success criteria

### Deliverable
A complete first experiment plan, even before code training starts.

---

# PHASE 8: Maximize learning, not just progress

## Task 8.1: Keep a “What I learned” section after each session

### Goal
Turn work into retained understanding.

At the end of every work session, write:
- 3 things I learned
- 2 things I still do not understand
- 1 thing I want to investigate next

### Where
`docs/research_log.md`

### Deliverable
Consistent learning summaries.

---

## Task 8.2: Build intuition intentionally

### What I should understand by the end of this phase
- what GPR radargrams are
- what anomalies look like
- why buried targets form hyperbolas
- what labels I actually have
- what the first ML task should be
- how current papers relate to the actual data I have

### If I cannot explain these in plain language yet
I should not rush into modeling.

---

# Immediate action checklist

## Do these first, in order

### Right now
- [x] Confirm dataset path under `data/raw/GPR_data/`
- [x] Create `docs/data_understanding.md`
- [x] Create `docs/research_log.md`
- [x] Create `notebooks/01_gpr_data_exploration.ipynb`

### First notebook session
- [x] Inspect directory structure
- [x] Identify file types
- [x] Load sample files
- [x] Visualize at least 20 samples (25+ figures generated)
- [x] Check labels (YOLO + VOC annotations parsed and validated)
- [x] Count classes/samples (2,524 total, 285 unique originals)
- [x] Save example figures (16 figures in `results/figures/`)

### First documentation session
- [x] Write initial `data_understanding.md`
- [x] Add first entry to `research_log.md` (3 dated entries now)
- [x] Update literature matrix with downloaded papers

### First reading session
- [x] Read AI in Paleontology review
- [x] Read vertebrate skeleton GPR paper
- [x] Read all 6 papers and create reading notes for each (`papers/reading_notes/`)
- [x] Write how each connects to dataset and project (literature synthesis in `docs/notes/03_literature_synthesis.md`)

### After that
- [x] Decide the first learning task type (3-class classification, then detection)
- [x] Draft `docs/experiment_01_baseline.md` (complete, experiment run, 99.5% accuracy)
- [x] Draft `docs/synthetic_data_plan.md` (complete with gprMax parameters from literature)

### Phase 2 work (in progress)
- [x] Run baseline classification experiment (`notebooks/02_baseline_classification.ipynb`)
- [x] Write experiment notes (`docs/notes/02_baseline_experiment_notes.md`)
- [ ] Run Grad-CAM analysis (`notebooks/03_gradcam_analysis.ipynb` - created, needs execution)
- [ ] Write Grad-CAM notes (`docs/notes/04_gradcam_notes.md` - template created)
- [x] Write detection experiment spec (`docs/experiment_02_detection.md`)
- [ ] Prepare detection data (`src/data/prepare_detection_data.py` - created, needs execution)
- [ ] Run object detection experiment
- [ ] Begin synthetic data generation with gprMax

---

# Definition of “done” for this stage

This stage is complete when I have:
- [x] a clear understanding of the first GPR dataset
- [x] a completed exploration notebook
- [x] a written data understanding document
- [x] at least 5 paper notes (6 completed)
- [x] an updated literature matrix (with gap analysis)
- [x] a research log with decisions and questions (3 entries + lit questions)
- [x] a draft baseline experiment plan (and the experiment is done)
- [x] a first synthetic data plan draft

**STATUS: This stage is COMPLETE.** Moving into Phase 2:
- Grad-CAM analysis (notebook created, ready to run)
- Object detection experiment (spec written, data prep script created)
- Synthetic fossil-target generation (plan written, gprMax parameters sourced from literature)

---

# Final reminder to myself

The goal right now is not to impress myself with models.

The goal is to become deeply fluent in:
- the problem
- the data
- the literature
- the research direction

If I do this phase carefully, the modeling phase will be dramatically better and far less confusing.
