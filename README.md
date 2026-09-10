# Syrian Hamster Behaviour Classification Artefact

## Project title

**Markerless Pose Estimation and Behaviour Classification of a Syrian Hamster Using Consumer Pet-Camera Footage**

This package contains the runnable artefact for the MSc thesis project.

The **primary artefact** is the Core-8 pose-based behaviour-classification pipeline. Core-10 and the non-pose system are included as optional comparison systems.

---

## Folder structure

```text
Thesis_Artefact_Package/
├── README.md
├── requirements.txt
│
├── models/
│   ├── core8/
│   │   ├── core8_pose_best.pt
│   │   ├── xgboost_core8_model.joblib
│   │   ├── median_imputer.joblib
│   │   ├── label_encoder.joblib
│   │   └── feature_columns.joblib
│   │
│   ├── core10/
│   │   ├── core10_pose_best.pt
│   │   ├── xgboost_core10_model.joblib
│   │   ├── median_imputer.joblib
│   │   ├── label_encoder.joblib
│   │   └── feature_columns.joblib
│   │
│   └── nonpose/
│       ├── core8_presence_gate_best.pt
│       ├── xgboost_nonpose_model.joblib
│       ├── median_imputer.joblib
│       ├── label_encoder.joblib
│       └── feature_columns.joblib
│
├── scripts/
│   ├── predict_behaviour_video_core8.py
│   ├── predict_behaviour_video_core10.py
│   └── predict_behaviour_video_nonpose_with_presence.py
│
├── sample_videos/
│   ├── sample_01.mp4
│   ├── sample_02.mp4
│   ├── sample_03.mp4
│   └── sample_04.mp4
│
└── outputs/
    ├── core8/
    │   ├── generated_outputs/
    │   └── example_outputs/
    │       ├── example_01_core8.mp4
    │       ├── example_02_core8.mp4
    │       ├── example_03_core8.mp4
    │       └── example_04_core8.mp4
    │
    ├── core10/
    │   ├── generated_outputs/
    │   └── example_outputs/
    │       ├── example_01_core10.mp4
    │       ├── example_02_core10.mp4
    │       ├── example_03_core10.mp4
    │       └── example_04_core10.mp4
    │
    ├── nonpose/
    │   ├── generated_outputs/
    │   └── example_outputs/
    │       ├── example_01_nonpose.mp4
    │       ├── example_02_nonpose.mp4
    │       ├── example_03_nonpose.mp4
    │       └── example_04_nonpose.mp4
    │
    └── comparison_examples/
        ├── example_01_three_way.mp4
        ├── example_02_three_way.mp4
        ├── example_03_three_way.mp4
        └── example_04_three_way.mp4
```

---

## Primary system: Core-8

The main runnable pipeline is:

**Input video → hamster detection + Core-8 pose estimation → cleaned frame-level landmarks → pose-derived temporal features → frozen XGBoost behaviour classifier → annotated video + CSV outputs**

Behaviour classes:

- Climbing
- Foraging
- Grooming
- Rearing
- Wheel running

The demonstration may also display:

- `No hamster detected`
- `Unclassified`

The hamster-presence and model-score thresholds used in the demonstration are post-processing choices and were **not** used in the reported held-out Test evaluation.

### Core-8 model configuration

The primary pipeline combines a YOLO pose-estimation model with a pose-derived XGBoost behaviour classifier.

**Pose model**

| Parameter | Configuration |
|---|---|
| Base model | `yolo26n-pose.pt` |
| Landmarks | 8 |
| Training frames | 143 Train / 28 Validation / 29 Test |
| Epochs | 80 |
| Early-stopping patience | 20 |
| Image size | 640 |
| Initial learning rate | 0.001 |
| Mosaic augmentation | 0.5 |
| Mixup | 0.0 |
| Batch size | Automatic |
| Random seed | 42 |
| Deterministic training | Enabled |
| Training device | Apple MPS |

The eight landmarks are: **nose, left eye, right eye, left ear, right ear, neck/shoulder, spine midpoint, and tail base**.

During inference, the Core-8 YOLO Pose model detects the hamster in each frame and predicts the eight body landmarks. The resulting landmark coordinates and confidence scores form the frame-level pose representation used by the next stage of the pipeline. Low-confidence keypoints are treated as missing before temporal feature extraction, with short gaps interpolated where applicable.

**Temporal feature extraction**

Pose predictions are processed in overlapping **2-second windows with a 1-second stride**. Keypoints below **0.30 confidence** are treated as missing, short gaps of up to **5 frames** are linearly interpolated, and windows require a minimum **70% pose-detection rate** to be included in the model-ready dataset.

Derived features describe body geometry, orientation, displacement, speed, directional movement and relative head position.

**Behaviour classifier**

| Parameter | Configuration |
|---|---|
| Model | XGBoost multi-class classifier |
| Trees | 500 |
| Maximum depth | 5 |
| Learning rate | 0.05 |
| Row subsampling | 0.85 |
| Feature subsampling | 0.85 |
| Objective | `multi:softprob` |
| Evaluation metric | `mlogloss` |
| Tree method | `hist` |
| Class balancing | Balanced sample weights |
| Missing values | Training-set median imputation |
| Random seed | 42 |

The frozen classifier predicts five behaviour classes: **Climbing, Foraging, Grooming, Rearing, and Wheel running**.

---

## Model folders

### `models/core8/`

Contains everything required to run the primary Core-8 artefact.

Expected files:

```text
core8_pose_best.pt
xgboost_core8_model.joblib
median_imputer.joblib
label_encoder.joblib
feature_columns.joblib
```

### `models/core10/`

Contains everything required to run the optional exploratory Core-10 artefact.

Expected files:

```text
core10_pose_best.pt
xgboost_core10_model.joblib
median_imputer.joblib
label_encoder.joblib
feature_columns.joblib
```

### `models/nonpose/`

Contains everything required to run the non-pose motion classifier with the practical hamster-presence gate.

Expected files:

```text
core8_presence_gate_best.pt
xgboost_nonpose_model.joblib
median_imputer.joblib
label_encoder.joblib
feature_columns.joblib
```

The copied `core8_presence_gate_best.pt` model is used **only to determine whether the hamster is reliably present**. The non-pose XGBoost behaviour classifier itself still receives only motion-derived features.

---

## Sample videos

Four short, privacy-safe test clips are included in:

```text
sample_videos/
```

| File | Duration | Visible behaviours |
|---|---:|---|
| `sample_01.mp4` | 1 min 02 s | Climbing, Rearing |
| `sample_02.mp4` | 24 s | Rearing, Climbing |
| `sample_03.mp4` | 19 s | Grooming, Rearing |
| `sample_04.mp4` | 24 s | Climbing, Grooming |

The scripts default to:

```text
sample_videos/sample_01.mp4
```

Any of the supplied clips can be run by changing the `--video` argument. The clips are intentionally short so the artefact can be tested quickly while still producing multiple overlapping 2-second behaviour windows.

---

## Installation

Python **3.10** is recommended.

Use of a dedicated virtual environment is **recommended** so that the artefact dependencies do not interfere with packages already installed on the assessor's system.

From the package root, a standard Python virtual environment can be created and activated with:

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **macOS note:** XGBoost requires the OpenMP runtime. If the artefact fails with an error mentioning `libomp.dylib`, install it with Homebrew:
>
> ```bash
> brew install libomp
> ```
>
> Then rerun the artefact command.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation because script execution is disabled, the environment can still be activated after allowing scripts for the current PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If a suitable Python environment is already active, the dependencies can instead be installed directly with:

```bash
python -m pip install -r requirements.txt
```

The current `requirements.txt` contains:

```text
ultralytics
opencv-python
numpy
pandas
scikit-learn==1.7.2
xgboost
joblib
torch
```

The scripts automatically select an available inference device in the following order:

**CUDA → Apple MPS → CPU**

A device can also be specified manually with `--device`.

---

## Running the primary Core-8 artefact

From the package root:

```bash
python scripts/predict_behaviour_video_core8.py --video sample_videos/sample_01.mp4
```

If `--video` is omitted, the script automatically attempts to use:

```text
sample_videos/sample_01.mp4
```

New outputs are written to:

```text
outputs/core8/generated_outputs/
```

Typical Core-8 outputs are:

```text
<video_name>_annotated.mp4
<video_name>_behaviour_predictions.csv
<video_name>_pose.csv
```

---

## Running Core-10

Core-10 is included as an exploratory comparison model.

```bash
python scripts/predict_behaviour_video_core10.py --video sample_videos/sample_01.mp4
```

New outputs are written to:

```text
outputs/core10/generated_outputs/
```

Typical Core-10 outputs are:

```text
<video_name>_annotated.mp4
<video_name>_behaviour_predictions.csv
<video_name>_pose.csv
```

Core-10 extends the Core-8 skeleton with left and right forepaw landmarks. It improved some Validation results but did not improve overall held-out Test generalisation, so Core-8 remains the primary artefact.

---

## Running the non-pose comparison system

The non-pose system classifies behaviour from motion-derived features such as frame difference and optical flow.

For the continuous-video demonstration, Core-8 YOLO Pose is used only as a practical hamster-presence gate.

```bash
python scripts/predict_behaviour_video_nonpose_with_presence.py --video sample_videos/sample_01.mp4
```

New outputs are written to:

```text
outputs/nonpose/generated_outputs/
```

Typical outputs are:

```text
<video_name>_nonpose_presence_gate_annotated.mp4
<video_name>_nonpose_predictions_presence_gate.csv
<video_name>_presence_gate.csv
```

The presence gate prevents obvious environmental motion, such as a freely moving exercise wheel while the hamster is absent, from being displayed as an accepted animal-behaviour prediction.

The non-pose XGBoost classifier itself remains fully non-pose.

---

## Output folders

Each pipeline has two output locations.

### `example_outputs/`

These folders contain pre-rendered example outputs supplied with the artefact:

```text
outputs/core8/example_outputs/
outputs/core10/example_outputs/
outputs/nonpose/example_outputs/
```

Matching example numbers across the three pipeline folders refer to the **same source video**. For example:

```text
example_01_core8.mp4
example_01_core10.mp4
example_01_nonpose.mp4
```

are outputs generated from the same underlying source clip. This allows direct qualitative comparison between the three systems.

The supplied example videos are:

| Example | Visible behaviours / purpose |
|---|---|
| `example_01` | Grooming, Rearing and Climbing |
| `example_02` | Rearing and Climbing, including more errors and unclassified moments |
| `example_03` | Climbing and Rearing |
| `example_04` | Wheel running |

The examples are intended to show both successful predictions and practical limitations in continuous-video deployment. They are not a substitute for the held-out quantitative Test evaluation reported in the dissertation.

### `comparison_examples/`

The `comparison_examples/` folder contains synchronised side-by-side videos showing all three pipeline outputs for the same source clip.

Each comparison video presents:

```text
Core-8 Pose | Core-10 Pose | Non-pose + Presence Gate
```

The aligned panels allow direct qualitative inspection of agreement, uncertainty and failure cases without requiring the three individual output videos to be opened separately.

```text
outputs/comparison_examples/
├── example_01_three_way.mp4
├── example_02_three_way.mp4
├── example_03_three_way.mp4
└── example_04_three_way.mp4
```

These videos are provided for qualitative comparison only and do not replace the held-out quantitative Test evaluation reported in the dissertation.

### `generated_outputs/`

These folders are used automatically when someone runs the artefact:

```text
outputs/core8/generated_outputs/
outputs/core10/generated_outputs/
outputs/nonpose/generated_outputs/
```

The scripts create the generated-output folders automatically if required.

---

## Interpretation

This artefact is a research prototype demonstrating the feasibility of pose estimation and behaviour classification from non-standardised consumer pet-camera footage.

It is not intended as:

- a veterinary diagnostic tool;
- an animal welfare decision system;
- a population-level model for Syrian hamsters;
- a system for inferring emotion, intention, motivation or health state.

The study is based on one animal, one domestic environment and one consumer camera.

---

## Privacy

Only privacy-safe sample and example videos are included in the submitted artefact.

Raw source recordings are not required for the runnable package.

Audio was not used in the modelling pipeline.

---

## Reproducibility

The artefact is intended primarily for **inference rather than retraining**.

Training procedures, dataset construction, episode-level Train/Validation/Test splitting, feature extraction and held-out evaluation are documented in the accompanying dissertation.

Core-8 is the primary submitted model. Core-10 and the non-pose system are included as comparison systems.

The demonstration rejection thresholds are intended to make continuous-video output more interpretable. They should not be treated as part of the reported scientific Test-set evaluation.
