from pathlib import Path
import argparse

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO


# ============================================================
# COMMAND-LINE INPUT + PORTABLE ARTEFACT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent

DEFAULT_VIDEO_PATH = (
    PACKAGE_ROOT
    / "sample_videos"
    / "sample_01.mp4"
)

parser = argparse.ArgumentParser(
    description=(
        "Run the non-pose hamster behaviour classifier "
        "with a Core-8 YOLO hamster-presence gate."
    )
)

parser.add_argument(
    "--video",
    type=Path,
    default=DEFAULT_VIDEO_PATH,
    help=(
        "Path to the input video. "
        "Defaults to sample_videos/sample_01.mp4."
    ),
)

parser.add_argument(
    "--device",
    default="auto",
    help=(
        "Inference device for the YOLO presence gate: "
        "auto, cpu, mps, cuda, cuda:0, etc. Default: auto."
    ),
)

args = parser.parse_args()

VIDEO_PATH = args.video.expanduser()

if not VIDEO_PATH.is_absolute():
    VIDEO_PATH = (Path.cwd() / VIDEO_PATH).resolve()


MODEL_DIR = (
    PACKAGE_ROOT
    / "models"
    / "nonpose"
)

OUTPUT_DIR = (
    PACKAGE_ROOT
    / "outputs"
    / "nonpose"
    / "generated_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


POSE_GATE_MODEL = (
    MODEL_DIR
    / "core8_presence_gate_best.pt"
)

MODEL_FILE = (
    MODEL_DIR
    / "xgboost_nonpose_model.joblib"
)

IMPUTER_FILE = (
    MODEL_DIR
    / "median_imputer.joblib"
)

LABEL_ENCODER_FILE = (
    MODEL_DIR
    / "label_encoder.joblib"
)

FEATURE_COLUMNS_FILE = (
    MODEL_DIR
    / "feature_columns.joblib"
)


def resolve_device(requested_device):

    requested_device = str(
        requested_device
    ).strip().lower()

    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda:0"

    if (
        hasattr(torch.backends, "mps")
        and
        torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = resolve_device(args.device)


# ------------------------------------------------------------
# Non-pose motion features
# ------------------------------------------------------------

WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 1.0

PROCESS_WIDTH = 320
MOTION_THRESHOLD = 20


# ------------------------------------------------------------
# Hamster-presence gate
# ------------------------------------------------------------

YOLO_CONFIDENCE = 0.25

MIN_HAMSTER_BOX_CONFIDENCE = 0.50

KEYPOINT_CONF_THRESHOLD = 0.30

MIN_STRONG_KEYPOINTS = 4

CORE_KEYPOINT_INDICES = [
    5,  # neck_shoulder
    6,  # spine_mid
]

MIN_HAMSTER_PRESENCE_RATE = 0.40


# ------------------------------------------------------------
# Demo-only behaviour rejection
# ------------------------------------------------------------

MIN_BEHAVIOUR_SCORE = 0.70


# ============================================================
# OUTPUT FILES
# ============================================================

stem = VIDEO_PATH.stem

PREDICTION_CSV = (
    OUTPUT_DIR
    / f"{stem}_nonpose_predictions_presence_gate.csv"
)

PRESENCE_CSV = (
    OUTPUT_DIR
    / f"{stem}_presence_gate.csv"
)

ANNOTATED_VIDEO = (
    OUTPUT_DIR
    / f"{stem}_nonpose_presence_gate_annotated.mp4"
)


# ============================================================
# HELPERS
# ============================================================

def safe_mean(values):

    arr = np.asarray(
        values,
        dtype=float
    )

    if np.all(np.isnan(arr)):
        return np.nan

    return float(
        np.nanmean(arr)
    )


def safe_std(values):

    arr = np.asarray(
        values,
        dtype=float
    )

    if np.sum(
        ~np.isnan(arr)
    ) < 2:
        return np.nan

    return float(
        np.nanstd(arr)
    )


def safe_max(values):

    arr = np.asarray(
        values,
        dtype=float
    )

    if np.all(np.isnan(arr)):
        return np.nan

    return float(
        np.nanmax(arr)
    )


def safe_min(values):

    arr = np.asarray(
        values,
        dtype=float
    )

    if np.all(np.isnan(arr)):
        return np.nan

    return float(
        np.nanmin(arr)
    )


def resize_gray(frame):

    h, w = frame.shape[:2]

    scale = (
        PROCESS_WIDTH
        /
        w
    )

    new_height = int(
        round(
            h
            *
            scale
        )
    )

    resized = cv2.resize(
        frame,
        (
            PROCESS_WIDTH,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    return gray


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    VIDEO_PATH,
    POSE_GATE_MODEL,
    MODEL_FILE,
    IMPUTER_FILE,
    LABEL_ENCODER_FILE,
    FEATURE_COLUMNS_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 72)
print("NON-POSE VIDEO DEMO WITH HAMSTER-PRESENCE GATE")
print("=" * 72)
print()


print(
    f"Package root: {PACKAGE_ROOT}"
)

print(
    f"Input video:  {VIDEO_PATH}"
)

print(
    f"Device:       {DEVICE}"
)

print()

print("Loading hamster-presence model...")

presence_model = YOLO(
    POSE_GATE_MODEL
)


print("Loading non-pose behaviour classifier...")

classifier = joblib.load(
    MODEL_FILE
)

imputer = joblib.load(
    IMPUTER_FILE
)

label_encoder = joblib.load(
    LABEL_ENCODER_FILE
)

feature_columns = joblib.load(
    FEATURE_COLUMNS_FILE
)


print(
    f"Behaviour features: "
    f"{len(feature_columns)}"
)

print(
    f"Classes: "
    f"{list(label_encoder.classes_)}"
)

print()


# ============================================================
# VIDEO INFO
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)


if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

frame_count = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


if fps <= 0:
    fps = 15.0


duration = (
    frame_count
    /
    fps
)


print(
    f"Video:      {VIDEO_PATH.name}"
)

print(
    f"Resolution: {width} × {height}"
)

print(
    f"FPS:        {fps:.3f}"
)

print(
    f"Frames:     {frame_count}"
)

print(
    f"Duration:   {duration:.2f} s"
)

print()


# ============================================================
# PASS 1
# NON-POSE MOTION FEATURES + HAMSTER PRESENCE
# ============================================================

print("=" * 72)
print("PASS 1: MOTION FEATURES + HAMSTER PRESENCE")
print("=" * 72)
print()


frame_rows = []
presence_rows = []

previous_gray = None

previous_motion_cx = np.nan
previous_motion_cy = np.nan

frame_index = 0


while True:

    ok, frame = cap.read()

    if not ok:
        break


    timestamp_seconds = (
        frame_index
        /
        fps
    )


    # ========================================================
    # HAMSTER PRESENCE GATE
    # ========================================================

    result = presence_model.predict(
        frame,
        conf=YOLO_CONFIDENCE,
        device=DEVICE,
        verbose=False,
    )[0]


    valid_hamster = False

    box_confidence = np.nan

    strong_keypoints = 0

    core_keypoints_valid = False


    if (
        result.boxes is not None
        and
        len(result.boxes) > 0
        and
        result.keypoints is not None
    ):

        box_scores = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )


        best_index = int(
            np.argmax(
                box_scores
            )
        )


        box_confidence = float(
            box_scores[
                best_index
            ]
        )


        kp_conf = (
            result.keypoints.conf[
                best_index
            ]
            .detach()
            .cpu()
            .numpy()
        )


        strong_keypoints = int(
            np.sum(
                kp_conf
                >=
                KEYPOINT_CONF_THRESHOLD
            )
        )


        if (
            len(kp_conf)
            >
            max(
                CORE_KEYPOINT_INDICES
            )
        ):

            core_keypoints_valid = all(

                kp_conf[
                    i
                ]
                >=
                KEYPOINT_CONF_THRESHOLD

                for i
                in CORE_KEYPOINT_INDICES
            )


        valid_hamster = bool(

            box_confidence
            >=
            MIN_HAMSTER_BOX_CONFIDENCE

            and

            strong_keypoints
            >=
            MIN_STRONG_KEYPOINTS

            and

            core_keypoints_valid
        )


    presence_rows.append(
        {
            "frame_index":
                frame_index,

            "timestamp_seconds":
                timestamp_seconds,

            "valid_hamster_detection":
                valid_hamster,

            "box_confidence":
                box_confidence,

            "strong_keypoint_count":
                strong_keypoints,

            "core_keypoints_valid":
                core_keypoints_valid,
        }
    )


    # ========================================================
    # NON-POSE MOTION FEATURES
    # ========================================================

    gray = resize_gray(
        frame
    )


    if previous_gray is None:

        previous_gray = gray

        frame_index += 1

        continue


    difference = cv2.absdiff(
        gray,
        previous_gray
    )


    mean_frame_difference = float(
        difference.mean()
        /
        255.0
    )


    std_frame_difference = float(
        difference.std()
        /
        255.0
    )


    max_frame_difference = float(
        difference.max()
        /
        255.0
    )


    motion_mask = (
        difference
        >=
        MOTION_THRESHOLD
    )


    moving_pixel_fraction = float(
        motion_mask.mean()
    )


    ys, xs = np.where(
        motion_mask
    )


    if len(xs) > 0:

        local_width = gray.shape[1]
        local_height = gray.shape[0]


        motion_cx = float(
            np.mean(xs)
            /
            local_width
        )


        motion_cy = float(
            np.mean(ys)
            /
            local_height
        )


        motion_x_spread = float(
            np.std(xs)
            /
            local_width
        )


        motion_y_spread = float(
            np.std(ys)
            /
            local_height
        )


        left_motion_fraction = float(
            np.mean(
                xs
                <
                local_width / 2
            )
        )


        upper_motion_fraction = float(
            np.mean(
                ys
                <
                local_height / 2
            )
        )


    else:

        motion_cx = np.nan
        motion_cy = np.nan

        motion_x_spread = np.nan
        motion_y_spread = np.nan

        left_motion_fraction = np.nan
        upper_motion_fraction = np.nan


    if (
        np.isfinite(
            previous_motion_cx
        )
        and
        np.isfinite(
            previous_motion_cy
        )
        and
        np.isfinite(
            motion_cx
        )
        and
        np.isfinite(
            motion_cy
        )
    ):

        motion_centroid_dx = (
            motion_cx
            -
            previous_motion_cx
        )


        motion_centroid_dy = (
            motion_cy
            -
            previous_motion_cy
        )


        motion_centroid_speed = float(
            np.sqrt(
                motion_centroid_dx ** 2
                +
                motion_centroid_dy ** 2
            )
            *
            fps
        )


    else:

        motion_centroid_dx = np.nan
        motion_centroid_dy = np.nan
        motion_centroid_speed = np.nan


    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )


    flow_x = flow[
        ...,
        0
    ]


    flow_y = flow[
        ...,
        1
    ]


    flow_magnitude = np.sqrt(
        flow_x ** 2
        +
        flow_y ** 2
    )


    mean_flow_magnitude = float(
        np.mean(
            flow_magnitude
        )
    )


    std_flow_magnitude = float(
        np.std(
            flow_magnitude
        )
    )


    max_flow_magnitude = float(
        np.max(
            flow_magnitude
        )
    )


    mean_flow_x = float(
        np.mean(
            flow_x
        )
    )


    mean_flow_y = float(
        np.mean(
            flow_y
        )
    )


    mean_abs_flow_x = float(
        np.mean(
            np.abs(
                flow_x
            )
        )
    )


    mean_abs_flow_y = float(
        np.mean(
            np.abs(
                flow_y
            )
        )
    )


    horizontal_vertical_ratio = float(
        mean_abs_flow_x
        /
        (
            mean_abs_flow_y
            +
            1e-8
        )
    )


    frame_rows.append(
        {
            "frame_index":
                frame_index,

            "timestamp_seconds":
                timestamp_seconds,

            "mean_frame_difference":
                mean_frame_difference,

            "std_frame_difference":
                std_frame_difference,

            "max_frame_difference":
                max_frame_difference,

            "moving_pixel_fraction":
                moving_pixel_fraction,

            "motion_cx":
                motion_cx,

            "motion_cy":
                motion_cy,

            "motion_x_spread":
                motion_x_spread,

            "motion_y_spread":
                motion_y_spread,

            "left_motion_fraction":
                left_motion_fraction,

            "upper_motion_fraction":
                upper_motion_fraction,

            "motion_centroid_dx":
                motion_centroid_dx,

            "motion_centroid_dy":
                motion_centroid_dy,

            "motion_centroid_speed":
                motion_centroid_speed,

            "mean_flow_magnitude":
                mean_flow_magnitude,

            "std_flow_magnitude":
                std_flow_magnitude,

            "max_flow_magnitude":
                max_flow_magnitude,

            "mean_flow_x":
                mean_flow_x,

            "mean_flow_y":
                mean_flow_y,

            "mean_abs_flow_x":
                mean_abs_flow_x,

            "mean_abs_flow_y":
                mean_abs_flow_y,

            "horizontal_vertical_ratio":
                horizontal_vertical_ratio,
        }
    )


    previous_gray = gray

    previous_motion_cx = motion_cx
    previous_motion_cy = motion_cy


    frame_index += 1


    if (
        frame_index % 100 == 0
        or
        frame_index == frame_count
    ):

        print(
            f"\rProcessing: "
            f"{frame_index}/"
            f"{frame_count} frames",
            end="",
            flush=True
        )


cap.release()


print()
print()


frame_df = pd.DataFrame(
    frame_rows
)


presence_df = pd.DataFrame(
    presence_rows
)


if len(frame_df) == 0:

    raise RuntimeError(
        "No motion features were generated."
    )


presence_df.to_csv(
    PRESENCE_CSV,
    index=False
)


# ============================================================
# PRESENCE SUMMARY
# ============================================================

presence_rate_all = float(
    presence_df[
        "valid_hamster_detection"
    ]
    .mean()
)


print(
    f"Strict hamster-presence rate: "
    f"{presence_rate_all * 100:.1f}%"
)

print()


# ============================================================
# PASS 2
# CREATE WINDOWS
# ============================================================

print("=" * 72)
print("PASS 2: NON-POSE BEHAVIOUR PREDICTION")
print("=" * 72)
print()


median_dt = (
    frame_df[
        "timestamp_seconds"
    ]
    .diff()
)


median_dt = (
    median_dt[
        median_dt > 0
    ]
    .median()
)


if (
    pd.isna(
        median_dt
    )
    or
    median_dt <= 0
):

    median_dt = (
        1.0
        /
        fps
    )


effective_duration = (
    frame_df[
        "timestamp_seconds"
    ].max()
    +
    median_dt
)


window_rows = []

window_start = 0.0
window_number = 1


while (
    window_start
    +
    WINDOW_SECONDS
    <=
    effective_duration
    +
    1e-6
):

    window_end = (
        window_start
        +
        WINDOW_SECONDS
    )


    w = frame_df[
        (
            frame_df[
                "timestamp_seconds"
            ]
            >=
            window_start
        )
        &
        (
            frame_df[
                "timestamp_seconds"
            ]
            <
            window_end
        )
    ].copy()


    if len(w) == 0:

        window_start += (
            STRIDE_SECONDS
        )

        continue


    # --------------------------------------------------------
    # Presence window
    # --------------------------------------------------------

    presence_window = presence_df[
        (
            presence_df[
                "timestamp_seconds"
            ]
            >=
            window_start
        )
        &
        (
            presence_df[
                "timestamp_seconds"
            ]
            <
            window_end
        )
    ]


    hamster_presence_rate = float(
        presence_window[
            "valid_hamster_detection"
        ]
        .mean()
    )


    row = {

        "window_id":
            f"W{window_number:05d}",

        "window_start_s":
            window_start,

        "window_end_s":
            window_end,

        "frames_in_window":
            len(w),

        "hamster_presence_rate":
            hamster_presence_rate,
    }


    # ========================================================
    # FRAME DIFFERENCE FEATURES
    # ========================================================

    for feature in [
        "mean_frame_difference",
        "std_frame_difference",
        "moving_pixel_fraction",
    ]:

        row[
            f"{feature}_mean"
        ] = safe_mean(
            w[
                feature
            ]
        )

        row[
            f"{feature}_std"
        ] = safe_std(
            w[
                feature
            ]
        )

        row[
            f"{feature}_max"
        ] = safe_max(
            w[
                feature
            ]
        )


    row[
        "max_frame_difference_mean"
    ] = safe_mean(
        w[
            "max_frame_difference"
        ]
    )


    row[
        "max_frame_difference_max"
    ] = safe_max(
        w[
            "max_frame_difference"
        ]
    )


    # ========================================================
    # MOTION DISTRIBUTION
    # ========================================================

    for feature in [
        "motion_x_spread",
        "motion_y_spread",
        "left_motion_fraction",
        "upper_motion_fraction",
    ]:

        row[
            f"{feature}_mean"
        ] = safe_mean(
            w[
                feature
            ]
        )

        row[
            f"{feature}_std"
        ] = safe_std(
            w[
                feature
            ]
        )


    # ========================================================
    # MOTION CENTROID
    # ========================================================

    row[
        "motion_cx_range"
    ] = (
        safe_max(
            w[
                "motion_cx"
            ]
        )
        -
        safe_min(
            w[
                "motion_cx"
            ]
        )
    )


    row[
        "motion_cy_range"
    ] = (
        safe_max(
            w[
                "motion_cy"
            ]
        )
        -
        safe_min(
            w[
                "motion_cy"
            ]
        )
    )


    for feature in [
        "motion_centroid_dx",
        "motion_centroid_dy",
        "motion_centroid_speed",
    ]:

        row[
            f"{feature}_mean"
        ] = safe_mean(
            w[
                feature
            ]
        )

        row[
            f"{feature}_std"
        ] = safe_std(
            w[
                feature
            ]
        )


    row[
        "motion_centroid_speed_max"
    ] = safe_max(
        w[
            "motion_centroid_speed"
        ]
    )


    # ========================================================
    # OPTICAL FLOW
    # ========================================================

    for feature in [
        "mean_flow_magnitude",
        "std_flow_magnitude",
        "mean_abs_flow_x",
        "mean_abs_flow_y",
        "horizontal_vertical_ratio",
    ]:

        row[
            f"{feature}_mean"
        ] = safe_mean(
            w[
                feature
            ]
        )

        row[
            f"{feature}_std"
        ] = safe_std(
            w[
                feature
            ]
        )


    row[
        "mean_flow_magnitude_max"
    ] = safe_max(
        w[
            "mean_flow_magnitude"
        ]
    )


    row[
        "max_flow_magnitude_mean"
    ] = safe_mean(
        w[
            "max_flow_magnitude"
        ]
    )


    row[
        "max_flow_magnitude_max"
    ] = safe_max(
        w[
            "max_flow_magnitude"
        ]
    )


    for feature in [
        "mean_flow_x",
        "mean_flow_y",
    ]:

        row[
            f"{feature}_mean"
        ] = safe_mean(
            w[
                feature
            ]
        )

        row[
            f"{feature}_std"
        ] = safe_std(
            w[
                feature
            ]
        )


    window_rows.append(
        row
    )


    window_number += 1

    window_start += (
        STRIDE_SECONDS
    )


features_df = pd.DataFrame(
    window_rows
)


# ============================================================
# VERIFY FROZEN NON-POSE FEATURE SET
# ============================================================

missing_features = [
    feature
    for feature in feature_columns
    if feature not in features_df.columns
]


if missing_features:

    print()
    print(
        "Missing required classifier features:"
    )


    for feature in missing_features:

        print(
            f" - {feature}"
        )


    raise RuntimeError(
        "Demo feature extraction does not match "
        "the frozen non-pose model."
    )


# ============================================================
# PREDICT
# ============================================================

X = features_df[
    feature_columns
].copy()


for col in feature_columns:

    X[
        col
    ] = pd.to_numeric(
        X[
            col
        ],
        errors="coerce"
    )


X_imp = imputer.transform(
    X
)


pred_encoded = classifier.predict(
    X_imp
)


probabilities = classifier.predict_proba(
    X_imp
)


predicted_behaviour = (
    label_encoder.inverse_transform(
        pred_encoded
    )
)


model_score = (
    probabilities.max(
        axis=1
    )
)


features_df[
    "raw_prediction"
] = predicted_behaviour


features_df[
    "model_score"
] = model_score


# ============================================================
# DISPLAY / REJECTION LOGIC
# ============================================================

display_predictions = []

display_states = []


for _, row in features_df.iterrows():

    # --------------------------------------------------------
    # STATE 1
    # No reliable hamster presence
    # --------------------------------------------------------

    if (
        row[
            "hamster_presence_rate"
        ]
        <
        MIN_HAMSTER_PRESENCE_RATE
    ):

        display_predictions.append(
            "No hamster detected"
        )

        display_states.append(
            "no_hamster"
        )


    # --------------------------------------------------------
    # STATE 2
    # Hamster present but behaviour score too low
    # --------------------------------------------------------

    elif (
        row[
            "model_score"
        ]
        <
        MIN_BEHAVIOUR_SCORE
    ):

        display_predictions.append(
            "Unclassified"
        )

        display_states.append(
            "uncertain"
        )


    # --------------------------------------------------------
    # STATE 3
    # Accepted non-pose behaviour prediction
    # --------------------------------------------------------

    else:

        display_predictions.append(
            row[
                "raw_prediction"
            ]
        )

        display_states.append(
            "classified"
        )


features_df[
    "display_prediction"
] = display_predictions


features_df[
    "display_state"
] = display_states


# ============================================================
# CLASS SCORES
# ============================================================

for i, class_name in enumerate(
    label_encoder.classes_
):

    safe_name = (
        class_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )


    features_df[
        f"score_{safe_name}"
    ] = probabilities[
        :,
        i
    ]


features_df.to_csv(
    PREDICTION_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    f"Behaviour windows created: "
    f"{len(features_df)}"
)

print()


print(
    "Demo classification states:"
)

print(
    features_df[
        "display_state"
    ]
    .value_counts()
    .to_string()
)

print()


print(
    "Displayed labels:"
)

print(
    features_df[
        "display_prediction"
    ]
    .value_counts()
    .to_string()
)

print()


accepted = features_df[
    features_df[
        "display_state"
    ]
    ==
    "classified"
]


print(
    "Accepted behaviour predictions:"
)

if len(accepted) > 0:

    print(
        accepted[
            "display_prediction"
        ]
        .value_counts()
        .to_string()
    )

else:

    print(
        "No behaviour windows passed "
        "the demo rejection rules."
    )

print()


# ============================================================
# MAP FRAME TO WINDOW
# ============================================================

def prediction_for_time(
    timestamp
):

    candidates = features_df[
        (
            features_df[
                "window_start_s"
            ]
            <=
            timestamp
        )
        &
        (
            features_df[
                "window_end_s"
            ]
            >
            timestamp
        )
    ]


    if len(candidates) == 0:

        return (
            "Waiting...",
            0.0,
            "waiting",
            0.0,
        )


    centres = (
        (
            candidates[
                "window_start_s"
            ]
            +
            candidates[
                "window_end_s"
            ]
        )
        /
        2.0
    )


    best_row = candidates.loc[
        (
            centres
            -
            timestamp
        )
        .abs()
        .idxmin()
    ]


    return (

        best_row[
            "display_prediction"
        ],

        float(
            best_row[
                "model_score"
            ]
        ),

        best_row[
            "display_state"
        ],

        float(
            best_row[
                "hamster_presence_rate"
            ]
        ),
    )


# ============================================================
# PASS 3
# CREATE ANNOTATED VIDEO
# ============================================================

print()
print("=" * 72)
print("PASS 3: CREATING ANNOTATED VIDEO")
print("=" * 72)
print()


cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(
    str(
        ANNOTATED_VIDEO
    ),
    fourcc,
    fps,
    (
        width,
        height
    ),
)


if not writer.isOpened():

    raise RuntimeError(
        "Could not create output video."
    )


frame_index = 0


while True:

    ok, frame = cap.read()

    if not ok:
        break


    timestamp = (
        frame_index
        /
        fps
    )


    (
        behaviour,
        score,
        state,
        presence_rate,

    ) = prediction_for_time(
        timestamp
    )


    # --------------------------------------------------------
    # Overlay panel
    # --------------------------------------------------------

    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (
            20,
            20
        ),
        (
            720,
            180
        ),
        (
            0,
            0,
            0
        ),
        -1,
    )


    frame = cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0
    )


    # --------------------------------------------------------
    # Main label
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Non-pose behaviour: {behaviour}",
        (
            40,
            65
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA,
    )


    # --------------------------------------------------------
    # Secondary information
    # --------------------------------------------------------

    if state == "classified":

        secondary_text = (
            f"Model score: "
            f"{score * 100:.1f}%"
        )


    elif state == "uncertain":

        secondary_text = (
            f"Prediction rejected "
            f"(score {score * 100:.1f}%)"
        )


    elif state == "no_hamster":

        secondary_text = (
            f"Hamster presence in window: "
            f"{presence_rate * 100:.0f}%"
        )


    else:

        secondary_text = ""


    cv2.putText(
        frame,
        secondary_text,
        (
            40,
            110
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA,
    )


    # --------------------------------------------------------
    # Presence line
    # --------------------------------------------------------

    cv2.putText(
        frame,
        (
            f"Presence gate: "
            f"{presence_rate * 100:.0f}%"
        ),
        (
            40,
            145
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (
            255,
            255,
            255
        ),
        1,
        cv2.LINE_AA,
    )


    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Time: {timestamp:.1f}s",
        (
            40,
            170
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (
            255,
            255,
            255
        ),
        1,
        cv2.LINE_AA,
    )


    writer.write(
        frame
    )


    frame_index += 1


    if (
        frame_index % 100 == 0
        or
        frame_index == frame_count
    ):

        print(
            f"\rRendering: "
            f"{frame_index}/"
            f"{frame_count} frames",
            end="",
            flush=True
        )


cap.release()

writer.release()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()

print("=" * 72)
print("NON-POSE PRESENCE-GATED VIDEO DEMO COMPLETE")
print("=" * 72)
print()


print(
    f"Annotated video:\n"
    f"{ANNOTATED_VIDEO}"
)

print()


print(
    f"Behaviour predictions:\n"
    f"{PREDICTION_CSV}"
)

print()


print(
    f"Presence gate results:\n"
    f"{PRESENCE_CSV}"
)

print()


print(
    "Demo logic:"
)

print(
    "- YOLO presence gate below threshold "
    "-> No hamster detected"
)

print(
    "- Hamster present but behaviour score "
    "below threshold -> Unclassified"
)

print(
    "- Hamster present and behaviour score "
    "accepted -> non-pose behaviour label"
)

print()


print(
    "IMPORTANT:"
)

print(
    "The XGBoost behaviour classifier itself remains fully non-pose."
)

print(
    "YOLO Pose is used only as a practical hamster-presence gate "
    "for the demonstration artefact."
)

print(
    "The presence gate and 0.70 behaviour-score threshold were NOT "
    "used in the reported non-pose held-out Test evaluation."
)

print(
    "The displayed XGBoost value is labelled 'Model score' because "
    "the classifier probabilities have not been formally calibrated."
)

print()