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
        "Run the exploratory Core-10 hamster pose + "
        "behaviour-classification artefact on a video."
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
        "Inference device: auto, cpu, mps, cuda, cuda:0, etc. "
        "Default: auto."
    ),
)

args = parser.parse_args()

VIDEO_PATH = args.video.expanduser()

if not VIDEO_PATH.is_absolute():
    VIDEO_PATH = (Path.cwd() / VIDEO_PATH).resolve()


MODEL_DIR = (
    PACKAGE_ROOT
    / "models"
    / "core10"
)

POSE_MODEL_PATH = (
    MODEL_DIR
    / "core10_pose_best.pt"
)

CLASSIFIER_DIR = MODEL_DIR

OUTPUT_DIR = (
    PACKAGE_ROOT
    / "outputs"
    / "core10"
    / "generated_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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

YOLO_CONFIDENCE = 0.25
KEYPOINT_CONF_THRESHOLD = 0.30
MAX_INTERPOLATION_GAP = 5

WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 1.0

MIN_HAMSTER_BOX_CONFIDENCE = 0.50
MIN_STRONG_KEYPOINTS = 4

REQUIRE_CORE_KEYPOINTS = True

CORE_KEYPOINTS = [
    "neck_shoulder",
    "spine_mid",
]

MIN_HAMSTER_PRESENCE_RATE = 0.40
MIN_BEHAVIOUR_SCORE = 0.70


BODY_PARTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "neck_shoulder",
    "spine_mid",
    "tail_base",
    "left_forepaw",
    "right_forepaw",
]


SKELETON_EDGES = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("nose", "neck_shoulder"),
    ("neck_shoulder", "spine_mid"),
    ("spine_mid", "tail_base"),
    ("neck_shoulder", "left_forepaw"),
    ("neck_shoulder", "right_forepaw"),
]


# ============================================================
# OUTPUT FILES
# ============================================================

stem = VIDEO_PATH.stem

POSE_CSV = (
    OUTPUT_DIR
    / f"{stem}_pose.csv"
)

PREDICTION_CSV = (
    OUTPUT_DIR
    / f"{stem}_behaviour_predictions.csv"
)

ANNOTATED_VIDEO = (
    OUTPUT_DIR
    / f"{stem}_annotated.mp4"
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


def distance(
    x1,
    y1,
    x2,
    y2
):

    return np.sqrt(
        (x1 - x2) ** 2
        +
        (y1 - y2) ** 2
    )


# ============================================================
# CHECK FILES
# ============================================================

if not VIDEO_PATH.exists():

    raise FileNotFoundError(
        f"Video not found:\n{VIDEO_PATH}"
    )


if not POSE_MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Pose model not found:\n{POSE_MODEL_PATH}"
    )


required_classifier_files = [
    "xgboost_core10_model.joblib",
    "median_imputer.joblib",
    "label_encoder.joblib",
    "feature_columns.joblib",
]


for filename in required_classifier_files:

    path = CLASSIFIER_DIR / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Missing classifier file:\n{path}"
        )


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 72)
print("CORE-10 HAMSTER BEHAVIOUR VIDEO DEMO")
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

print("Loading YOLO Pose model...")

pose_model = YOLO(
    POSE_MODEL_PATH
)

print("Loading behaviour classifier...")

classifier = joblib.load(
    CLASSIFIER_DIR
    / "xgboost_core10_model.joblib"
)

imputer = joblib.load(
    CLASSIFIER_DIR
    / "median_imputer.joblib"
)

label_encoder = joblib.load(
    CLASSIFIER_DIR
    / "label_encoder.joblib"
)

feature_columns = joblib.load(
    CLASSIFIER_DIR
    / "feature_columns.joblib"
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


print("Video:")

print(
    f"File:       "
    f"{VIDEO_PATH.name}"
)

print(
    f"Resolution: "
    f"{width} × {height}"
)

print(
    f"FPS:        "
    f"{fps:.3f}"
)

print(
    f"Frames:     "
    f"{frame_count}"
)

print(
    f"Duration:   "
    f"{duration:.2f} s"
)

print()


# ============================================================
# PASS 1 — POSE ESTIMATION
# ============================================================

print("=" * 72)
print("PASS 1: CORE-10 POSE ESTIMATION")
print("=" * 72)
print()


pose_rows = []

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


    result = pose_model.predict(
        frame,
        conf=YOLO_CONFIDENCE,
        device=DEVICE,
        verbose=False,
    )[0]


    row = {

        "frame_index":
            frame_index,

        "timestamp_seconds":
            timestamp,

        "detection_found":
            False,

        "valid_hamster_detection":
            False,

        "box_confidence":
            np.nan,

        "strong_keypoint_count":
            0,

        "core_keypoints_valid":
            False,
    }


    for bodypart in BODY_PARTS:

        row[
            f"{bodypart}_x"
        ] = np.nan

        row[
            f"{bodypart}_y"
        ] = np.nan

        row[
            f"{bodypart}_confidence"
        ] = np.nan


    if (
        result.boxes is not None
        and
        len(result.boxes) > 0
        and
        result.keypoints is not None
    ):

        box_confidences = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )


        best_index = int(
            np.argmax(
                box_confidences
            )
        )


        kp_xy = (
            result.keypoints.xy[
                best_index
            ]
            .detach()
            .cpu()
            .numpy()
        )


        kp_conf = (
            result.keypoints.conf[
                best_index
            ]
            .detach()
            .cpu()
            .numpy()
        )


        if (
            len(kp_xy)
            >=
            len(BODY_PARTS)
        ):

            row[
                "detection_found"
            ] = True


            row[
                "box_confidence"
            ] = float(
                box_confidences[
                    best_index
                ]
            )


            for i, bodypart in enumerate(
                BODY_PARTS
            ):

                row[
                    f"{bodypart}_x"
                ] = float(
                    kp_xy[
                        i,
                        0
                    ]
                )

                row[
                    f"{bodypart}_y"
                ] = float(
                    kp_xy[
                        i,
                        1
                    ]
                )

                row[
                    f"{bodypart}_confidence"
                ] = float(
                    kp_conf[
                        i
                    ]
                )


            strong_keypoint_count = 0


            for bodypart in BODY_PARTS:

                kp_score = row[
                    f"{bodypart}_confidence"
                ]


                if (
                    not pd.isna(
                        kp_score
                    )
                    and
                    kp_score
                    >=
                    KEYPOINT_CONF_THRESHOLD
                ):

                    strong_keypoint_count += 1


            row[
                "strong_keypoint_count"
            ] = strong_keypoint_count


            core_keypoints_valid = all(

                (
                    not pd.isna(
                        row[
                            f"{bodypart}_confidence"
                        ]
                    )
                )
                and
                (
                    row[
                        f"{bodypart}_confidence"
                    ]
                    >=
                    KEYPOINT_CONF_THRESHOLD
                )

                for bodypart
                in CORE_KEYPOINTS
            )


            row[
                "core_keypoints_valid"
            ] = core_keypoints_valid


            box_ok = (
                row[
                    "box_confidence"
                ]
                >=
                MIN_HAMSTER_BOX_CONFIDENCE
            )


            keypoints_ok = (
                strong_keypoint_count
                >=
                MIN_STRONG_KEYPOINTS
            )


            core_ok = (
                core_keypoints_valid
                if REQUIRE_CORE_KEYPOINTS
                else True
            )


            row[
                "valid_hamster_detection"
            ] = bool(
                box_ok
                and
                keypoints_ok
                and
                core_ok
            )


    pose_rows.append(
        row
    )


    frame_index += 1


    if (
        frame_index % 100 == 0
        or
        frame_index == frame_count
    ):

        print(
            f"\rPose inference: "
            f"{frame_index}/"
            f"{frame_count} frames",
            end="",
            flush=True
        )


cap.release()


print()
print()


pose_df = pd.DataFrame(
    pose_rows
)


pose_df.to_csv(
    POSE_CSV,
    index=False
)


# ============================================================
# DETECTION SUMMARY
# ============================================================

raw_detection_rate = float(
    pose_df[
        "detection_found"
    ]
    .mean()
)

valid_hamster_rate = float(
    pose_df[
        "valid_hamster_detection"
    ]
    .mean()
)


print(
    f"Raw YOLO detection rate: "
    f"{raw_detection_rate * 100:.1f}%"
)

print(
    f"Strict hamster-presence rate: "
    f"{valid_hamster_rate * 100:.1f}%"
)

print()


# ============================================================
# PREPROCESS KEYPOINTS
# ============================================================

df = pose_df.copy()


for bodypart in BODY_PARTS:

    x_col = f"{bodypart}_x"
    y_col = f"{bodypart}_y"
    conf_col = f"{bodypart}_confidence"


    low_confidence = (
        df[
            conf_col
        ]
        <
        KEYPOINT_CONF_THRESHOLD
    )


    invalid_hamster = (
        ~df[
            "valid_hamster_detection"
        ]
    )


    reject_point = (
        low_confidence
        |
        invalid_hamster
    )


    df.loc[
        reject_point,
        [
            x_col,
            y_col,
        ]
    ] = np.nan


    df[
        f"{bodypart}_x_norm"
    ] = (
        df[
            x_col
        ]
        /
        width
    )


    df[
        f"{bodypart}_y_norm"
    ] = (
        df[
            y_col
        ]
        /
        height
    )


    df[
        f"{bodypart}_x_norm"
    ] = (
        df[
            f"{bodypart}_x_norm"
        ]
        .interpolate(
            method="linear",
            limit=MAX_INTERPOLATION_GAP,
            limit_area="inside",
        )
    )


    df[
        f"{bodypart}_y_norm"
    ] = (
        df[
            f"{bodypart}_y_norm"
        ]
        .interpolate(
            method="linear",
            limit=MAX_INTERPOLATION_GAP,
            limit_area="inside",
        )
    )


# ============================================================
# TIMING
# ============================================================

time_diff = (
    df[
        "timestamp_seconds"
    ]
    .diff()
)


median_dt = (
    1.0
    /
    fps
)


dt = (
    time_diff
    .where(
        time_diff > 0,
        median_dt
    )
)


# ============================================================
# CORE GEOMETRY
# ============================================================

df["body_length"] = distance(
    df["neck_shoulder_x_norm"],
    df["neck_shoulder_y_norm"],
    df["tail_base_x_norm"],
    df["tail_base_y_norm"],
)


df["neck_spine_distance"] = distance(
    df["neck_shoulder_x_norm"],
    df["neck_shoulder_y_norm"],
    df["spine_mid_x_norm"],
    df["spine_mid_y_norm"],
)


df["spine_tail_distance"] = distance(
    df["spine_mid_x_norm"],
    df["spine_mid_y_norm"],
    df["tail_base_x_norm"],
    df["tail_base_y_norm"],
)


df["nose_neck_distance"] = distance(
    df["nose_x_norm"],
    df["nose_y_norm"],
    df["neck_shoulder_x_norm"],
    df["neck_shoulder_y_norm"],
)


df["nose_spine_distance"] = distance(
    df["nose_x_norm"],
    df["nose_y_norm"],
    df["spine_mid_x_norm"],
    df["spine_mid_y_norm"],
)


# ============================================================
# TORSO CENTRE
# ============================================================

df[
    "torso_x"
] = df[
    [
        "neck_shoulder_x_norm",
        "spine_mid_x_norm",
        "tail_base_x_norm",
    ]
].mean(
    axis=1,
    skipna=True
)


df[
    "torso_y"
] = df[
    [
        "neck_shoulder_y_norm",
        "spine_mid_y_norm",
        "tail_base_y_norm",
    ]
].mean(
    axis=1,
    skipna=True
)


# ============================================================
# BODY ORIENTATION
# ============================================================

body_dx = (
    df[
        "neck_shoulder_x_norm"
    ]
    -
    df[
        "tail_base_x_norm"
    ]
)


body_dy = (
    df[
        "neck_shoulder_y_norm"
    ]
    -
    df[
        "tail_base_y_norm"
    ]
)


df[
    "body_angle"
] = np.arctan2(
    body_dy,
    body_dx
)


df[
    "body_verticality"
] = (
    np.abs(
        body_dy
    )
    /
    (
        df[
            "body_length"
        ]
        +
        1e-8
    )
)


df[
    "near_vertical"
] = (
    df[
        "body_verticality"
    ]
    >=
    0.75
).astype(float)


# ============================================================
# SPEEDS — CORE + FOREPAWS
# ============================================================

movement_parts = [
    "nose",
    "neck_shoulder",
    "spine_mid",
    "tail_base",
    "left_forepaw",
    "right_forepaw",
]


for bodypart in movement_parts:

    dx = (
        df[
            f"{bodypart}_x_norm"
        ]
        .diff()
    )

    dy = (
        df[
            f"{bodypart}_y_norm"
        ]
        .diff()
    )


    df[
        f"{bodypart}_dx_rate"
    ] = (
        dx
        /
        dt
    )


    df[
        f"{bodypart}_dy_rate"
    ] = (
        dy
        /
        dt
    )


    df[
        f"{bodypart}_speed"
    ] = (
        np.sqrt(
            dx ** 2
            +
            dy ** 2
        )
        /
        dt
    )


torso_dx = (
    df[
        "torso_x"
    ]
    .diff()
)


torso_dy = (
    df[
        "torso_y"
    ]
    .diff()
)


df[
    "torso_dx_rate"
] = (
    torso_dx
    /
    dt
)


df[
    "torso_dy_rate"
] = (
    torso_dy
    /
    dt
)


df[
    "torso_speed"
] = (
    np.sqrt(
        torso_dx ** 2
        +
        torso_dy ** 2
    )
    /
    dt
)


# ============================================================
# RELATIVE HEAD MOVEMENT
# ============================================================

df[
    "nose_rel_neck_x"
] = (
    df[
        "nose_x_norm"
    ]
    -
    df[
        "neck_shoulder_x_norm"
    ]
)


df[
    "nose_rel_neck_y"
] = (
    df[
        "nose_y_norm"
    ]
    -
    df[
        "neck_shoulder_y_norm"
    ]
)


rel_neck_dx = (
    df[
        "nose_rel_neck_x"
    ]
    .diff()
)


rel_neck_dy = (
    df[
        "nose_rel_neck_y"
    ]
    .diff()
)


df[
    "nose_relative_neck_speed"
] = (
    np.sqrt(
        rel_neck_dx ** 2
        +
        rel_neck_dy ** 2
    )
    /
    dt
)


df[
    "nose_rel_spine_x"
] = (
    df[
        "nose_x_norm"
    ]
    -
    df[
        "spine_mid_x_norm"
    ]
)


df[
    "nose_rel_spine_y"
] = (
    df[
        "nose_y_norm"
    ]
    -
    df[
        "spine_mid_y_norm"
    ]
)


rel_spine_dx = (
    df[
        "nose_rel_spine_x"
    ]
    .diff()
)


rel_spine_dy = (
    df[
        "nose_rel_spine_y"
    ]
    .diff()
)


df[
    "relative_head_speed"
] = (
    np.sqrt(
        rel_spine_dx ** 2
        +
        rel_spine_dy ** 2
    )
    /
    dt
)


# ============================================================
# FOREPAW GEOMETRY
# ============================================================

df[
    "left_forepaw_nose_distance"
] = distance(
    df["left_forepaw_x_norm"],
    df["left_forepaw_y_norm"],
    df["nose_x_norm"],
    df["nose_y_norm"],
)


df[
    "right_forepaw_nose_distance"
] = distance(
    df["right_forepaw_x_norm"],
    df["right_forepaw_y_norm"],
    df["nose_x_norm"],
    df["nose_y_norm"],
)


df[
    "left_forepaw_neck_distance"
] = distance(
    df["left_forepaw_x_norm"],
    df["left_forepaw_y_norm"],
    df["neck_shoulder_x_norm"],
    df["neck_shoulder_y_norm"],
)


df[
    "right_forepaw_neck_distance"
] = distance(
    df["right_forepaw_x_norm"],
    df["right_forepaw_y_norm"],
    df["neck_shoulder_x_norm"],
    df["neck_shoulder_y_norm"],
)


df[
    "forepaw_distance"
] = distance(
    df["left_forepaw_x_norm"],
    df["left_forepaw_y_norm"],
    df["right_forepaw_x_norm"],
    df["right_forepaw_y_norm"],
)


# ============================================================
# FOREPAWS RELATIVE TO TORSO
# ============================================================

for side in [
    "left_forepaw",
    "right_forepaw",
]:

    df[
        f"{side}_rel_torso_x"
    ] = (
        df[
            f"{side}_x_norm"
        ]
        -
        df[
            "torso_x"
        ]
    )


    df[
        f"{side}_rel_torso_y"
    ] = (
        df[
            f"{side}_y_norm"
        ]
        -
        df[
            "torso_y"
        ]
    )


    rel_dx = (
        df[
            f"{side}_rel_torso_x"
        ]
        .diff()
    )


    rel_dy = (
        df[
            f"{side}_rel_torso_y"
        ]
        .diff()
    )


    df[
        f"{side}_relative_torso_speed"
    ] = (
        np.sqrt(
            rel_dx ** 2
            +
            rel_dy ** 2
        )
        /
        dt
    )


df[
    "mean_forepaw_speed"
] = df[
    [
        "left_forepaw_speed",
        "right_forepaw_speed",
    ]
].mean(
    axis=1,
    skipna=True
)


# ============================================================
# BODY ANGLE CHANGE
# ============================================================

angles = (
    df[
        "body_angle"
    ]
    .to_numpy(
        dtype=float
    )
)


valid_angles = (
    ~np.isnan(
        angles
    )
)


unwrapped_angles = (
    angles.copy()
)


if valid_angles.sum() >= 2:

    unwrapped_angles[
        valid_angles
    ] = np.unwrap(
        angles[
            valid_angles
        ]
    )


angle_series = pd.Series(
    unwrapped_angles,
    index=df.index
)


df[
    "body_angle_change_rate"
] = (
    angle_series
    .diff()
    .abs()
    /
    dt
)


# ============================================================
# PASS 2 — CREATE WINDOWS
# ============================================================

print("=" * 72)
print("PASS 2: CORE-10 BEHAVIOUR PREDICTION")
print("=" * 72)
print()


window_rows = []

window_start = 0.0
window_number = 1


effective_duration = (
    df[
        "timestamp_seconds"
    ].max()
    +
    median_dt
)


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


    w = df[
        (
            df[
                "timestamp_seconds"
            ]
            >=
            window_start
        )
        &
        (
            df[
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


    row = {

        "window_id":
            f"W{window_number:05d}",

        "window_start_s":
            window_start,

        "window_end_s":
            window_end,

        "frames_in_window":
            len(w),

        "detection_rate":
            float(
                w[
                    "detection_found"
                ]
                .mean()
            ),

        "hamster_presence_rate":
            float(
                w[
                    "valid_hamster_detection"
                ]
                .mean()
            ),
    }


    # ========================================================
    # QUALITY FEATURES
    # ========================================================

    row[
        "box_confidence_mean"
    ] = safe_mean(
        w[
            "box_confidence"
        ]
    )


    row[
        "strong_keypoint_count_mean"
    ] = safe_mean(
        w[
            "strong_keypoint_count"
        ]
    )


    availability_values = []


    for bodypart in BODY_PARTS:

        available = (
            w[
                f"{bodypart}_x_norm"
            ].notna()
            &
            w[
                f"{bodypart}_y_norm"
            ].notna()
        )


        availability = float(
            available.mean()
        )


        row[
            f"{bodypart}_availability"
        ] = availability


        row[
            f"{bodypart}_confidence_mean"
        ] = safe_mean(
            w[
                f"{bodypart}_confidence"
            ]
        )


        availability_values.append(
            availability
        )


    row[
        "overall_keypoint_availability"
    ] = float(
        np.mean(
            availability_values
        )
    )


    # ========================================================
    # CORE GEOMETRY
    # ========================================================

    geometry_features = [
        "body_length",
        "neck_spine_distance",
        "spine_tail_distance",
        "nose_neck_distance",
        "nose_spine_distance",
        "body_verticality",
    ]


    for feature in geometry_features:

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
    # BODY ANGLE
    # ========================================================

    row[
        "body_angle_mean"
    ] = safe_mean(
        w[
            "body_angle"
        ]
    )


    row[
        "body_angle_std"
    ] = safe_std(
        w[
            "body_angle"
        ]
    )


    row[
        "body_angle_change_mean"
    ] = safe_mean(
        w[
            "body_angle_change_rate"
        ]
    )


    row[
        "body_angle_change_max"
    ] = safe_max(
        w[
            "body_angle_change_rate"
        ]
    )


    row[
        "near_vertical_fraction"
    ] = safe_mean(
        w[
            "near_vertical"
        ]
    )


    # ========================================================
    # TORSO POSITION
    # ========================================================

    row[
        "torso_x_mean"
    ] = safe_mean(
        w[
            "torso_x"
        ]
    )


    row[
        "torso_y_mean"
    ] = safe_mean(
        w[
            "torso_y"
        ]
    )


    row[
        "torso_x_range"
    ] = (
        safe_max(
            w[
                "torso_x"
            ]
        )
        -
        safe_min(
            w[
                "torso_x"
            ]
        )
    )


    row[
        "torso_y_range"
    ] = (
        safe_max(
            w[
                "torso_y"
            ]
        )
        -
        safe_min(
            w[
                "torso_y"
            ]
        )
    )


    valid_torso = w[
        [
            "torso_x",
            "torso_y",
        ]
    ].dropna()


    if len(valid_torso) >= 2:

        first = valid_torso.iloc[0]
        last = valid_torso.iloc[-1]


        net_dx = (
            last[
                "torso_x"
            ]
            -
            first[
                "torso_x"
            ]
        )


        net_dy = (
            last[
                "torso_y"
            ]
            -
            first[
                "torso_y"
            ]
        )


        row[
            "torso_net_dx"
        ] = net_dx


        row[
            "torso_net_dy"
        ] = net_dy


        row[
            "torso_net_displacement"
        ] = float(
            np.sqrt(
                net_dx ** 2
                +
                net_dy ** 2
            )
        )


    else:

        row[
            "torso_net_dx"
        ] = np.nan

        row[
            "torso_net_dy"
        ] = np.nan

        row[
            "torso_net_displacement"
        ] = np.nan


    # ========================================================
    # SPEED FEATURES
    # ========================================================

    speed_features = [
        "nose_speed",
        "neck_shoulder_speed",
        "spine_mid_speed",
        "tail_base_speed",
        "torso_speed",
        "relative_head_speed",
        "nose_relative_neck_speed",
        "left_forepaw_speed",
        "right_forepaw_speed",
        "mean_forepaw_speed",
        "left_forepaw_relative_torso_speed",
        "right_forepaw_relative_torso_speed",
    ]


    for feature in speed_features:

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


    # ========================================================
    # DIRECTIONAL MOVEMENT
    # ========================================================

    directional_features = [
        "torso_dx_rate",
        "torso_dy_rate",
        "neck_shoulder_dx_rate",
        "neck_shoulder_dy_rate",
        "nose_dx_rate",
        "nose_dy_rate",
        "left_forepaw_dx_rate",
        "left_forepaw_dy_rate",
        "right_forepaw_dx_rate",
        "right_forepaw_dy_rate",
    ]


    for feature in directional_features:

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
    # RELATIVE HEAD POSITION
    # ========================================================

    relative_features = [
        "nose_rel_neck_x",
        "nose_rel_neck_y",
        "nose_rel_spine_x",
        "nose_rel_spine_y",
    ]


    for feature in relative_features:

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
    # FOREPAW GEOMETRY
    # ========================================================

    paw_geometry_features = [
        "left_forepaw_nose_distance",
        "right_forepaw_nose_distance",
        "left_forepaw_neck_distance",
        "right_forepaw_neck_distance",
        "forepaw_distance",
        "left_forepaw_rel_torso_x",
        "left_forepaw_rel_torso_y",
        "right_forepaw_rel_torso_x",
        "right_forepaw_rel_torso_y",
    ]


    for feature in paw_geometry_features:

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
# VERIFY FEATURE SET
# ============================================================

missing_features = [

    feature

    for feature in feature_columns

    if feature not in features_df.columns
]


if missing_features:

    print()
    print("Missing required classifier features:")

    for feature in missing_features:
        print(f" - {feature}")


    raise RuntimeError(
        "Demo feature extraction does not match "
        "the frozen Core-10 model."
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


prediction_score = (
    probabilities.max(
        axis=1
    )
)


features_df[
    "raw_prediction"
] = predicted_behaviour


features_df[
    "model_score"
] = prediction_score


# ============================================================
# DISPLAY / REJECTION LOGIC
# ============================================================

display_predictions = []
display_states = []


for _, row in features_df.iterrows():

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

print()
print(
    f"Behaviour windows created: "
    f"{len(features_df)}"
)

print()

print("Demo classification states:")

print(
    features_df[
        "display_state"
    ]
    .value_counts()
    .to_string()
)

print()

print("Displayed labels:")

print(
    features_df[
        "display_prediction"
    ]
    .value_counts()
    .to_string()
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
# DRAWING
# ============================================================

def get_point(
    pose_row,
    bodypart
):

    conf = pose_row[
        f"{bodypart}_confidence"
    ]

    x = pose_row[
        f"{bodypart}_x"
    ]

    y = pose_row[
        f"{bodypart}_y"
    ]


    if (
        pd.isna(conf)
        or
        pd.isna(x)
        or
        pd.isna(y)
        or
        conf
        <
        KEYPOINT_CONF_THRESHOLD
    ):

        return None


    return (
        int(x),
        int(y)
    )


def draw_pose(
    frame,
    pose_row
):

    if not bool(
        pose_row[
            "valid_hamster_detection"
        ]
    ):

        return frame


    points = {}


    for bodypart in BODY_PARTS:

        points[
            bodypart
        ] = get_point(
            pose_row,
            bodypart
        )


    for start, end in SKELETON_EDGES:

        p1 = points[
            start
        ]

        p2 = points[
            end
        ]


        if (
            p1 is not None
            and
            p2 is not None
        ):

            cv2.line(
                frame,
                p1,
                p2,
                (
                    255,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA,
            )


    for bodypart, point in points.items():

        if point is None:
            continue


        cv2.circle(
            frame,
            point,
            4,
            (
                0,
                255,
                255
            ),
            -1,
            cv2.LINE_AA,
        )


    return frame


# ============================================================
# PASS 3 — ANNOTATED VIDEO
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


    pose_row = pose_df.iloc[
        frame_index
    ]


    frame = draw_pose(
        frame,
        pose_row
    )


    (
        behaviour,
        model_score,
        state,
        presence_rate,

    ) = prediction_for_time(
        timestamp
    )


    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (
            20,
            20
        ),
        (
            680,
            160
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


    cv2.putText(
        frame,
        f"Behaviour: {behaviour}",
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


    if state == "classified":

        secondary_text = (
            f"Model score: "
            f"{model_score * 100:.1f}%"
        )

    elif state == "uncertain":

        secondary_text = (
            f"Prediction rejected "
            f"(score {model_score * 100:.1f}%)"
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
            105
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA,
    )


    cv2.putText(
        frame,
        f"Time: {timestamp:.1f}s",
        (
            40,
            140
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
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
# FINAL
# ============================================================

print()
print()

print("=" * 72)
print("CORE-10 VIDEO DEMO COMPLETE")
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
    f"Pose detections:\n"
    f"{POSE_CSV}"
)

print()

print(
    "Demo-only thresholds:"
)

print(
    f"- minimum hamster presence: "
    f"{MIN_HAMSTER_PRESENCE_RATE * 100:.0f}%"
)

print(
    f"- minimum behaviour model score: "
    f"{MIN_BEHAVIOUR_SCORE * 100:.0f}%"
)

print()

print(
    "These display thresholds are for the demonstration artefact only "
    "and were not used in the reported held-out Test evaluation."
)

print()