"""
EyeControl AI - Gaze Calibration
Phase 5.1

Collects personalized eye-feature reference points for:
CENTER, LEFT, RIGHT, UP, DOWN

The calibration does NOT modify the trained model.
It only records the user's real eye feature values and saves them
for the future gaze-to-screen mapping stage.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from app.vision.camera import Camera
from app.vision.face_landmarks import FaceLandmarksDetector
from app.vision.eye_features import EyeFeatureExtractor


# ============================================================
# Configuration
# ============================================================

CALIBRATION_FILE = "data/gaze_calibration.json"

WINDOW_NAME = "EyeControl AI - Gaze Calibration"

# Number of valid frames collected for every calibration point.
SAMPLES_PER_POINT = 30

# Ignore the first frames after starting/after changing target.
WARMUP_FRAMES = 15

# Time to wait after changing the target before collection.
TARGET_SETTLE_TIME = 0.8

# Small delay between accepted samples.
SAMPLE_INTERVAL = 0.03

# Target radius.
TARGET_RADIUS = 18

# Target colors.
BACKGROUND_COLOR = (25, 25, 25)
TARGET_COLOR = (0, 220, 255)
TEXT_COLOR = (255, 255, 255)
SECONDARY_TEXT_COLOR = (180, 180, 180)


# ============================================================
# Calibration points
# ============================================================

CALIBRATION_POINTS = [
    ("CENTER", 0.50, 0.50),
    ("LEFT", 0.10, 0.50),
    ("RIGHT", 0.90, 0.50),
    ("UP", 0.50, 0.10),
    ("DOWN", 0.50, 0.90),
]


# ============================================================
# Feature definition
# ============================================================

FEATURE_NAMES = [
    "left_normalized_x",
    "left_normalized_y",
    "right_normalized_x",
    "right_normalized_y",
    "left_ear",
    "right_ear",
]


@dataclass
class CalibrationPoint:
    name: str

    screen_x: int
    screen_y: int

    sample_count: int

    left_normalized_x: float
    left_normalized_y: float

    right_normalized_x: float
    right_normalized_y: float

    left_ear: float
    right_ear: float


# ============================================================
# Helpers
# ============================================================

def features_to_vector(features) -> Optional[np.ndarray]:
    """
    Convert EyeFeatures into the exact six features used by
    the Phase 3 Random Forest model.
    """

    if features is None:
        return None

    values = np.array(
        [
            features.left_normalized_x,
            features.left_normalized_y,
            features.right_normalized_x,
            features.right_normalized_y,
            features.left_ear,
            features.right_ear,
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(values)):
        return None

    return values


def calculate_average(samples: List[np.ndarray]) -> np.ndarray:
    """
    Calculate the average feature vector.
    """

    data = np.vstack(samples)

    # Median first helps reduce occasional landmark outliers.
    median = np.median(data, axis=0)

    distances = np.linalg.norm(data - median, axis=1)

    # Keep the closest 80% of samples to the median.
    keep_count = max(1, int(len(samples) * 0.80))

    keep_indices = np.argsort(distances)[:keep_count]

    filtered = data[keep_indices]

    return np.mean(filtered, axis=0)


def save_calibration(
    points: List[CalibrationPoint],
    screen_width: int,
    screen_height: int,
) -> None:

    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)

    payload = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "screen_width": screen_width,
        "screen_height": screen_height,
        "samples_per_point": SAMPLES_PER_POINT,
        "feature_columns": FEATURE_NAMES,
        "points": {
            point.name: asdict(point)
            for point in points
        },
    }

    with open(CALIBRATION_FILE, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=4,
        )


def draw_target(
    frame,
    point_name: str,
    target_x: int,
    target_y: int,
    current_sample: int,
    total_samples: int,
    message: str = "",
):
    """
    Draw the calibration target and instructions.
    """

    height, width = frame.shape[:2]

    frame[:] = BACKGROUND_COLOR

    # Target outer ring.
    cv2.circle(
        frame,
        (target_x, target_y),
        TARGET_RADIUS + 8,
        (100, 100, 100),
        2,
    )

    # Target.
    cv2.circle(
        frame,
        (target_x, target_y),
        TARGET_RADIUS,
        TARGET_COLOR,
        -1,
    )

    # Center.
    cv2.circle(
        frame,
        (target_x, target_y),
        5,
        (255, 255, 255),
        -1,
    )

    title = f"EyeControl AI - Calibration"

    cv2.putText(
        frame,
        title,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    instruction = f"Look at: {point_name}"

    text_size = cv2.getTextSize(
        instruction,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        2,
    )[0]

    cv2.putText(
        frame,
        instruction,
        (
            max(20, (width - text_size[0]) // 2),
            height - 100,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    progress = f"Samples: {current_sample}/{total_samples}"

    cv2.putText(
        frame,
        progress,
        (30, height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        SECONDARY_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    if message:
        msg_size = cv2.getTextSize(
            message,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2,
        )[0]

        cv2.putText(
            frame,
            message,
            (
                max(20, (width - msg_size[0]) // 2),
                80,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            SECONDARY_TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )


# ============================================================
# Main calibration
# ============================================================

def run_calibration() -> bool:

    print("=" * 70)
    print("EyeControl AI - GAZE CALIBRATION")
    print("=" * 70)

    print()
    print("Calibration points:")
    print("1. CENTER")
    print("2. LEFT")
    print("3. RIGHT")
    print("4. UP")
    print("5. DOWN")
    print()
    print("Instructions:")
    print("- Look directly at the yellow target.")
    print("- Keep your head as still as possible.")
    print("- Press SPACE when you are ready.")
    print("- The program will collect samples automatically.")
    print("- Press ESC at any time to cancel.")
    print()

    camera = Camera()
    detector = FaceLandmarksDetector()
    extractor = EyeFeatureExtractor()

    if not camera.is_opened():
        print("ERROR: Camera could not be opened.")
        detector.close()
        return False

    # --------------------------------------------------------
    # Determine screen size
    # --------------------------------------------------------

    screen_width = 1920
    screen_height = 1080

    try:
        screen_width = int(
            cv2.getWindowImageRect(WINDOW_NAME)[2]
        )
    except Exception:
        pass

    # Create initial calibration window.
    calibration_frame = np.zeros(
        (1080, 1920, 3),
        dtype=np.uint8,
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    # --------------------------------------------------------
    # Use a known desktop resolution from the fullscreen frame.
    # --------------------------------------------------------

    screen_height, screen_width = calibration_frame.shape[:2]

    print(
        f"Calibration resolution: "
        f"{screen_width} x {screen_height}"
    )

    results: List[CalibrationPoint] = []

    try:

        # ----------------------------------------------------
        # Calibration loop
        # ----------------------------------------------------

        for index, (point_name, relative_x, relative_y) in enumerate(
            CALIBRATION_POINTS,
            start=1,
        ):

            target_x = int(screen_width * relative_x)
            target_y = int(screen_height * relative_y)

            print()
            print(
                f"[{index}/5] Preparing {point_name}..."
            )

            # ------------------------------------------------
            # Warmup
            # ------------------------------------------------

            warmup_start = time.time()

            while time.time() - warmup_start < TARGET_SETTLE_TIME:

                ok, camera_frame = camera.read_frame()

                if not ok or camera_frame is None:
                    continue

                display = np.zeros_like(calibration_frame)

                draw_target(
                    display,
                    point_name,
                    target_x,
                    target_y,
                    0,
                    SAMPLES_PER_POINT,
                    "Keep your head still...",
                )

                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    print("Calibration cancelled.")
                    return False

            # ------------------------------------------------
            # Wait for SPACE
            # ------------------------------------------------

            while True:

                ok, camera_frame = camera.read_frame()

                if not ok or camera_frame is None:
                    continue

                display = np.zeros_like(calibration_frame)

                draw_target(
                    display,
                    point_name,
                    target_x,
                    target_y,
                    0,
                    SAMPLES_PER_POINT,
                    "Look at the target, then press SPACE",
                )

                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    print("Calibration cancelled.")
                    return False

                if key == 32:
                    break

            # ------------------------------------------------
            # Collect samples
            # ------------------------------------------------

            print(
                f"Collecting {SAMPLES_PER_POINT} samples..."
            )

            samples: List[np.ndarray] = []

            warmup_count = 0

            last_sample_time = 0.0

            while len(samples) < SAMPLES_PER_POINT:

                ok, camera_frame = camera.read_frame()

                if not ok or camera_frame is None:
                    continue

                landmarks = detector.process_frame(
                    camera_frame
                )

                features = extractor.extract_features(
                    landmarks
                )

                vector = features_to_vector(features)

                # Ignore invalid frames.
                if vector is None:
                    continue

                # Give MediaPipe a few frames to settle.
                if warmup_count < WARMUP_FRAMES:
                    warmup_count += 1

                    display = np.zeros_like(calibration_frame)

                    draw_target(
                        display,
                        point_name,
                        target_x,
                        target_y,
                        len(samples),
                        SAMPLES_PER_POINT,
                        "Stabilizing...",
                    )

                    cv2.imshow(WINDOW_NAME, display)

                    key = cv2.waitKey(1) & 0xFF

                    if key == 27:
                        print("Calibration cancelled.")
                        return False

                    continue

                now = time.time()

                if now - last_sample_time < SAMPLE_INTERVAL:
                    cv2.imshow(
                        WINDOW_NAME,
                        np.zeros_like(calibration_frame),
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == 27:
                        print("Calibration cancelled.")
                        return False

                    continue

                last_sample_time = now

                samples.append(vector)

                display = np.zeros_like(calibration_frame)

                draw_target(
                    display,
                    point_name,
                    target_x,
                    target_y,
                    len(samples),
                    SAMPLES_PER_POINT,
                    "Collecting...",
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    print("Calibration cancelled.")
                    return False

            # ------------------------------------------------
            # Average samples
            # ------------------------------------------------

            average = calculate_average(samples)

            point = CalibrationPoint(
                name=point_name,
                screen_x=target_x,
                screen_y=target_y,
                sample_count=len(samples),

                left_normalized_x=float(average[0]),
                left_normalized_y=float(average[1]),

                right_normalized_x=float(average[2]),
                right_normalized_y=float(average[3]),

                left_ear=float(average[4]),
                right_ear=float(average[5]),
            )

            results.append(point)

            print(
                f"{point_name}: "
                f"LX={point.left_normalized_x:.4f}, "
                f"LY={point.left_normalized_y:.4f}, "
                f"RX={point.right_normalized_x:.4f}, "
                f"RY={point.right_normalized_y:.4f}, "
                f"LEAR={point.left_ear:.4f}, "
                f"REAR={point.right_ear:.4f}"
            )

            # ------------------------------------------------
            # Short confirmation
            # ------------------------------------------------

            confirmation_start = time.time()

            while time.time() - confirmation_start < 0.6:

                display = np.zeros_like(calibration_frame)

                draw_target(
                    display,
                    point_name,
                    target_x,
                    target_y,
                    SAMPLES_PER_POINT,
                    SAMPLES_PER_POINT,
                    "Point recorded",
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    print("Calibration cancelled.")
                    return False

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_calibration(
            results,
            screen_width,
            screen_height,
        )

        print()
        print("=" * 70)
        print("CALIBRATION COMPLETED")
        print("=" * 70)
        print()
        print(
            f"Saved to:\n{os.path.abspath(CALIBRATION_FILE)}"
        )
        print()

        for point in results:
            print(
                f"{point.name:>6} | "
                f"screen=({point.screen_x}, {point.screen_y}) | "
                f"LX={point.left_normalized_x:.4f} "
                f"LY={point.left_normalized_y:.4f} | "
                f"RX={point.right_normalized_x:.4f} "
                f"RY={point.right_normalized_y:.4f}"
            )

        print()
        print("Press any key to close.")

        while True:

            display = np.zeros_like(calibration_frame)

            cv2.putText(
                display,
                "Calibration completed successfully",
                (250, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "Press any key to close",
                (560, 540),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                SECONDARY_TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            key = cv2.waitKey(30) & 0xFF

            if key != 255:
                break

        return True

    finally:

        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_calibration()