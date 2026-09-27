"""
EyeControl AI - Gaze Coordinate Test
Phase 5.2

IMPORTANT:
This test NEVER moves the mouse.

It only converts the current eye features into
estimated screen coordinates and displays them.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque

import cv2
import numpy as np

from app.vision.camera import Camera
from app.vision.face_landmarks import FaceLandmarksDetector
from app.vision.eye_features import EyeFeatureExtractor


# ============================================================
# Configuration
# ============================================================

CALIBRATION_FILE = "data/gaze_calibration.json"

WINDOW_NAME = "EyeControl AI - Gaze Coordinate Test"

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Number of recent coordinate samples used for display smoothing.
COORDINATE_SMOOTHING_SIZE = 5

# Minimum movement around CENTER.
# This is only for coordinate testing.
CENTER_DEAD_ZONE_X = 0.025
CENTER_DEAD_ZONE_Y = 0.025

# Maximum coordinate change per frame.
# Prevents sudden jumps caused by a bad landmark frame.
MAX_COORDINATE_STEP = 80


# ============================================================
# Load calibration
# ============================================================

def load_calibration():

    if not os.path.exists(CALIBRATION_FILE):
        raise FileNotFoundError(
            f"Calibration file not found:\n{CALIBRATION_FILE}"
        )

    with open(
        CALIBRATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data


# ============================================================
# Feature extraction
# ============================================================

def get_gaze_vector(features):

    if features is None:
        return None

    values = np.array(
        [
            features.left_normalized_x,
            features.left_normalized_y,
            features.right_normalized_x,
            features.right_normalized_y,
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(values)):
        return None

    return values


# ============================================================
# Calibration vectors
# ============================================================

def get_calibration_vector(calibration, name):

    point = calibration["points"][name]

    return np.array(
        [
            point["left_normalized_x"],
            point["left_normalized_y"],
            point["right_normalized_x"],
            point["right_normalized_y"],
        ],
        dtype=np.float64,
    )


# ============================================================
# Calculate average gaze coordinates
# ============================================================

def calculate_gaze_axes(
    gaze,
    center,
    left,
    right,
    up,
    down,
):
    """
    Convert eye features into normalized X/Y coordinates.

    X uses LEFT <-> CENTER <-> RIGHT.
    Y uses UP <-> CENTER <-> DOWN.

    We intentionally do NOT use EAR.
    """

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    x_distance_left = np.linalg.norm(
        gaze[[0, 2]] - left[[0, 2]]
    )

    x_distance_right = np.linalg.norm(
        gaze[[0, 2]] - right[[0, 2]]
    )

    center_x = np.mean(
        [
            center[0],
            center[2],
        ]
    )

    gaze_x = np.mean(
        [
            gaze[0],
            gaze[2],
        ]
    )

    left_x = np.mean(
        [
            left[0],
            left[2],
        ]
    )

    right_x = np.mean(
        [
            right[0],
            right[2],
        ]
    )

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    gaze_y = np.mean(
        [
            gaze[1],
            gaze[3],
        ]
    )

    center_y = np.mean(
        [
            center[1],
            center[3],
        ]
    )

    up_y = np.mean(
        [
            up[1],
            up[3],
        ]
    )

    down_y = np.mean(
        [
            down[1],
            down[3],
        ]
    )

    # --------------------------------------------------------
    # X interpolation
    # --------------------------------------------------------

    if gaze_x < center_x:

        denominator = center_x - left_x

        if abs(denominator) < 1e-6:
            normalized_x = 0.5
        else:
            normalized_x = 0.5 * (
                (gaze_x - left_x)
                / denominator
            )

    else:

        denominator = right_x - center_x

        if abs(denominator) < 1e-6:
            normalized_x = 0.5
        else:
            normalized_x = 0.5 + 0.5 * (
                (gaze_x - center_x)
                / denominator
            )

    # --------------------------------------------------------
    # Y interpolation
    # --------------------------------------------------------

    if gaze_y < center_y:

        denominator = center_y - up_y

        if abs(denominator) < 1e-6:
            normalized_y = 0.5
        else:
            normalized_y = 0.5 * (
                (gaze_y - up_y)
                / denominator
            )

    else:

        denominator = down_y - center_y

        if abs(denominator) < 1e-6:
            normalized_y = 0.5
        else:
            normalized_y = 0.5 + 0.5 * (
                (gaze_y - center_y)
                / denominator
            )

    # --------------------------------------------------------
    # Clamp
    # --------------------------------------------------------

    normalized_x = float(
        np.clip(normalized_x, 0.0, 1.0)
    )

    normalized_y = float(
        np.clip(normalized_y, 0.0, 1.0)
    )

    # --------------------------------------------------------
    # CENTER dead zone
    # --------------------------------------------------------

    center_delta_x = abs(
        gaze_x - center_x
    )

    center_delta_y = abs(
        gaze_y - center_y
    )

    if (
        center_delta_x < CENTER_DEAD_ZONE_X
        and center_delta_y < CENTER_DEAD_ZONE_Y
    ):
        normalized_x = 0.5
        normalized_y = 0.5

    return normalized_x, normalized_y


# ============================================================
# Safe coordinate smoothing
# ============================================================

def smooth_coordinate(
    history,
    value,
    max_step,
):

    if not history:
        history.append(value)
        return value

    previous = history[-1]

    delta = value - previous

    if abs(delta) > max_step:
        value = previous + np.sign(delta) * max_step

    history.append(value)

    return float(
        np.mean(history)
    )


# ============================================================
# Drawing
# ============================================================

def draw_screen_preview(
    width,
    height,
    screen_x,
    screen_y,
    gaze_x,
    gaze_y,
):

    preview_width = 960
    preview_height = 540

    preview = np.zeros(
        (
            preview_height,
            preview_width,
            3,
        ),
        dtype=np.uint8,
    )

    # Border
    cv2.rectangle(
        preview,
        (2, 2),
        (
            preview_width - 3,
            preview_height - 3,
        ),
        (100, 100, 100),
        2,
    )

    # Convert actual screen coordinate to preview coordinate.
    px = int(
        screen_x
        / width
        * preview_width
    )

    py = int(
        screen_y
        / height
        * preview_height
    )

    px = max(
        0,
        min(preview_width - 1, px),
    )

    py = max(
        0,
        min(preview_height - 1, py),
    )

    # Center reference.
    center_px = preview_width // 2
    center_py = preview_height // 2

    cv2.circle(
        preview,
        (
            center_px,
            center_py,
        ),
        35,
        (80, 80, 80),
        1,
    )

    # Gaze point.
    cv2.circle(
        preview,
        (px, py),
        12,
        (0, 220, 255),
        -1,
    )

    cv2.line(
        preview,
        (
            center_px,
            center_py,
        ),
        (px, py),
        (120, 120, 120),
        1,
    )

    cv2.putText(
        preview,
        f"X: {screen_x:4d}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        f"Y: {screen_y:4d}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        f"Gaze X: {gaze_x:.3f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        f"Gaze Y: {gaze_y:.3f}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return preview


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("EyeControl AI - GAZE COORDINATE TEST")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print("The mouse WILL NOT move.")
    print("This program only displays estimated coordinates.")
    print()

    calibration = load_calibration()

    global SCREEN_WIDTH
    global SCREEN_HEIGHT

    SCREEN_WIDTH = int(
        calibration.get(
            "screen_width",
            1920,
        )
    )

    SCREEN_HEIGHT = int(
        calibration.get(
            "screen_height",
            1080,
        )
    )

    center = get_calibration_vector(
        calibration,
        "CENTER",
    )

    left = get_calibration_vector(
        calibration,
        "LEFT",
    )

    right = get_calibration_vector(
        calibration,
        "RIGHT",
    )

    up = get_calibration_vector(
        calibration,
        "UP",
    )

    down = get_calibration_vector(
        calibration,
        "DOWN",
    )

    print(
        f"Screen: {SCREEN_WIDTH} x {SCREEN_HEIGHT}"
    )

    print()
    print("Calibration references loaded.")
    print()

    camera = Camera()
    detector = FaceLandmarksDetector()
    extractor = EyeFeatureExtractor()

    if not camera.is_opened():
        print("ERROR: Camera could not be opened.")
        detector.close()
        return

    x_history = deque(
        maxlen=COORDINATE_SMOOTHING_SIZE
    )

    y_history = deque(
        maxlen=COORDINATE_SMOOTHING_SIZE
    )

    previous_screen_x = None
    previous_screen_y = None

    last_print_time = 0.0

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    try:

        while True:

            ok, frame = camera.read_frame()

            if not ok or frame is None:
                continue

            landmarks = detector.process_frame(
                frame
            )

            features = extractor.extract_features(
                landmarks
            )

            gaze = get_gaze_vector(
                features
            )

            if gaze is None:

                display = np.zeros(
                    (
                        540,
                        960,
                        3,
                    ),
                    dtype=np.uint8,
                )

                cv2.putText(
                    display,
                    "NO FACE / NO GAZE",
                    (300, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    display,
                    "Press Q or ESC to exit",
                    (330, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    break

                continue

            # ------------------------------------------------
            # Calculate normalized gaze coordinates
            # ------------------------------------------------

            normalized_x, normalized_y = (
                calculate_gaze_axes(
                    gaze,
                    center,
                    left,
                    right,
                    up,
                    down,
                )
            )

            # ------------------------------------------------
            # Convert to screen coordinates
            # ------------------------------------------------

            raw_screen_x = (
                normalized_x
                * (SCREEN_WIDTH - 1)
            )

            raw_screen_y = (
                normalized_y
                * (SCREEN_HEIGHT - 1)
            )

            # ------------------------------------------------
            # Limit sudden jumps
            # ------------------------------------------------

            if previous_screen_x is not None:

                dx = (
                    raw_screen_x
                    - previous_screen_x
                )

                if abs(dx) > MAX_COORDINATE_STEP:

                    raw_screen_x = (
                        previous_screen_x
                        + np.sign(dx)
                        * MAX_COORDINATE_STEP
                    )

            if previous_screen_y is not None:

                dy = (
                    raw_screen_y
                    - previous_screen_y
                )

                if abs(dy) > MAX_COORDINATE_STEP:

                    raw_screen_y = (
                        previous_screen_y
                        + np.sign(dy)
                        * MAX_COORDINATE_STEP
                    )

            previous_screen_x = raw_screen_x
            previous_screen_y = raw_screen_y

            # ------------------------------------------------
            # Smooth coordinates
            # ------------------------------------------------

            screen_x = smooth_coordinate(
                x_history,
                raw_screen_x,
                MAX_COORDINATE_STEP,
            )

            screen_y = smooth_coordinate(
                y_history,
                raw_screen_y,
                MAX_COORDINATE_STEP,
            )

            screen_x_int = int(
                np.clip(
                    screen_x,
                    0,
                    SCREEN_WIDTH - 1,
                )
            )

            screen_y_int = int(
                np.clip(
                    screen_y,
                    0,
                    SCREEN_HEIGHT - 1,
                )
            )

            # ------------------------------------------------
            # Preview
            # ------------------------------------------------

            preview = draw_screen_preview(
                SCREEN_WIDTH,
                SCREEN_HEIGHT,
                screen_x_int,
                screen_y_int,
                normalized_x,
                normalized_y,
            )

            # ------------------------------------------------
            # Camera preview beside coordinate preview
            # ------------------------------------------------

            camera_preview = cv2.resize(
                frame,
                (
                    640,
                    360,
                ),
            )

            combined = np.zeros(
                (
                    620,
                    1600,
                    3,
                ),
                dtype=np.uint8,
            )

            combined[
                40:400,
                20:660
            ] = camera_preview

            combined[
                40:580,
                680:1640
            ] = cv2.resize(
                preview,
                (
                    960,
                    540,
                ),
            )

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            cv2.putText(
                combined,
                "CAMERA",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                combined,
                "SCREEN COORDINATE PREVIEW - MOUSE DISABLED",
                (680, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                combined,
                (
                    f"Screen: "
                    f"({screen_x_int}, {screen_y_int})"
                ),
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                combined,
                (
                    f"Gaze: "
                    f"({normalized_x:.3f}, "
                    f"{normalized_y:.3f})"
                ),
                (20, 485),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                combined,
                "Q / ESC = Exit",
                (20, 530),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                combined,
            )

            # ------------------------------------------------
            # Console diagnostic
            # ------------------------------------------------

            now = time.time()

            if now - last_print_time >= 0.5:

                print(
                    f"Gaze=({normalized_x:.3f}, "
                    f"{normalized_y:.3f}) | "
                    f"Screen=({screen_x_int}, "
                    f"{screen_y_int})"
                )

                last_print_time = now

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

    finally:

        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    print()
    print("Coordinate test finished.")
    print("The mouse was NOT moved.")


if __name__ == "__main__":
    main()