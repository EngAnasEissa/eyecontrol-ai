import csv
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
import joblib
import pyautogui

from app.vision.camera import Camera
from app.vision.face_landmarks import FaceLandmarksDetector
from app.vision.eye_features import EyeFeatureExtractor, EyeFeatures


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "phase3_baseline_random_forest.joblib"
)

VALIDATION_CSV = PROJECT_ROOT / "phase2_validation.csv"


# ============================================================
# Phase 2 validation settings
# ============================================================

VALIDATION_RECORD_SECONDS = 3.0

VALIDATION_STATES = [
    "FORWARD",
    "LEFT",
    "FORWARD",
    "RIGHT",
    "FORWARD",
    "UP",
    "FORWARD",
    "DOWN",
    "FORWARD",
    "EYES_OPEN",
    "EYES_CLOSED",
]

MANUAL_VALIDATION_KEYS = {
    ord("1"): "FORWARD",
    ord("2"): "LEFT",
    ord("3"): "RIGHT",
    ord("4"): "UP",
    ord("5"): "DOWN",
    ord("6"): "EYES_OPEN",
    ord("8"): "EYES_CLOSED",
}


# ============================================================
# Mouse control
# ============================================================

mouse_control_enabled = False

# ------------------------------------------------------------
# CENTER / neutral states
# ------------------------------------------------------------

# The current classifier does not produce a continuous
# gaze coordinate. Therefore these states represent
# the current central / neutral region.
MOUSE_DEAD_ZONE_STATES = {
    "FORWARD",
    "EYES_OPEN",
    "EYES_CLOSED",
    "NO_FACE",
}

MOUSE_CENTER_STATE = "CENTER"

DIRECTIONAL_STATES = {
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
}


# ============================================================
# Mouse controller timing
# ============================================================

# Faster control loop than the old 0.08 second interval.
# The movement itself is still small and controlled.
MOUSE_MOVE_INTERVAL = 0.02

last_mouse_move_time = 0.0


# ============================================================
# Direction stabilization
# ============================================================

# Number of consecutive predictions required before
# changing the active mouse direction.
DIRECTION_CONFIRMATION_COUNT = 2

current_mouse_direction = MOUSE_CENTER_STATE

candidate_mouse_direction: Optional[str] = None
candidate_direction_count = 0


# ============================================================
# Mouse velocity controller
# ============================================================

# Velocity is measured approximately in pixels/second.

MOUSE_MIN_SPEED = 25.0
MOUSE_MAX_SPEED = 220.0

# How quickly the mouse accelerates while the user
# keeps looking in the same direction.
MOUSE_ACCELERATION = 420.0

# How quickly the mouse slows when the direction becomes
# CENTER / neutral / unavailable.
MOUSE_DECELERATION = 700.0

current_mouse_speed = 0.0

# Time at which the current direction became stable.
direction_start_time: Optional[float] = None

# Fractional movement accumulator.
# This allows smooth sub-pixel accumulation instead of
# throwing away small movements on every frame.
mouse_accumulator_x = 0.0
mouse_accumulator_y = 0.0


# ============================================================
# Safe screen boundary
# ============================================================

# Keep the cursor away from the physical screen corners.
# PyAutoGUI FAILSAFE remains enabled.
EDGE_MARGIN = 20


# ============================================================
# Prediction history
# ============================================================

SMOOTHING_WINDOW_SIZE = 4

prediction_history = deque(
    maxlen=SMOOTHING_WINDOW_SIZE
)


# ============================================================
# Eye Click settings
# ============================================================

eye_click_enabled = False

EYE_CLICK_HOLD_TIME = 0.70

eye_closed_start_time: Optional[float] = None

eye_click_triggered = False


# ============================================================
# Validation state
# ============================================================

validation_state: Optional[str] = None
validation_start_time: Optional[float] = None


# ============================================================
# Validation functions
# ============================================================

def begin_validation(state: str) -> None:
    global validation_state
    global validation_start_time

    validation_state = state
    validation_start_time = time.time()

    print(
        f"[VALIDATION] Started: {state} "
        f"for {VALIDATION_RECORD_SECONDS:.1f} seconds"
    )


def finish_validation() -> None:
    global validation_state
    global validation_start_time

    validation_state = None
    validation_start_time = None


def append_validation_row(
    features: EyeFeatures,
    label: str,
) -> None:

    file_exists = VALIDATION_CSV.exists()

    row = {
        "left_norm_x": features.left_normalized_x,
        "left_norm_y": features.left_normalized_y,
        "right_norm_x": features.right_normalized_x,
        "right_norm_y": features.right_normalized_y,
        "left_ear": features.left_ear,
        "right_ear": features.right_ear,
        "label": label,
    }

    with open(
        VALIDATION_CSV,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Prediction smoothing
# ============================================================

def smooth_prediction(prediction: str) -> str:
    """
    Lightweight prediction smoothing.

    The old implementation used the entire history as a
    majority vote and could keep an old direction active
    longer than desired.

    This implementation mainly removes isolated noisy
    predictions while the dedicated Mouse Controller
    performs the actual direction stabilization.
    """

    prediction_history.append(prediction)

    if prediction == "NO_FACE":
        return "NO_FACE"

    # Neutral states are returned immediately.
    if prediction in MOUSE_DEAD_ZONE_STATES:
        return prediction

    # Directional prediction:
    # accept it if it has recent support.
    recent_count = 0

    for item in prediction_history:
        if item == prediction:
            recent_count += 1

    if recent_count >= 2:
        return prediction

    # If the new directional prediction is isolated,
    # retain the most recent stable directional state.
    for item in reversed(list(prediction_history)[:-1]):

        if item in DIRECTIONAL_STATES:
            return item

    return prediction


# ============================================================
# Mouse controller reset
# ============================================================

def reset_mouse_controller() -> None:
    """
    Completely reset mouse-control state.

    This is called whenever Mouse Control is enabled/disabled
    and prevents old predictions, velocity, or acceleration
    from affecting the new session.
    """

    global current_mouse_direction
    global candidate_mouse_direction
    global candidate_direction_count
    global current_mouse_speed
    global direction_start_time
    global mouse_accumulator_x
    global mouse_accumulator_y
    global last_mouse_move_time

    current_mouse_direction = MOUSE_CENTER_STATE

    candidate_mouse_direction = None
    candidate_direction_count = 0

    current_mouse_speed = 0.0
    direction_start_time = None

    mouse_accumulator_x = 0.0
    mouse_accumulator_y = 0.0

    last_mouse_move_time = time.time()


# ============================================================
# Mouse direction stabilizer
# ============================================================

def update_mouse_direction(
    prediction: str,
    now: float,
) -> str:
    """
    Update the stable mouse direction.

    The controller uses a current direction and a candidate
    direction. A new direction must be confirmed before
    becoming active.

    This prevents:

        RIGHT
        LEFT
        RIGHT
        LEFT

    from directly shaking the mouse.
    """

    global current_mouse_direction
    global candidate_mouse_direction
    global candidate_direction_count
    global direction_start_time
    global current_mouse_speed
    global mouse_accumulator_x
    global mouse_accumulator_y

    # --------------------------------------------------------
    # Convert neutral classifier states to CENTER
    # --------------------------------------------------------

    if prediction in MOUSE_DEAD_ZONE_STATES:
        desired_direction = MOUSE_CENTER_STATE
    elif prediction in DIRECTIONAL_STATES:
        desired_direction = prediction
    else:
        desired_direction = MOUSE_CENTER_STATE

    # --------------------------------------------------------
    # Same direction as current
    # --------------------------------------------------------

    if desired_direction == current_mouse_direction:

        candidate_mouse_direction = None
        candidate_direction_count = 0

        return current_mouse_direction

    # --------------------------------------------------------
    # Candidate direction changed
    # --------------------------------------------------------

    if desired_direction != candidate_mouse_direction:

        candidate_mouse_direction = desired_direction
        candidate_direction_count = 1

        return current_mouse_direction

    # --------------------------------------------------------
    # Candidate direction continues
    # --------------------------------------------------------

    candidate_direction_count += 1

    # --------------------------------------------------------
    # Confirm new direction
    # --------------------------------------------------------

    if candidate_direction_count >= DIRECTION_CONFIRMATION_COUNT:

        previous_direction = current_mouse_direction

        current_mouse_direction = desired_direction

        candidate_mouse_direction = None
        candidate_direction_count = 0

        direction_start_time = now

        # When changing direction, do not carry the previous
        # velocity into the new direction.
        current_mouse_speed = 0.0

        # Prevent old fractional movement from causing
        # an unexpected jump.
        mouse_accumulator_x = 0.0
        mouse_accumulator_y = 0.0

        print(
            "[MOUSE DIRECTION] "
            f"{previous_direction} -> "
            f"{current_mouse_direction}"
        )

    return current_mouse_direction


# ============================================================
# Mouse velocity update
# ============================================================

def update_mouse_velocity(
    direction: str,
    now: float,
) -> float:
    """
    Calculate smooth mouse velocity.

    Directional states accelerate gradually.

    CENTER / neutral states decelerate smoothly toward zero.
    """

    global current_mouse_speed
    global direction_start_time

    # --------------------------------------------------------
    # CENTER / no movement
    # --------------------------------------------------------

    if direction == MOUSE_CENTER_STATE:

        current_mouse_speed = max(
            0.0,
            current_mouse_speed
            - MOUSE_DECELERATION * MOUSE_MOVE_INTERVAL,
        )

        return current_mouse_speed

    # --------------------------------------------------------
    # Directional movement
    # --------------------------------------------------------

    if direction in DIRECTIONAL_STATES:

        if direction_start_time is None:
            direction_start_time = now

        held_time = max(
            0.0,
            now - direction_start_time,
        )

        # Gradually increase target speed based on how long
        # the user maintains the direction.
        target_speed = (
            MOUSE_MIN_SPEED
            + MOUSE_ACCELERATION * held_time
        )

        target_speed = min(
            MOUSE_MAX_SPEED,
            target_speed,
        )

        # Smoothly approach target speed.
        acceleration_step = (
            MOUSE_ACCELERATION
            * MOUSE_MOVE_INTERVAL
        )

        if current_mouse_speed < target_speed:

            current_mouse_speed = min(
                target_speed,
                current_mouse_speed
                + acceleration_step,
            )

        else:

            current_mouse_speed = max(
                target_speed,
                current_mouse_speed
                - acceleration_step,
            )

        return current_mouse_speed

    return 0.0


# ============================================================
# Safe mouse movement
# ============================================================

def move_mouse_safely(
    direction: str,
    speed: float,
    elapsed_time: float,
) -> None:
    """
    Move the mouse while respecting screen boundaries.

    PyAutoGUI FAILSAFE remains enabled.
    """

    global mouse_accumulator_x
    global mouse_accumulator_y

    if direction not in DIRECTIONAL_STATES:
        return

    if speed <= 0.0:
        return

    if elapsed_time <= 0.0:
        return

    # --------------------------------------------------------
    # Convert velocity into movement for this time interval.
    # --------------------------------------------------------

    movement = speed * elapsed_time

    if direction == "LEFT":
        mouse_accumulator_x -= movement

    elif direction == "RIGHT":
        mouse_accumulator_x += movement

    elif direction == "UP":
        mouse_accumulator_y -= movement

    elif direction == "DOWN":
        mouse_accumulator_y += movement

    # --------------------------------------------------------
    # Convert accumulated fractional movement to pixels.
    # --------------------------------------------------------

    move_x = int(mouse_accumulator_x)
    move_y = int(mouse_accumulator_y)

    mouse_accumulator_x -= move_x
    mouse_accumulator_y -= move_y

    if move_x == 0 and move_y == 0:
        return

    # --------------------------------------------------------
    # Screen size
    # --------------------------------------------------------

    screen_width, screen_height = pyautogui.size()

    min_x = EDGE_MARGIN
    max_x = screen_width - EDGE_MARGIN - 1

    min_y = EDGE_MARGIN
    max_y = screen_height - EDGE_MARGIN - 1

    # --------------------------------------------------------
    # Current cursor position
    # --------------------------------------------------------

    current_x, current_y = pyautogui.position()

    target_x = current_x + move_x
    target_y = current_y + move_y

    # --------------------------------------------------------
    # Clamp target to safe screen region
    # --------------------------------------------------------

    safe_target_x = max(
        min_x,
        min(
            max_x,
            target_x,
        ),
    )

    safe_target_y = max(
        min_y,
        min(
            max_y,
            target_y,
        ),
    )

    safe_dx = safe_target_x - current_x
    safe_dy = safe_target_y - current_y

    # --------------------------------------------------------
    # If the cursor reached a boundary, discard accumulated
    # movement toward that boundary.
    # --------------------------------------------------------

    if safe_dx == 0:

        if move_x < 0:
            mouse_accumulator_x = 0.0

        elif move_x > 0:
            mouse_accumulator_x = 0.0

    if safe_dy == 0:

        if move_y < 0:
            mouse_accumulator_y = 0.0

        elif move_y > 0:
            mouse_accumulator_y = 0.0

    # --------------------------------------------------------
    # Perform movement
    # --------------------------------------------------------

    if safe_dx != 0 or safe_dy != 0:

        pyautogui.moveRel(
            safe_dx,
            safe_dy,
            duration=0,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    global validation_state
    global validation_start_time

    global mouse_control_enabled
    global last_mouse_move_time

    global eye_click_enabled
    global eye_closed_start_time
    global eye_click_triggered

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    model_bundle = joblib.load(
        MODEL_PATH
    )

    if not isinstance(model_bundle, dict):

        raise TypeError(
            "Invalid model file format. "
            "Expected a dictionary/model bundle."
        )

    required_keys = [
        "model",
        "feature_columns",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in model_bundle
    ]

    if missing_keys:

        raise KeyError(
            "Model bundle is missing required keys: "
            f"{missing_keys}"
        )

    model = model_bundle["model"]

    expected_features = list(
        model_bundle["feature_columns"]
    )

    if not hasattr(model, "predict"):

        raise TypeError(
            "The 'model' entry inside the model bundle "
            "does not provide a predict() method."
        )

    expected_feature_set = {
        "left_norm_x",
        "left_norm_y",
        "right_norm_x",
        "right_norm_y",
        "left_ear",
        "right_ear",
    }

    if set(expected_features) != expected_feature_set:

        raise ValueError(
            "Unexpected model feature columns.\n"
            f"Expected: {expected_feature_set}\n"
            f"Model:    {set(expected_features)}"
        )

    if len(expected_features) != 6:

        raise ValueError(
            "Expected exactly 6 model features, "
            f"but found {len(expected_features)}."
        )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print("=" * 60)
    print("EyeControl AI")
    print("=" * 60)

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Model type: {type(model).__name__}"
    )

    print(
        f"Feature columns: {expected_features}"
    )

    if "classes" in model_bundle:

        print(
            f"Classes: {model_bundle['classes']}"
        )

    if "random_state" in model_bundle:

        print(
            f"Random state: "
            f"{model_bundle['random_state']}"
        )

    if "n_estimators" in model_bundle:

        print(
            f"N estimators: "
            f"{model_bundle['n_estimators']}"
        )

    print()

    print("Controls:")
    print("M     = Toggle Mouse Control")
    print("C     = Toggle Eye Click")
    print("1     = Validation FORWARD")
    print("2     = Validation LEFT")
    print("3     = Validation RIGHT")
    print("4     = Validation UP")
    print("5     = Validation DOWN")
    print("6     = Validation EYES_OPEN")
    print("8     = Validation EYES_CLOSED")
    print("Q/ESC = Exit")

    print()

    print("Mouse controller:")
    print(
        f"Direction confirmation: "
        f"{DIRECTION_CONFIRMATION_COUNT}"
    )

    print(
        f"Movement interval: "
        f"{MOUSE_MOVE_INTERVAL:.3f}s"
    )

    print(
        f"Speed range: "
        f"{MOUSE_MIN_SPEED:.0f}-"
        f"{MOUSE_MAX_SPEED:.0f} px/s"
    )

    print(
        "Center state: FORWARD / neutral"
    )

    print(
        f"Edge margin: {EDGE_MARGIN}px"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Vision components
    # --------------------------------------------------------

    camera = Camera()
    detector = FaceLandmarksDetector()
    extractor = EyeFeatureExtractor()

    if not camera.is_opened():

        raise RuntimeError(
            "Camera could not be opened."
        )

    try:

        while True:

            # =================================================
            # Read camera frame
            # =================================================

            ret, frame = camera.read_frame()

            if not ret or frame is None:
                continue

            # =================================================
            # Detect face landmarks
            # =================================================

            landmarks = detector.process_frame(
                frame
            )

            prediction_text = "NO_FACE"
            smoothed_prediction = "NO_FACE"

            features: Optional[EyeFeatures] = None

            # =================================================
            # Extract eye features
            # =================================================

            if landmarks is not None:

                features = extractor.extract_features(
                    landmarks
                )

                if features is not None:

                    # =========================================
                    # Prepare model input
                    # =========================================

                    feature_values = {
                        "left_norm_x":
                            features.left_normalized_x,

                        "left_norm_y":
                            features.left_normalized_y,

                        "right_norm_x":
                            features.right_normalized_x,

                        "right_norm_y":
                            features.right_normalized_y,

                        "left_ear":
                            features.left_ear,

                        "right_ear":
                            features.right_ear,
                    }

                    feature_row = pd.DataFrame(
                        [[
                            feature_values[column]
                            for column in expected_features
                        ]],
                        columns=expected_features,
                    )

                    # =========================================
                    # Random Forest prediction
                    # =========================================

                    prediction = model.predict(
                        feature_row
                    )[0]

                    prediction_text = str(
                        prediction
                    )

                    # =========================================
                    # Prediction smoothing
                    # =========================================

                    smoothed_prediction = (
                        smooth_prediction(
                            prediction_text
                        )
                    )

                    now = time.time()

                    # =========================================
                    # Mouse direction controller
                    # =========================================

                    if mouse_control_enabled:

                        stable_direction = (
                            update_mouse_direction(
                                smoothed_prediction,
                                now,
                            )
                        )

                        mouse_speed = (
                            update_mouse_velocity(
                                stable_direction,
                                now,
                            )
                        )

                        if (
                            now
                            - last_mouse_move_time
                            >= MOUSE_MOVE_INTERVAL
                        ):

                            movement_elapsed = (
                                now
                                - last_mouse_move_time
                            )

                            # Limit abnormal time gaps.
                            # This prevents a large cursor jump
                            # if the application was temporarily
                            # paused.
                            movement_elapsed = min(
                                movement_elapsed,
                                0.10,
                            )

                            move_mouse_safely(
                                stable_direction,
                                mouse_speed,
                                movement_elapsed,
                            )

                            last_mouse_move_time = now

                    # =========================================
                    # Eye Click
                    # =========================================

                    if eye_click_enabled:

                        if (
                            smoothed_prediction
                            == "EYES_CLOSED"
                        ):

                            if (
                                eye_closed_start_time
                                is None
                            ):

                                eye_closed_start_time = now

                            closed_duration = (
                                now
                                - eye_closed_start_time
                            )

                            if (
                                closed_duration
                                >= EYE_CLICK_HOLD_TIME
                                and not eye_click_triggered
                            ):

                                pyautogui.click()

                                eye_click_triggered = True

                                print(
                                    "[EYE CLICK] "
                                    "Left click triggered."
                                )

                        else:

                            eye_closed_start_time = None
                            eye_click_triggered = False

                    else:

                        eye_closed_start_time = None
                        eye_click_triggered = False

                    # =========================================
                    # Phase 2 validation recording
                    # =========================================

                    if (
                        validation_state is not None
                        and validation_start_time is not None
                    ):

                        elapsed = (
                            now
                            - validation_start_time
                        )

                        if (
                            elapsed
                            <= VALIDATION_RECORD_SECONDS
                        ):

                            append_validation_row(
                                features,
                                validation_state,
                            )

                        else:

                            print(
                                "[VALIDATION] "
                                f"Finished: "
                                f"{validation_state}"
                            )

                            finish_validation()

            else:

                # =================================================
                # No face detected
                # =================================================

                prediction_history.append(
                    "NO_FACE"
                )

                if mouse_control_enabled:

                    now = time.time()

                    stable_direction = (
                        update_mouse_direction(
                            "NO_FACE",
                            now,
                        )
                    )

                    mouse_speed = (
                        update_mouse_velocity(
                            stable_direction,
                            now,
                        )
                    )

                    if (
                        now
                        - last_mouse_move_time
                        >= MOUSE_MOVE_INTERVAL
                    ):

                        last_mouse_move_time = now

            # =================================================
            # HUD
            # =================================================

            cv2.putText(
                frame,
                f"Prediction: {prediction_text}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Smoothed: {smoothed_prediction}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            mouse_status = (
                "ON"
                if mouse_control_enabled
                else "OFF"
            )

            eye_click_status = (
                "ON"
                if eye_click_enabled
                else "OFF"
            )

            cv2.putText(
                frame,
                f"MOUSE: {mouse_status}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"EYE CLICK: {eye_click_status}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # -------------------------------------------------
            # Display actual mouse-controller direction
            # -------------------------------------------------

            if mouse_control_enabled:

                controller_direction = (
                    current_mouse_direction
                )

                controller_speed = (
                    current_mouse_speed
                )

            else:

                controller_direction = (
                    MOUSE_CENTER_STATE
                )

                controller_speed = 0.0

            cv2.putText(
                frame,
                f"CONTROLLER: "
                f"{controller_direction}",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 180, 0),
                2,
            )

            cv2.putText(
                frame,
                f"SPEED: "
                f"{controller_speed:.0f} px/s",
                (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 180, 0),
                2,
            )

            cv2.putText(
                frame,
                "M:Mouse  C:EyeClick  Q/ESC:Exit",
                (20, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "1:F  2:L  3:R  4:U  5:D  6:OPEN  8:CLOSED",
                (20, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                2,
            )

            # -------------------------------------------------
            # Validation status
            # -------------------------------------------------

            if validation_state is not None:

                cv2.putText(
                    frame,
                    f"VALIDATING: "
                    f"{validation_state}",
                    (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

            # =================================================
            # Show frame
            # =================================================

            cv2.imshow(
                "EyeControl AI",
                frame,
            )

            # =================================================
            # Keyboard
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            # -------------------------------------------------
            # Exit
            # -------------------------------------------------

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            # -------------------------------------------------
            # Toggle Mouse
            # -------------------------------------------------

            if key in (
                ord("m"),
                ord("M"),
            ):

                mouse_control_enabled = (
                    not mouse_control_enabled
                )

                # Completely clear old predictions.
                prediction_history.clear()

                # Completely reset the mouse controller.
                reset_mouse_controller()

                print(
                    "[MOUSE] "
                    f"{'ON' if mouse_control_enabled else 'OFF'}"
                )

                if mouse_control_enabled:

                    print(
                        "[MOUSE] "
                        "Controller reset. "
                        "Waiting for stable direction..."
                    )

                else:

                    print(
                        "[MOUSE] "
                        "Controller stopped."
                    )

            # -------------------------------------------------
            # Toggle Eye Click
            # -------------------------------------------------

            if key in (
                ord("c"),
                ord("C"),
            ):

                eye_click_enabled = (
                    not eye_click_enabled
                )

                eye_closed_start_time = None
                eye_click_triggered = False

                print(
                    "[EYE CLICK] "
                    f"{'ON' if eye_click_enabled else 'OFF'}"
                )

            # -------------------------------------------------
            # Manual validation
            # -------------------------------------------------

            if key in MANUAL_VALIDATION_KEYS:

                begin_validation(
                    MANUAL_VALIDATION_KEYS[key]
                )

    finally:

        camera.release()

        detector.close()

        cv2.destroyAllWindows()

        print("EyeControl AI stopped.")


if __name__ == "__main__":
    main()