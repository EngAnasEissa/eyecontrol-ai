from pathlib import Path
import joblib
import cv2
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "phase3_baseline_random_forest.joblib"

FEATURE_COLUMNS = [
    "left_norm_x",
    "left_norm_y",
    "right_norm_x",
    "right_norm_y",
    "left_ear",
    "right_ear",
]


def main():
    print("=" * 60)
    print("PHASE 3.8 - LIVE MODEL TEST")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print("[FAIL] Model file not found:")
        print(MODEL_PATH)
        return

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    classes = artifact["classes"]
    saved_features = artifact["feature_columns"]

    if list(saved_features) != FEATURE_COLUMNS:
        print("[FAIL] Feature columns do not match the trained model.")
        print("Model:", saved_features)
        print("Expected:", FEATURE_COLUMNS)
        return

    print("[PASS] Model loaded successfully.")
    print("[PASS] Features match.")
    print("Classes:", list(classes))
    print()
    print("This test only verifies live prediction.")
    print("Press Q to quit.")
    print()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[FAIL] Could not open camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[FAIL] Could not read camera frame.")
            break

        # Temporary test values.
        # These will be replaced by the real eye features
        # from main.py after we verify the model loads correctly.
        features = {
            "left_norm_x": 0.54,
            "left_norm_y": 0.46,
            "right_norm_x": 0.46,
            "right_norm_y": 0.45,
            "left_ear": 0.31,
            "right_ear": 0.31,
        }

        X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        prediction = model.predict(X)[0]

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.imshow("EyeControl AI - Live Model Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("[PASS] Live model test finished.")


if __name__ == "__main__":
    main()