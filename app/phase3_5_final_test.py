# ========================================
# PHASE 3.5 - FINAL TEST EVALUATION
# ========================================

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ========================================
# PATHS
# ========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "phase2_balanced.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "phase3_baseline_random_forest.joblib"
)


# ========================================
# CONFIGURATION
# ========================================

FEATURE_COLUMNS = [
    "left_norm_x",
    "left_norm_y",
    "right_norm_x",
    "right_norm_y",
    "left_ear",
    "right_ear",
]

TARGET_COLUMN = "state"

EXPECTED_CLASSES = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "EYES_OPEN",
    "EYES_CLOSED",
]

RANDOM_STATE = 42


# ========================================
# HEADER
# ========================================

print("=" * 70)
print("PHASE 3.5 - FINAL TEST EVALUATION")
print("=" * 70)


# ========================================
# CHECK FILES
# ========================================

print("\n[1] Checking required files...")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_PATH}"
    )

print("[PASS] Dataset found")
print("[PASS] Trained model found")


# ========================================
# LOAD DATASET
# ========================================

print("\n[2] Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"[PASS] Dataset loaded: {len(df)} rows")


# ========================================
# VERIFY DATA
# ========================================

print("\n[3] Verifying dataset...")

missing_features = [
    column
    for column in FEATURE_COLUMNS
    if column not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Missing target column: {TARGET_COLUMN}"
    )

actual_classes = sorted(df[TARGET_COLUMN].unique())
expected_classes_sorted = sorted(EXPECTED_CLASSES)

if actual_classes != expected_classes_sorted:
    raise ValueError(
        f"Unexpected classes.\n"
        f"Expected: {expected_classes_sorted}\n"
        f"Found: {actual_classes}"
    )

print("[PASS] Feature columns verified")
print("[PASS] Target column verified")
print("[PASS] Seven classes verified")


# ========================================
# PREPARE DATA
# ========================================

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


# ========================================
# RECREATE EXACT PHASE 3.2 SPLIT
# ========================================

print("\n[4] Recreating exact Phase 3.2 split...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_validation)}")
print(f"Test samples       : {len(X_test)}")

if len(X_train) != 490:
    raise ValueError("Unexpected training size")

if len(X_validation) != 105:
    raise ValueError("Unexpected validation size")

if len(X_test) != 105:
    raise ValueError("Unexpected test size")

print("[PASS] Exact split recreated")


# ========================================
# LOAD MODEL
# ========================================

print("\n[5] Loading trained baseline model...")

artifact = joblib.load(MODEL_PATH)

if not isinstance(artifact, dict):
    raise ValueError(
        "Unexpected model artifact format"
    )

model = artifact["model"]

saved_features = artifact["feature_columns"]
saved_classes = artifact["classes"]

if saved_features != FEATURE_COLUMNS:
    raise ValueError(
        "Model feature configuration does not match dataset"
    )

if saved_classes != EXPECTED_CLASSES:
    raise ValueError(
        "Model class configuration does not match expected classes"
    )

print("[PASS] Model loaded")
print("[PASS] Feature configuration verified")
print("[PASS] Class configuration verified")


# ========================================
# FINAL TEST PREDICTION
# ========================================

print("\n[6] Running FINAL test evaluation...")
print("Important: Test data is being used for the first time.")

test_predictions = model.predict(X_test)


# ========================================
# TEST ACCURACY
# ========================================

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("\n" + "=" * 70)
print("FINAL TEST ACCURACY")
print("=" * 70)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ========================================
# CLASSIFICATION REPORT
# ========================================

print("\n[7] Final test classification report")
print("-" * 70)

print(
    classification_report(
        y_test,
        test_predictions,
        labels=EXPECTED_CLASSES,
        zero_division=0,
    )
)


# ========================================
# CONFUSION MATRIX
# ========================================

print("\n[8] Final test confusion matrix")
print("-" * 70)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=EXPECTED_CLASSES,
)

print("Rows    = Actual")
print("Columns = Predicted\n")

print("                 " + " ".join(
    f"{label:>13}"
    for label in EXPECTED_CLASSES
))

for label, row in zip(EXPECTED_CLASSES, cm):
    print(
        f"{label:>15} "
        + " ".join(
            f"{value:>13}"
            for value in row
        )
    )


# ========================================
# ERROR ANALYSIS
# ========================================

print("\n[9] Test error analysis")
print("-" * 70)

error_count = 0

for actual, predicted in zip(
    y_test,
    test_predictions
):
    if actual != predicted:
        error_count += 1
        print(
            f"Actual: {actual:<12} "
            f"Predicted: {predicted}"
        )

print(f"\nTotal test errors: {error_count}")
print(
    f"Total correct predictions: "
    f"{len(y_test) - error_count}"
)


# ========================================
# FINAL STATUS
# ========================================

print("\n" + "=" * 70)
print("PHASE 3.5 FINAL TEST EVALUATION COMPLETED")
print("=" * 70)

print("\nImportant:")
print("- Test set was evaluated only after training and validation.")
print("- No training was performed during this phase.")
print("- No dataset was modified.")
print("- main.py was NOT modified.")
print("- Baseline model was NOT modified.")

print("\n[PASS] PHASE 3.5 FINAL TEST EVALUATION COMPLETE")