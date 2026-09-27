# ========================================
# PHASE 3.6 - BASELINE ERROR ANALYSIS
# ========================================

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


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
print("PHASE 3.6 - BASELINE ERROR ANALYSIS")
print("=" * 70)

print("\nThis phase is READ-ONLY.")
print("- No training")
print("- No dataset modification")
print("- No main.py modification")
print("- No model modification")


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
        f"Model not found:\n{MODEL_PATH}"
    )

print("[PASS] Dataset found")
print("[PASS] Model found")


# ========================================
# LOAD DATA
# ========================================

print("\n[2] Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"[PASS] Dataset loaded: {len(df)} rows")


# ========================================
# PREPARE DATA
# ========================================

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


# ========================================
# RECREATE EXACT TEST SPLIT
# ========================================

print("\n[3] Recreating exact Phase 3.2 split...")

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

print(f"Test samples: {len(X_test)}")
print("[PASS] Exact test split recreated")


# ========================================
# LOAD MODEL
# ========================================

print("\n[4] Loading baseline model...")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]

print("[PASS] Model loaded")


# ========================================
# PREDICTIONS
# ========================================

print("\n[5] Generating test predictions...")

predictions = model.predict(X_test)

print("[PASS] Predictions generated")


# ========================================
# BUILD TEST RESULTS
# ========================================

results = X_test.copy()

results["actual"] = y_test.values
results["predicted"] = predictions

results["correct"] = (
    results["actual"] == results["predicted"]
)


# ========================================
# TOTAL ERROR SUMMARY
# ========================================

print("\n[6] Overall error summary")
print("-" * 70)

total_samples = len(results)
total_errors = int((~results["correct"]).sum())
total_correct = int(results["correct"].sum())

print(f"Total test samples : {total_samples}")
print(f"Correct            : {total_correct}")
print(f"Errors             : {total_errors}")
print(
    f"Error rate         : "
    f"{(total_errors / total_samples) * 100:.2f}%"
)


# ========================================
# ERROR PAIRS
# ========================================

print("\n[7] Error pairs")
print("-" * 70)

errors = results[~results["correct"]].copy()

if errors.empty:
    print("[PASS] No errors found.")
else:
    error_pairs = (
        errors
        .groupby(["actual", "predicted"])
        .size()
        .sort_values(ascending=False)
    )

    for (actual, predicted), count in error_pairs.items():
        print(
            f"{actual:<15} -> "
            f"{predicted:<15} : {count}"
        )


# ========================================
# PER-CLASS ERRORS
# ========================================

print("\n[8] Errors by actual class")
print("-" * 70)

for state in EXPECTED_CLASSES:

    state_results = results[
        results["actual"] == state
    ]

    state_errors = state_results[
        ~state_results["correct"]
    ]

    print(
        f"{state:<15} "
        f"Total={len(state_results):2d} "
        f"Errors={len(state_errors):2d}"
    )


# ========================================
# FEATURE MEANS FOR CORRECT VS ERROR
# ========================================

print("\n[9] Feature comparison")
print("-" * 70)

if not errors.empty:

    correct = results[
        results["correct"]
    ]

    print("\nMean feature values - CORRECT predictions:")

    for feature in FEATURE_COLUMNS:
        print(
            f"{feature:<15}: "
            f"{correct[feature].mean():.6f}"
        )

    print("\nMean feature values - INCORRECT predictions:")

    for feature in FEATURE_COLUMNS:
        print(
            f"{feature:<15}: "
            f"{errors[feature].mean():.6f}"
        )

else:
    print("No incorrect samples to analyze.")


# ========================================
# ERROR DETAILS
# ========================================

print("\n[10] Detailed error samples")
print("-" * 70)

if errors.empty:

    print("[PASS] No error samples.")

else:

    display_columns = [
        "actual",
        "predicted",
        "left_norm_x",
        "left_norm_y",
        "right_norm_x",
        "right_norm_y",
        "left_ear",
        "right_ear",
    ]

    for index, row in errors.iterrows():

        print(f"\nSample index: {index}")
        print(
            f"Actual    : {row['actual']}"
        )
        print(
            f"Predicted : {row['predicted']}"
        )

        for feature in FEATURE_COLUMNS:
            print(
                f"{feature:<15}: "
                f"{row[feature]:.6f}"
            )


# ========================================
# CONFUSION MATRIX
# ========================================

print("\n[11] Confusion matrix")
print("-" * 70)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=EXPECTED_CLASSES,
)

print("Rows = Actual")
print("Columns = Predicted\n")

print(
    "                 "
    + " ".join(
        f"{label:>13}"
        for label in EXPECTED_CLASSES
    )
)

for label, row in zip(
    EXPECTED_CLASSES,
    cm
):

    print(
        f"{label:>15} "
        + " ".join(
            f"{value:>13}"
            for value in row
        )
    )


# ========================================
# FINAL STATUS
# ========================================

print("\n" + "=" * 70)
print("PHASE 3.6 ERROR ANALYSIS COMPLETED")
print("=" * 70)

print("\n[PASS] Analysis completed successfully.")
print("[PASS] No dataset was modified.")
print("[PASS] No model was modified.")
print("[PASS] main.py was not modified.")
print("[PASS] No training was performed.")