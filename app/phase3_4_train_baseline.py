# ========================================
# PHASE 3.4 - TRAIN BASELINE MODEL
# ========================================

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ========================================
# PATHS
# ========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "phase2_balanced.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "phase3_baseline_random_forest.joblib"


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
# LOAD DATA
# ========================================

print("=" * 70)
print("PHASE 3.4 - BASELINE MODEL TRAINING")
print("=" * 70)

print("\n[1] Loading balanced dataset...")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print(f"[PASS] Dataset loaded: {len(df)} rows")


# ========================================
# VERIFY DATA
# ========================================

print("\n[2] Verifying dataset...")

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
# PREPARE X AND Y
# ========================================

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


# ========================================
# RECREATE EXACT PHASE 3.2 SPLIT
# ========================================

print("\n[3] Creating train/validation/test split...")

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

print("[PASS] Split matches Phase 3.2")


# ========================================
# CREATE BASELINE MODEL
# ========================================

print("\n[4] Creating Random Forest baseline...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=None,
    class_weight=None,
)

print("Model: RandomForestClassifier")
print("n_estimators: 200")
print("random_state: 42")
print("max_depth: None")
print("class_weight: None")

print("[PASS] Baseline configuration verified")


# ========================================
# TRAIN
# ========================================

print("\n[5] Training model...")
print("Please wait...")

model.fit(X_train, y_train)

print("[PASS] Model training completed")


# ========================================
# TRAINING EVALUATION
# ========================================

print("\n[6] Training accuracy...")

train_predictions = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")


# ========================================
# VALIDATION EVALUATION
# ========================================

print("\n[7] Validation accuracy...")

validation_predictions = model.predict(X_validation)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

print(f"Validation Accuracy: {validation_accuracy:.4f}")
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")


# ========================================
# VALIDATION CLASSIFICATION REPORT
# ========================================

print("\n[8] Validation classification report")
print("-" * 70)

print(
    classification_report(
        y_validation,
        validation_predictions,
        labels=EXPECTED_CLASSES,
        zero_division=0,
    )
)


# ========================================
# SAVE MODEL
# ========================================

print("\n[9] Saving trained model...")

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

artifact = {
    "model": model,
    "feature_columns": FEATURE_COLUMNS,
    "classes": EXPECTED_CLASSES,
    "random_state": RANDOM_STATE,
    "n_estimators": 200,
}

joblib.dump(
    artifact,
    MODEL_PATH
)

print(f"[PASS] Model saved to:")
print(MODEL_PATH)


# ========================================
# FINAL STATUS
# ========================================

print("\n" + "=" * 70)
print("PHASE 3.4 TRAINING COMPLETED")
print("=" * 70)

print("\nImportant:")
print("- Training set was used for training.")
print("- Validation set was used for validation.")
print("- Test set was NOT used for evaluation.")
print("- Original datasets were NOT modified.")
print("- main.py was NOT modified.")
print("- Baseline model was saved separately.")

print("\n[PASS] PHASE 3.4 BASELINE TRAINING COMPLETE")