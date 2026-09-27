# ========================================
# PHASE 3.7 - FEATURE IMPORTANCE ANALYSIS
# ========================================

from pathlib import Path

import joblib
import pandas as pd


# ========================================
# PATHS
# ========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "phase3_baseline_random_forest.joblib"
)

DATASET_PATH = PROJECT_ROOT / "phase2_balanced.csv"


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

EXPECTED_CLASSES = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "EYES_OPEN",
    "EYES_CLOSED",
]


# ========================================
# HEADER
# ========================================

print("=" * 70)
print("PHASE 3.7 - FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

print("\nThis phase is READ-ONLY.")
print("- No training")
print("- No dataset modification")
print("- No model modification")
print("- No main.py modification")


# ========================================
# CHECK FILES
# ========================================

print("\n[1] Checking required files...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

print("[PASS] Model found")
print("[PASS] Dataset found")


# ========================================
# LOAD MODEL
# ========================================

print("\n[2] Loading trained model...")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
saved_features = artifact["feature_columns"]
saved_classes = artifact["classes"]

if saved_features != FEATURE_COLUMNS:
    raise ValueError(
        "Saved feature list does not match expected features."
    )

if saved_classes != EXPECTED_CLASSES:
    raise ValueError(
        "Saved class list does not match expected classes."
    )

print("[PASS] Model loaded")
print("[PASS] Feature configuration verified")
print("[PASS] Class configuration verified")


# ========================================
# FEATURE IMPORTANCE
# ========================================

print("\n[3] Calculating feature importance...")
print("-" * 70)

importance_values = model.feature_importances_

if len(importance_values) != len(FEATURE_COLUMNS):
    raise ValueError(
        "Feature importance count does not match feature count."
    )

feature_importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": importance_values,
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)

print("\nFeature importance ranking:\n")

for position, row in feature_importance.iterrows():

    print(
        f"{position + 1}. "
        f"{row['feature']:<15} "
        f"{row['importance']:.6f} "
        f"({row['importance'] * 100:.2f}%)"
    )


# ========================================
# IMPORTANCE TOTAL
# ========================================

importance_sum = feature_importance["importance"].sum()

print("\n[4] Importance validation")
print("-" * 70)

print(
    f"Total importance: "
    f"{importance_sum:.6f}"
)

if abs(importance_sum - 1.0) > 0.000001:
    raise ValueError(
        "Feature importance values do not sum to 1."
    )

print("[PASS] Feature importance sums to 1.0")


# ========================================
# GROUP FEATURES
# ========================================

print("\n[5] Comparing feature groups")
print("-" * 70)

coordinate_features = [
    "left_norm_x",
    "left_norm_y",
    "right_norm_x",
    "right_norm_y",
]

ear_features = [
    "left_ear",
    "right_ear",
]

coordinate_importance = feature_importance[
    feature_importance["feature"].isin(
        coordinate_features
    )
]["importance"].sum()

ear_importance = feature_importance[
    feature_importance["feature"].isin(
        ear_features
    )
]["importance"].sum()

print(
    f"Eye-position features : "
    f"{coordinate_importance:.6f} "
    f"({coordinate_importance * 100:.2f}%)"
)

print(
    f"EAR features          : "
    f"{ear_importance:.6f} "
    f"({ear_importance * 100:.2f}%)"
)


# ========================================
# TOP FEATURE
# ========================================

top_feature = feature_importance.iloc[0]

print("\n[6] Most important feature")
print("-" * 70)

print(
    f"Feature    : {top_feature['feature']}"
)

print(
    f"Importance  : "
    f"{top_feature['importance']:.6f}"
)

print(
    f"Percentage  : "
    f"{top_feature['importance'] * 100:.2f}%"
)


# ========================================
# LOW IMPORTANCE FEATURES
# ========================================

print("\n[7] Lowest importance features")
print("-" * 70)

for _, row in feature_importance.tail(2).iterrows():

    print(
        f"{row['feature']:<15} "
        f"{row['importance']:.6f} "
        f"({row['importance'] * 100:.2f}%)"
    )


# ========================================
# DATASET FEATURE STATISTICS
# ========================================

print("\n[8] Dataset feature statistics")
print("-" * 70)

df = pd.read_csv(DATASET_PATH)

print(
    "\nMean values across all 700 samples:\n"
)

for feature in FEATURE_COLUMNS:

    print(
        f"{feature:<15}: "
        f"{df[feature].mean():.6f}"
    )


# ========================================
# FEATURE VARIATION
# ========================================

print("\n[9] Feature standard deviation")
print("-" * 70)

for feature in FEATURE_COLUMNS:

    print(
        f"{feature:<15}: "
        f"{df[feature].std():.6f}"
    )


# ========================================
# FINAL STATUS
# ========================================

print("\n" + "=" * 70)
print("PHASE 3.7 FEATURE ANALYSIS COMPLETED")
print("=" * 70)

print("\n[PASS] Feature importance calculated.")
print("[PASS] Feature groups compared.")
print("[PASS] Dataset statistics calculated.")
print("[PASS] No dataset was modified.")
print("[PASS] No model was modified.")
print("[PASS] main.py was not modified.")
print("[PASS] No training was performed.")