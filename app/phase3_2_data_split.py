
import csv
from pathlib import Path

from sklearn.model_selection import train_test_split


# ============================================
# PHASE 3.2 - DATA SPLIT
# ============================================
# This script ONLY:
# - reads phase2_balanced.csv
# - extracts features and labels
# - creates Train / Validation / Test splits
#
# It does NOT:
# - modify phase2_balanced.csv
# - train a model
# - create a model
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = PROJECT_ROOT / "phase2_balanced.csv"

EXPECTED_STATES = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "EYES_OPEN",
    "EYES_CLOSED",
]

FEATURE_COLUMNS = [
    "left_norm_x",
    "left_norm_y",
    "right_norm_x",
    "right_norm_y",
    "left_ear",
    "right_ear",
]

EXPECTED_TOTAL = 700

RANDOM_STATE = 42


print("=" * 60)
print("             PHASE 3.2 - DATA SPLIT")
print("=" * 60)


# ============================================
# 1. CHECK DATASET
# ============================================

print("\n" + "=" * 60)
print("DATASET")
print("=" * 60)

if not CSV_FILE.exists():
    print(f"[FAIL] Dataset not found: {CSV_FILE}")
    input("\nPress Enter to exit...")
    raise SystemExit

print(f"[PASS] Dataset found:")
print(CSV_FILE)


# ============================================
# 2. READ DATA
# ============================================

X = []
y = []

with open(
    CSV_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        features = [
            float(row[column])
            for column in FEATURE_COLUMNS
        ]

        state = row["state"].strip()

        X.append(features)
        y.append(state)


print(f"\nTotal samples: {len(X)}")

if len(X) == EXPECTED_TOTAL:
    print("[PASS] Total sample count is 700.")
else:
    print("[FAIL] Unexpected sample count.")


# ============================================
# 3. ORIGINAL CLASS DISTRIBUTION
# ============================================

print("\n" + "=" * 60)
print("ORIGINAL CLASS DISTRIBUTION")
print("=" * 60)

for state in EXPECTED_STATES:
    count = y.count(state)
    print(f"{state:<13}: {count}")


# ============================================
# 4. TRAIN / TEMP SPLIT
# ============================================
# 70% Training
# 30% Temporary
#
# Stratification keeps class distribution.
# ============================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ============================================
# 5. VALIDATION / TEST SPLIT
# ============================================
# Split the temporary 30% equally:
#
# 15% Validation
# 15% Test
# ============================================

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)


# ============================================
# 6. SPLIT COUNTS
# ============================================

print("\n" + "=" * 60)
print("SPLIT SIZES")
print("=" * 60)

print(f"Training   : {len(X_train)}")
print(f"Validation : {len(X_val)}")
print(f"Test       : {len(X_test)}")

total_split = (
    len(X_train)
    + len(X_val)
    + len(X_test)
)

print(f"Total       : {total_split}")


# ============================================
# 7. CLASS DISTRIBUTION PER SPLIT
# ============================================

print("\n" + "=" * 60)
print("TRAINING DISTRIBUTION")
print("=" * 60)

for state in EXPECTED_STATES:
    print(f"{state:<13}: {y_train.count(state)}")


print("\n" + "=" * 60)
print("VALIDATION DISTRIBUTION")
print("=" * 60)

for state in EXPECTED_STATES:
    print(f"{state:<13}: {y_val.count(state)}")


print("\n" + "=" * 60)
print("TEST DISTRIBUTION")
print("=" * 60)

for state in EXPECTED_STATES:
    print(f"{state:<13}: {y_test.count(state)}")


# ============================================
# 8. FINAL VERIFICATION
# ============================================

print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

split_size_ok = (
    len(X_train) == 490
    and len(X_val) == 105
    and len(X_test) == 105
)

class_balance_ok = True

for state in EXPECTED_STATES:

    train_count = y_train.count(state)
    val_count = y_val.count(state)
    test_count = y_test.count(state)

    if train_count != 70:
        class_balance_ok = False

    if val_count != 15:
        class_balance_ok = False

    if test_count != 15:
        class_balance_ok = False


if split_size_ok:
    print("[PASS] Split sizes are correct.")
else:
    print("[FAIL] Split sizes are incorrect.")


if class_balance_ok:
    print("[PASS] Every class is balanced across all splits.")
else:
    print("[FAIL] Class distribution is incorrect.")


if total_split == EXPECTED_TOTAL:
    print("[PASS] All 700 samples are accounted for.")
else:
    print("[FAIL] Sample count mismatch.")


if (
    split_size_ok
    and class_balance_ok
    and total_split == EXPECTED_TOTAL
):
    print("\n[PASS] PHASE 3.2 DATA SPLIT VERIFIED")
else:
    print("\n[CHECK] PHASE 3.2 REQUIRES ATTENTION")


print("\nIMPORTANT:")
print("- phase2_balanced.csv was NOT modified.")
print("- No model was trained.")
print("- No model file was created.")
print("- Random state = 42")


print("\n" + "=" * 60)
print("                    END")
print("=" * 60)

input("\nPress Enter to exit...")

