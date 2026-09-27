import csv
import math
from pathlib import Path

# ============================================
# PHASE 2 - BALANCED DATA QUALITY CHECK
# ============================================
# This script ONLY checks phase2_balanced.csv.
# It does NOT modify any file.
# ============================================

CSV_FILE = Path("phase2_balanced.csv")

EXPECTED_COLUMNS = [
    "timestamp",
    "state",
    "left_norm_x",
    "left_norm_y",
    "right_norm_x",
    "right_norm_y",
    "left_ear",
    "right_ear",
]

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

# Expected physical ranges
FEATURE_RANGES = {
    "left_norm_x": (0.0, 1.0),
    "left_norm_y": (0.0, 1.0),
    "right_norm_x": (0.0, 1.0),
    "right_norm_y": (0.0, 1.0),
    "left_ear": (0.0, 1.0),
    "right_ear": (0.0, 1.0),
}


print("=" * 60)
print("        PHASE 2 BALANCED DATA QUALITY CHECK")
print("=" * 60)

# --------------------------------------------
# Check file
# --------------------------------------------

if not CSV_FILE.exists():
    print(f"\n[ERROR] File not found: {CSV_FILE}")
    input("\nPress Enter to exit...")
    raise SystemExit

# --------------------------------------------
# Read CSV
# --------------------------------------------

with open(
    CSV_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.DictReader(file)

    columns = reader.fieldnames
    rows = list(reader)

print(f"\nTotal rows: {len(rows)}")

# --------------------------------------------
# COLUMN CHECK
# --------------------------------------------

print("\n" + "=" * 60)
print("COLUMN CHECK")
print("=" * 60)

print("\nExpected columns:")
for column in EXPECTED_COLUMNS:
    print(f"  {column}")

print("\nActual columns:")
for column in columns or []:
    print(f"  {column}")

if columns == EXPECTED_COLUMNS:
    print("\n[PASS] Columns are correct.")
else:
    print("\n[FAIL] Column structure is incorrect.")

# --------------------------------------------
# STATE COUNTS
# --------------------------------------------

print("\n" + "=" * 60)
print("STATE COUNTS")
print("=" * 60)

state_counts = {state: 0 for state in EXPECTED_STATES}
unexpected_states = set()

for row in rows:
    state = row.get("state", "").strip()

    if state in state_counts:
        state_counts[state] += 1
    else:
        unexpected_states.add(state)

for state in EXPECTED_STATES:
    print(f"{state:<13}: {state_counts[state]}")

if unexpected_states:
    print("\n[FAIL] Unexpected states:")
    for state in sorted(unexpected_states):
        print(f"  {state}")
else:
    print("\n[PASS] No unexpected states.")

# --------------------------------------------
# BALANCE CHECK
# --------------------------------------------

print("\n" + "=" * 60)
print("BALANCE CHECK")
print("=" * 60)

balance_ok = True

for state in EXPECTED_STATES:
    if state_counts[state] != 100:
        balance_ok = False
        print(
            f"[FAIL] {state}: "
            f"{state_counts[state]} samples"
        )

if balance_ok:
    print("[PASS] Every state contains exactly 100 samples.")

# --------------------------------------------
# EMPTY / NON-NUMERIC / NON-FINITE CHECK
# --------------------------------------------

print("\n" + "=" * 60)
print("DATA VALIDITY")
print("=" * 60)

empty_values = 0
non_numeric_values = 0
non_finite_values = 0

for row_number, row in enumerate(rows, start=2):

    for column in EXPECTED_COLUMNS:

        value = row.get(column, "")

        if value is None or value.strip() == "":
            empty_values += 1
            continue

        if column in FEATURE_COLUMNS:

            try:
                number = float(value)

                if not math.isfinite(number):
                    non_finite_values += 1

            except ValueError:
                non_numeric_values += 1

print(f"Empty values       : {empty_values}")
print(f"Non-numeric values : {non_numeric_values}")
print(f"Non-finite values  : {non_finite_values}")

if empty_values == 0:
    print("[PASS] No empty values.")

if non_numeric_values == 0:
    print("[PASS] No non-numeric feature values.")

if non_finite_values == 0:
    print("[PASS] No NaN/Infinity values.")

# --------------------------------------------
# RANGE CHECK
# --------------------------------------------

print("\n" + "=" * 60)
print("FEATURE RANGE CHECK")
print("=" * 60)

range_violations = 0

for feature in FEATURE_COLUMNS:

    minimum, maximum = FEATURE_RANGES[feature]

    feature_min = float("inf")
    feature_max = float("-inf")

    violations = 0

    for row in rows:

        try:
            value = float(row[feature])

            feature_min = min(feature_min, value)
            feature_max = max(feature_max, value)

            if value < minimum or value > maximum:
                violations += 1

        except (ValueError, TypeError):
            continue

    print(
        f"{feature:<13}: "
        f"min={feature_min:.6f} "
        f"max={feature_max:.6f} "
        f"violations={violations}"
    )

    range_violations += violations

if range_violations == 0:
    print("\n[PASS] All feature values are within valid ranges.")
else:
    print(
        f"\n[FAIL] Total range violations: "
        f"{range_violations}"
    )

# --------------------------------------------
# DUPLICATE CHECK
# --------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

feature_rows = set()
duplicate_rows = 0

for row in rows:

    key = tuple(
        row.get(feature, "")
        for feature in FEATURE_COLUMNS
    )

    if key in feature_rows:
        duplicate_rows += 1
    else:
        feature_rows.add(key)

print(f"Duplicate feature rows: {duplicate_rows}")

if duplicate_rows == 0:
    print("[PASS] No duplicate feature rows.")
else:
    print("[CHECK] Duplicate feature rows detected.")

# --------------------------------------------
# FINAL RESULT
# --------------------------------------------

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

all_pass = (
    len(rows) == 700
    and columns == EXPECTED_COLUMNS
    and not unexpected_states
    and balance_ok
    and empty_values == 0
    and non_numeric_values == 0
    and non_finite_values == 0
    and range_violations == 0
    and duplicate_rows == 0
)

if all_pass:
    print("\n[PASS] PHASE 2 BALANCED DATASET VERIFIED")
    print("\n700 valid samples")
    print("7 states")
    print("100 samples per state")
else:
    print("\n[CHECK] Some quality checks require attention.")

print("\nIMPORTANT:")
print("- phase2_balanced.csv was NOT modified.")
print("- phase2_validation.csv was NOT modified.")

print("\n" + "=" * 60)
print("                    END")
print("=" * 60)

input("\nPress Enter to exit...")