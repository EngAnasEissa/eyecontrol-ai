
import csv
import importlib.util
import sys
from pathlib import Path


# ============================================
# PHASE 3.1 - ENVIRONMENT CHECK
# ============================================
# This script ONLY checks the environment
# and the Phase 2 balanced dataset.
#
# It does NOT:
# - modify any file
# - install packages
# - train a model
# - create a model
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "phase2_balanced.csv"

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

REQUIRED_FEATURES = [
    "left_norm_x",
    "left_norm_y",
    "right_norm_x",
    "right_norm_y",
    "left_ear",
    "right_ear",
]

PACKAGES = [
    "numpy",
    "sklearn",
]


def check_package(package_name):
    return importlib.util.find_spec(package_name) is not None


print("=" * 60)
print("           PHASE 3.1 ENVIRONMENT CHECK")
print("=" * 60)

print("\nPython")
print("-" * 60)
print(f"Version : {sys.version}")
print(f"Executable:")
print(sys.executable)

print("\nProject")
print("-" * 60)
print(f"Project root : {PROJECT_ROOT}")
print(f"Dataset      : {CSV_FILE}")

print("\n" + "=" * 60)
print("DATASET EXISTENCE")
print("=" * 60)

if CSV_FILE.exists():
    print("[PASS] phase2_balanced.csv exists.")
else:
    print("[FAIL] phase2_balanced.csv was not found.")

print("\n" + "=" * 60)
print("REQUIRED PACKAGES")
print("=" * 60)

package_status = {}

for package in PACKAGES:
    available = check_package(package)
    package_status[package] = available

    if available:
        print(f"[PASS] {package}")
    else:
        print(f"[MISSING] {package}")

print("\n" + "=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)

dataset_ok = True

if CSV_FILE.exists():

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)

    print(f"Rows: {len(rows)}")

    if columns == EXPECTED_COLUMNS:
        print("[PASS] Columns are correct.")
    else:
        print("[FAIL] Columns are incorrect.")
        dataset_ok = False

    states = sorted(
        set(
            row.get("state", "").strip()
            for row in rows
        )
    )

    print("\nStates found:")
    for state in states:
        print(f"  {state}")

    if set(states) == set(EXPECTED_STATES):
        print("[PASS] Expected states are present.")
    else:
        print("[FAIL] State list does not match expected states.")
        dataset_ok = False

    if len(rows) == 700:
        print("[PASS] Dataset contains exactly 700 rows.")
    else:
        print("[FAIL] Dataset row count is not 700.")
        dataset_ok = False

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in (columns or [])
    ]

    if not missing_features:
        print("[PASS] All six model features are available.")
    else:
        print("[FAIL] Missing features:")
        for feature in missing_features:
            print(f"  {feature}")
        dataset_ok = False

else:
    dataset_ok = False


print("\n" + "=" * 60)
print("PHASE 3.1 RESULT")
print("=" * 60)

environment_ok = all(package_status.values())

if dataset_ok:
    print("[PASS] Dataset is ready for Phase 3.")
else:
    print("[CHECK] Dataset requires attention.")

if environment_ok:
    print("[PASS] Required Python packages are available.")
else:
    print("[CHECK] One or more required packages are missing.")

if dataset_ok and environment_ok:
    print("\n[PASS] PHASE 3.1 ENVIRONMENT READY")
else:
    print("\n[CHECK] PHASE 3.1 IS NOT READY")

print("\nIMPORTANT:")
print("- No files were modified.")
print("- No packages were installed.")
print("- No model was trained.")

print("\n" + "=" * 60)
print("                    END")
print("=" * 60)

input("\nPress Enter to exit...")

