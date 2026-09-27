import csv
import random
from pathlib import Path

# ============================================
# PHASE 2 - CREATE BALANCED DATASET
# ============================================
# This script:
# 1. Reads phase2_validation.csv
# 2. Checks every class
# 3. Creates phase2_balanced.csv ONLY if
#    every class has at least 100 samples.
#
# It NEVER modifies the original CSV.
# ============================================

INPUT_FILE = Path("phase2_validation.csv")
OUTPUT_FILE = Path("phase2_balanced.csv")

TARGET_PER_CLASS = 100

STATES = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "EYES_OPEN",
    "EYES_CLOSED",
]

print("=" * 60)
print("       PHASE 2 BALANCED DATASET CREATOR")
print("=" * 60)

# --------------------------------------------
# Check input file
# --------------------------------------------

if not INPUT_FILE.exists():
    print(f"\n[ERROR] File not found: {INPUT_FILE}")
    input("\nPress Enter to exit...")
    raise SystemExit

# --------------------------------------------
# Read original CSV
# --------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    fieldnames = reader.fieldnames
    rows = list(reader)

print(f"\nOriginal rows: {len(rows)}")

# --------------------------------------------
# Check columns
# --------------------------------------------

expected_columns = [
    "timestamp",
    "state",
    "left_norm_x",
    "left_norm_y",
    "right_norm_x",
    "right_norm_y",
    "left_ear",
    "right_ear",
]

if fieldnames != expected_columns:
    print("\n[ERROR] CSV columns are not as expected.")
    print("\nExpected:")
    print(expected_columns)

    print("\nFound:")
    print(fieldnames)

    input("\nPress Enter to exit...")
    raise SystemExit

print("[PASS] CSV columns are correct.")

# --------------------------------------------
# Separate rows by state
# --------------------------------------------

data_by_state = {state: [] for state in STATES}

for row in rows:
    state = row["state"].strip()

    if state in data_by_state:
        data_by_state[state].append(row)

# --------------------------------------------
# Show counts
# --------------------------------------------

print("\n" + "-" * 60)
print("AVAILABLE SAMPLES")
print("-" * 60)

missing = False

for state in STATES:
    count = len(data_by_state[state])

    if count < TARGET_PER_CLASS:
        missing = True
        print(
            f"{state:<13}: {count:>3} "
            f"[NEED {TARGET_PER_CLASS - count} MORE]"
        )
    else:
        print(
            f"{state:<13}: {count:>3} "
            f"[READY]"
        )

# --------------------------------------------
# Stop if any class is below target
# --------------------------------------------

if missing:
    print("\n" + "=" * 60)
    print("[STOP] Balanced dataset cannot be created yet.")
    print("=" * 60)

    print("\nEvery state must have at least 100 real samples.")

    print("\nThe original file was NOT modified:")
    print(INPUT_FILE)

    print("\nNo balanced file was created.")

    input("\nPress Enter to exit...")
    raise SystemExit

# --------------------------------------------
# Create exactly 100 samples per state
# --------------------------------------------

balanced_rows = []

random.seed(42)

for state in STATES:
    selected = random.sample(
        data_by_state[state],
        TARGET_PER_CLASS
    )

    balanced_rows.extend(selected)

# Shuffle final dataset
random.shuffle(balanced_rows)

# --------------------------------------------
# Write balanced dataset
# --------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(balanced_rows)

# --------------------------------------------
# Final report
# --------------------------------------------

print("\n" + "=" * 60)
print("       BALANCED DATASET CREATED")
print("=" * 60)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print(f"\nTotal rows: {len(balanced_rows)}")

print("\nFinal class counts:")

for state in STATES:
    print(f"{state:<13}: {TARGET_PER_CLASS}")

print("\nExpected:")
print("7 classes × 100 samples = 700 samples")

print("\nIMPORTANT:")
print("- Original phase2_validation.csv was NOT modified.")
print("- No samples were duplicated.")
print("- Selection uses a fixed random seed (42).")
print("- Exactly 100 samples are selected from each state.")

print("\n" + "=" * 60)
print("                    END")
print("=" * 60)

input("\nPress Enter to exit...")