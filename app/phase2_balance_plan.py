import csv
from pathlib import Path

# ============================================
# PHASE 2 - DATA BALANCE PLAN
# This script ONLY analyzes the data.
# It does NOT modify phase2_validation.csv.
# ============================================

CSV_FILE = Path("phase2_validation.csv")

TARGET_PER_CLASS = 100

EXPECTED_STATES = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "EYES_OPEN",
    "EYES_CLOSED",
]

print("=" * 60)
print("           PHASE 2 DATA BALANCE PLAN")
print("=" * 60)

if not CSV_FILE.exists():
    print(f"\n[ERROR] File not found: {CSV_FILE}")
    print("Make sure the script is running from the project folder.")
    input("\nPress Enter to exit...")
    raise SystemExit

# Read CSV without pandas
counts = {state: 0 for state in EXPECTED_STATES}
total_rows = 0

with open(CSV_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        total_rows += 1
        state = row.get("state", "").strip()

        if state in counts:
            counts[state] += 1

print(f"\nCurrent total rows: {total_rows}")
print(f"Target per class   : {TARGET_PER_CLASS}")

print("\n" + "-" * 60)
print("CURRENT COUNTS")
print("-" * 60)

total_needed = 0

for state in EXPECTED_STATES:
    current = counts[state]

    if current < TARGET_PER_CLASS:
        needed = TARGET_PER_CLASS - current
        status = f"NEED +{needed}"
        total_needed += needed

    elif current == TARGET_PER_CLASS:
        needed = 0
        status = "TARGET REACHED"

    else:
        needed = 0
        status = f"EXCESS ({current - TARGET_PER_CLASS})"

    print(f"{state:<13}: {current:>3}  ->  {status}")

print("\n" + "-" * 60)
print("BALANCING SUMMARY")
print("-" * 60)

print(f"Additional real samples needed: {total_needed}")

print("\nRecommended target:")
print("100 real samples per state")

print("\nImportant:")
print("- phase2_validation.csv was NOT modified.")
print("- Existing RIGHT samples were NOT deleted.")
print("- Samples will NOT be duplicated.")
print("- New samples will be recorded using the existing validation system.")

print("\n" + "=" * 60)
print("                    END OF PLAN")
print("=" * 60)

input("\nPress Enter to exit...")