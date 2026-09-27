import csv
import math
import collections

path = "phase2_validation.csv"

expected = [
    "timestamp",
    "state",
    "left_norm_x",
    "left_norm_y",
    "right_norm_x",
    "right_norm_y",
    "left_ear",
    "right_ear",
]

states = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "EYES_OPEN",
    "EYES_CLOSED",
]

with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print()
print("========================================")
print("       PHASE 2 VALIDATION REPORT")
print("========================================")

print()
print("Total data rows:", len(rows))

if rows:
    columns = list(rows[0].keys())
    print("Columns:", columns)
    print("Columns correct:", columns == expected)
else:
    columns = []
    print("Columns: NO DATA")
    print("Columns correct: False")

# -----------------------------
# STATES
# -----------------------------

counts = collections.Counter(row.get("state", "") for row in rows)

print()
print("--- STATES ---")

for state in states:
    print(state, ":", counts.get(state, 0), "rows")

extra = set(counts) - set(states)

print("Unexpected states:", extra if extra else "NONE")

missing = [state for state in states if counts.get(state, 0) == 0]

print("Missing states:", missing if missing else "NONE")

# -----------------------------
# DATA QUALITY
# -----------------------------

wrong_columns = 0
empty_values = 0
nonnumeric = 0
nonfinite = 0

ranges = {
    column: [float("inf"), float("-inf")]
    for column in expected[2:]
}

for row in rows:

    if list(row.keys()) != expected:
        wrong_columns += 1

    for column in expected:
        if row.get(column, "") == "":
            empty_values += 1

    for column in expected[2:]:

        try:
            value = float(row[column])
        except (ValueError, TypeError):
            nonnumeric += 1
            continue

        if not math.isfinite(value):
            nonfinite += 1
            continue

        ranges[column][0] = min(ranges[column][0], value)
        ranges[column][1] = max(ranges[column][1], value)

print()
print("--- DATA QUALITY ---")

print("Rows with wrong columns:", wrong_columns)
print("Empty values:", empty_values)
print("Non-numeric values:", nonnumeric)
print("Non-finite values:", nonfinite)

print()
print("--- RANGES ---")

for column, values in ranges.items():

    minimum, maximum = values

    if minimum == float("inf"):
        print(column, ": NO VALID DATA")
    else:
        print(
            column,
            ":",
            round(minimum, 6),
            "to",
            round(maximum, 6)
        )

print()
print("========================================")
print("          END REPORT")
print("========================================")