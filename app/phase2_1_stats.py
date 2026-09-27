import csv
import math
import statistics
from collections import defaultdict

PATH = "phase2_validation.csv"

FEATURES = [
    "left_norm_x",
    "left_norm_y",
    "right_norm_x",
    "right_norm_y",
    "left_ear",
    "right_ear",
]

STATES = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "EYES_OPEN",
    "EYES_CLOSED",
]

with open(PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

data = defaultdict(lambda: defaultdict(list))

for row in rows:
    state = row["state"]

    for feature in FEATURES:
        value = float(row[feature])

        if math.isfinite(value):
            data[state][feature].append(value)

print()
print("============================================================")
print("              PHASE 2.1 STATISTICAL REPORT")
print("============================================================")

print()
print("Total rows:", len(rows))

print()
print("---------------- SAMPLE COUNTS ----------------")

for state in STATES:
    print(f"{state:12} : {len(data[state][FEATURES[0]])}")

print()
print("============================================================")
print("              STATISTICS BY STATE")
print("============================================================")

for state in STATES:

    print()
    print(f"### {state}")
    print("-" * 75)

    for feature in FEATURES:

        values = data[state][feature]

        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        minimum = min(values)
        maximum = max(values)
        value_range = maximum - minimum

        if mean != 0:
            cv = (std / abs(mean)) * 100
        else:
            cv = 0.0

        print(
            f"{feature:12} "
            f"mean={mean:.6f}  "
            f"std={std:.6f}  "
            f"min={minimum:.6f}  "
            f"max={maximum:.6f}  "
            f"range={value_range:.6f}  "
            f"CV={cv:.2f}%"
        )

print()
print("============================================================")
print("          GLOBAL FEATURE STATISTICS")
print("============================================================")

for feature in FEATURES:

    values = []

    for state in STATES:
        values.extend(data[state][feature])

    mean = statistics.mean(values)
    std = statistics.stdev(values)
    minimum = min(values)
    maximum = max(values)

    print()
    print(feature)
    print(f"  Mean : {mean:.6f}")
    print(f"  Std  : {std:.6f}")
    print(f"  Min  : {minimum:.6f}")
    print(f"  Max  : {maximum:.6f}")
    print(f"  Range: {maximum - minimum:.6f}")

print()
print("============================================================")
print("       EYES_OPEN vs EYES_CLOSED EAR ANALYSIS")
print("============================================================")

for feature in ["left_ear", "right_ear"]:

    open_values = data["EYES_OPEN"][feature]
    closed_values = data["EYES_CLOSED"][feature]

    open_mean = statistics.mean(open_values)
    closed_mean = statistics.mean(closed_values)

    difference = open_mean - closed_mean

    print()
    print(feature)
    print(f"  EYES_OPEN mean   : {open_mean:.6f}")
    print(f"  EYES_CLOSED mean : {closed_mean:.6f}")
    print(f"  Difference       : {difference:.6f}")

print()
print("============================================================")
print("       DIRECTION FEATURE ANALYSIS")
print("============================================================")

direction_states = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
]

for feature in [
    "left_norm_x",
    "left_norm_y",
    "right_norm_x",
    "right_norm_y",
]:

    print()
    print(feature)

    for state in direction_states:

        values = data[state][feature]
        mean = statistics.mean(values)

        print(f"  {state:8} mean = {mean:.6f}")

print()
print("============================================================")
print("                 END OF PHASE 2.1")
print("============================================================")