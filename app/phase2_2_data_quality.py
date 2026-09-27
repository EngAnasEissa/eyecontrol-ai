import csv
import math
import statistics
from collections import Counter, defaultdict

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


# ============================================================
# LOAD DATA
# ============================================================

with open(PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    columns = reader.fieldnames


print()
print("============================================================")
print("             PHASE 2.2 DATA QUALITY REPORT")
print("============================================================")

print()
print("Total rows:", len(rows))

print()
print("---------------- COLUMN CHECK ----------------")

print("Expected columns:")
for column in EXPECTED_COLUMNS:
    print(" ", column)

print()
print("Actual columns:")
for column in columns:
    print(" ", column)

print()
print("Columns correct:", columns == EXPECTED_COLUMNS)


# ============================================================
# BASIC VALIDATION
# ============================================================

print()
print("============================================================")
print("                 BASIC DATA VALIDATION")
print("============================================================")

empty_values = 0
non_numeric_values = 0
non_finite_values = 0
unexpected_states = Counter()

for row in rows:

    if row["state"] not in STATES:
        unexpected_states[row["state"]] += 1

    for feature in FEATURES:

        value = row[feature]

        if value == "":
            empty_values += 1
            continue

        try:
            number = float(value)

            if not math.isfinite(number):
                non_finite_values += 1

        except ValueError:
            non_numeric_values += 1


print()
print("Empty values       :", empty_values)
print("Non-numeric values :", non_numeric_values)
print("Non-finite values  :", non_finite_values)

print()
print("Unexpected states:")

if unexpected_states:
    for state, count in unexpected_states.items():
        print(f"  {state}: {count}")
else:
    print("  NONE")


# ============================================================
# DUPLICATE CHECK
# ============================================================

print()
print("============================================================")
print("                  DUPLICATE ANALYSIS")
print("============================================================")

feature_rows = []

for row in rows:

    values = tuple(row[feature] for feature in FEATURES)

    feature_rows.append((row["state"], values))

duplicate_counter = Counter(feature_rows)

exact_duplicate_rows = sum(
    count - 1
    for count in duplicate_counter.values()
    if count > 1
)

duplicate_groups = sum(
    1
    for count in duplicate_counter.values()
    if count > 1
)

print()
print("Duplicate groups :", duplicate_groups)
print("Duplicate rows   :", exact_duplicate_rows)


# ============================================================
# PHYSICAL RANGE CHECK
# ============================================================

print()
print("============================================================")
print("              FEATURE RANGE VALIDATION")
print("============================================================")

# Normalized coordinates should normally be inside [0, 1].
# EAR should be positive. Extremely large values can indicate
# bad landmark geometry or invalid feature extraction.

range_limits = {
    "left_norm_x": (0.0, 1.0),
    "left_norm_y": (0.0, 1.0),
    "right_norm_x": (0.0, 1.0),
    "right_norm_y": (0.0, 1.0),
    "left_ear": (0.0, None),
    "right_ear": (0.0, None),
}

range_violations = defaultdict(int)

for row in rows:

    for feature in FEATURES:

        value = float(row[feature])

        minimum, maximum = range_limits[feature]

        if minimum is not None and value < minimum:
            range_violations[feature] += 1

        if maximum is not None and value > maximum:
            range_violations[feature] += 1


print()

for feature in FEATURES:

    violations = range_violations[feature]

    if violations == 0:
        print(f"{feature:12} : OK")
    else:
        print(f"{feature:12} : {violations} violations")


# ============================================================
# STATE COUNTS
# ============================================================

print()
print("============================================================")
print("                   STATE COUNTS")
print("============================================================")

state_data = defaultdict(list)

for row in rows:
    state = row["state"]
    state_data[state].append(row)


for state in STATES:
    print(f"{state:12} : {len(state_data[state])}")


# ============================================================
# OUTLIER ANALYSIS USING IQR
# ============================================================

print()
print("============================================================")
print("             OUTLIER ANALYSIS (IQR METHOD)")
print("============================================================")

print()
print("IQR = Interquartile Range")
print("Outlier rule = below Q1 - 1.5*IQR or above Q3 + 1.5*IQR")


def percentile(values, p):

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    weight = position - lower

    return (
        values[lower]
        + weight * (values[upper] - values[lower])
    )


for state in STATES:

    print()
    print(f"### {state}")
    print("-" * 75)

    for feature in FEATURES:

        values = [
            float(row[feature])
            for row in state_data[state]
        ]

        q1 = percentile(values, 0.25)
        q3 = percentile(values, 0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [
            value
            for value in values
            if value < lower_bound or value > upper_bound
        ]

        percentage = (len(outliers) / len(values)) * 100

        print(
            f"{feature:12} "
            f"outliers={len(outliers):3} "
            f"({percentage:6.2f}%)  "
            f"lower={lower_bound:.6f}  "
            f"upper={upper_bound:.6f}"
        )


# ============================================================
# EAR ANALYSIS
# ============================================================

print()
print("============================================================")
print("                    EAR ANALYSIS")
print("============================================================")

for state in STATES:

    left_values = [
        float(row["left_ear"])
        for row in state_data[state]
    ]

    right_values = [
        float(row["right_ear"])
        for row in state_data[state]
    ]

    left_mean = statistics.mean(left_values)
    right_mean = statistics.mean(right_values)

    left_median = statistics.median(left_values)
    right_median = statistics.median(right_values)

    print()
    print(state)

    print(
        f"  Left EAR  mean={left_mean:.6f} "
        f"median={left_median:.6f}"
    )

    print(
        f"  Right EAR mean={right_mean:.6f} "
        f"median={right_median:.6f}"
    )


# ============================================================
# EYES OPEN / CLOSED SEPARATION
# ============================================================

print()
print("============================================================")
print("          EYES OPEN / CLOSED SEPARATION CHECK")
print("============================================================")

for feature in ["left_ear", "right_ear"]:

    open_values = [
        float(row[feature])
        for row in state_data["EYES_OPEN"]
    ]

    closed_values = [
        float(row[feature])
        for row in state_data["EYES_CLOSED"]
    ]

    open_mean = statistics.mean(open_values)
    closed_mean = statistics.mean(closed_values)

    open_min = min(open_values)
    closed_max = max(closed_values)

    overlap = not (
        closed_max < open_min
    )

    print()
    print(feature)

    print(f"  EYES_OPEN mean   : {open_mean:.6f}")
    print(f"  EYES_CLOSED mean : {closed_mean:.6f}")
    print(f"  EYES_OPEN min    : {open_min:.6f}")
    print(f"  EYES_CLOSED max  : {closed_max:.6f}")

    print(
        "  Approximate overlap:",
        "YES" if overlap else "NO"
    )


# ============================================================
# DIRECTION FEATURE SEPARATION
# ============================================================

print()
print("============================================================")
print("             DIRECTION FEATURE SEPARATION")
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

        values = [
            float(row[feature])
            for row in state_data[state]
        ]

        mean = statistics.mean(values)
        median = statistics.median(values)

        print(
            f"  {state:8} "
            f"mean={mean:.6f} "
            f"median={median:.6f}"
        )


# ============================================================
# SAMPLE BALANCE
# ============================================================

print()
print("============================================================")
print("                 SAMPLE BALANCE")
print("============================================================")

counts = {
    state: len(state_data[state])
    for state in STATES
}

minimum_count = min(counts.values())
maximum_count = max(counts.values())

print()
print("Minimum class size:", minimum_count)
print("Maximum class size:", maximum_count)

print()
print("Relative to smallest class:")

for state in STATES:

    ratio = counts[state] / minimum_count

    print(
        f"  {state:12} "
        f"{counts[state]:3} samples "
        f"({ratio:.2f}x)"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("============================================================")
print("                    FINAL SUMMARY")
print("============================================================")

print()

if columns == EXPECTED_COLUMNS:
    print("[PASS] Column structure is correct.")
else:
    print("[CHECK] Column structure requires review.")

if empty_values == 0:
    print("[PASS] No empty values.")
else:
    print("[CHECK] Empty values detected.")

if non_numeric_values == 0:
    print("[PASS] No non-numeric values.")
else:
    print("[CHECK] Non-numeric values detected.")

if non_finite_values == 0:
    print("[PASS] No NaN/Infinity values.")
else:
    print("[CHECK] Non-finite values detected.")

if not unexpected_states:
    print("[PASS] No unexpected states.")
else:
    print("[CHECK] Unexpected states detected.")

if exact_duplicate_rows == 0:
    print("[PASS] No exact duplicate feature rows.")
else:
    print("[CHECK] Duplicate feature rows detected.")

if not range_violations:
    print("[PASS] All feature values are within physical ranges.")
else:
    print("[CHECK] Range violations detected.")

print()
print("IMPORTANT:")
print("This analysis does NOT modify phase2_validation.csv.")
print("No rows are deleted or changed.")

print()
print("============================================================")
print("                 END OF PHASE 2.2")
print("============================================================")