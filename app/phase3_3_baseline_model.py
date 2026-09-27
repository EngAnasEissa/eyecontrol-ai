
from sklearn.ensemble import RandomForestClassifier


# ============================================
# PHASE 3.3 - BASELINE MODEL CHECK
# ============================================
# This script ONLY checks the baseline model
# configuration.
#
# It does NOT:
# - read the dataset
# - train the model
# - create a model file
# - modify any project file
# ============================================


MODEL_NAME = "RandomForestClassifier"

MODEL_PARAMETERS = {
    "n_estimators": 200,
    "random_state": 42,
    "max_depth": None,
    "class_weight": None,
}


EXPECTED_FEATURE_COUNT = 6

EXPECTED_CLASS_COUNT = 7


FEATURES = [
    "left_norm_x",
    "left_norm_y",
    "right_norm_x",
    "right_norm_y",
    "left_ear",
    "right_ear",
]


CLASSES = [
    "FORWARD",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "EYES_OPEN",
    "EYES_CLOSED",
]


print("=" * 60)
print("        PHASE 3.3 - BASELINE MODEL CHECK")
print("=" * 60)


# ============================================
# 1. MODEL IMPORT
# ============================================

print("\n" + "=" * 60)
print("MODEL IMPORT")
print("=" * 60)

try:

    model = RandomForestClassifier(
        n_estimators=MODEL_PARAMETERS["n_estimators"],
        random_state=MODEL_PARAMETERS["random_state"],
        max_depth=MODEL_PARAMETERS["max_depth"],
        class_weight=MODEL_PARAMETERS["class_weight"],
    )

    print("[PASS] RandomForestClassifier imported.")
    print("[PASS] Model object created.")

except Exception as error:

    print("[FAIL] Could not create model.")
    print(error)

    input("\nPress Enter to exit...")
    raise SystemExit


# ============================================
# 2. FEATURE CHECK
# ============================================

print("\n" + "=" * 60)
print("FEATURE CONFIGURATION")
print("=" * 60)

print(f"Feature count: {len(FEATURES)}")

for feature in FEATURES:
    print(f"  {feature}")

if len(FEATURES) == EXPECTED_FEATURE_COUNT:
    print("\n[PASS] Six model features are configured.")
else:
    print("\n[FAIL] Unexpected feature count.")


# ============================================
# 3. CLASS CHECK
# ============================================

print("\n" + "=" * 60)
print("CLASS CONFIGURATION")
print("=" * 60)

print(f"Class count: {len(CLASSES)}")

for class_name in CLASSES:
    print(f"  {class_name}")

if len(CLASSES) == EXPECTED_CLASS_COUNT:
    print("\n[PASS] Seven classes are configured.")
else:
    print("\n[FAIL] Unexpected class count.")


# ============================================
# 4. MODEL PARAMETERS
# ============================================

print("\n" + "=" * 60)
print("MODEL PARAMETERS")
print("=" * 60)

for parameter, value in MODEL_PARAMETERS.items():
    print(f"{parameter:<15}: {value}")


# ============================================
# 5. MODEL TYPE
# ============================================

print("\n" + "=" * 60)
print("MODEL TYPE")
print("=" * 60)

print(f"Model: {MODEL_NAME}")

if isinstance(model, RandomForestClassifier):
    print("[PASS] Correct model type.")
else:
    print("[FAIL] Incorrect model type.")


# ============================================
# 6. FINAL RESULT
# ============================================

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

configuration_ok = (
    isinstance(model, RandomForestClassifier)
    and len(FEATURES) == EXPECTED_FEATURE_COUNT
    and len(CLASSES) == EXPECTED_CLASS_COUNT
    and MODEL_PARAMETERS["n_estimators"] == 200
    and MODEL_PARAMETERS["random_state"] == 42
)

if configuration_ok:

    print("[PASS] PHASE 3.3 BASELINE MODEL CONFIGURATION VERIFIED")

else:

    print("[CHECK] Baseline model configuration requires attention.")


print("\nIMPORTANT:")
print("- No dataset was modified.")
print("- No model was trained.")
print("- No model file was created.")


print("\n" + "=" * 60)
print("                    END")
print("=" * 60)

input("\nPress Enter to exit...")
