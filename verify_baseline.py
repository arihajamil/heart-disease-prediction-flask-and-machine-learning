"""
Heart Disease Prediction - Regression baseline for the cleanup work
=========================================================
Runs five fixed patients through the real prediction pipeline and records
every number the app would produce for them.

    python verify_baseline.py --save      write baseline_predictions.json
    python verify_baseline.py             compare against the saved file

The point is to prove that refactoring changed nothing. It captures more
than the headline probability: the engineered features, the scaled vector,
every individual model's vote, the risk band and the model agreement. If
any of those shift by even a rounding step, the comparison fails loudly.
"""

import os
import sys
import json
import logging

logging.disable(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_FILE = os.path.join(BASE_DIR, "baseline_predictions.json")

import ml_engine   # noqa: E402
from config import Config   # noqa: E402


# Five patients spanning the whole risk range, chosen so a change anywhere
# in preprocessing, scaling, feature order or model loading would move at
# least one of them.
SAMPLES = [
    {"label": "S1 - healthy young woman",
     "form": {"patient_name": "Sample One", "patient_id": "PT-9001",
              "age": 28, "gender": 0, "height": 163, "weight": 55,
              "ap_hi": 108, "ap_lo": 70, "cholesterol": 1, "gluc": 1,
              "smoke": 0, "alco": 0, "active": 1, "notes": ""}},

    {"label": "S2 - healthy middle-aged man",
     "form": {"patient_name": "Sample Two", "patient_id": "PT-9002",
              "age": 41, "gender": 1, "height": 178, "weight": 78,
              "ap_hi": 122, "ap_lo": 80, "cholesterol": 1, "gluc": 1,
              "smoke": 0, "alco": 0, "active": 1, "notes": ""}},

    {"label": "S3 - borderline, several mild factors",
     "form": {"patient_name": "Sample Three", "patient_id": "PT-9003",
              "age": 49, "gender": 0, "height": 160, "weight": 76,
              "ap_hi": 134, "ap_lo": 86, "cholesterol": 2, "gluc": 1,
              "smoke": 0, "alco": 0, "active": 0, "notes": ""}},

    {"label": "S4 - hypertensive older man",
     "form": {"patient_name": "Sample Four", "patient_id": "PT-9004",
              "age": 58, "gender": 1, "height": 172, "weight": 94,
              "ap_hi": 152, "ap_lo": 94, "cholesterol": 3, "gluc": 2,
              "smoke": 1, "alco": 1, "active": 0, "notes": ""}},

    {"label": "S5 - high risk, every factor present",
     "form": {"patient_name": "Sample Five", "patient_id": "PT-9005",
              "age": 66, "gender": 1, "height": 170, "weight": 99,
              "ap_hi": 168, "ap_lo": 100, "cholesterol": 3, "gluc": 3,
              "smoke": 1, "alco": 1, "active": 0, "notes": ""}},
]

# Run each sample through every single model as well as the ensemble, so a
# change affecting only one classifier cannot hide behind the average.
MODEL_CHOICES = ["Ensemble Voting (All Models)"] + list(ml_engine.MODEL_FILES)


def capture():
    """Build the full record of what the pipeline produces for every sample."""
    ml_engine.init_engine(Config.MODELS_DIR)
    if not ml_engine.is_ready():
        print("ERROR: models are not loaded, cannot capture a baseline.")
        sys.exit(1)

    record = {
        "feature_order": ml_engine.get_feature_names(),
        "models_available": ml_engine.available_models(),
        "risk_thresholds": {"low_max": Config.RISK_LOW_MAX,
                            "moderate_max": Config.RISK_MODERATE_MAX},
        "decision_threshold": Config.POSITIVE_THRESHOLD,
        "samples": [],
    }

    for sample in SAMPLES:
        form = sample["form"]

        # Validation must agree too - a rule change would show up here
        is_valid, problems = ml_engine.validate_form(form)
        vector, readable = ml_engine.engineer_features(form)

        entry = {
            "label": sample["label"],
            "input": form,
            "valid": is_valid,
            "validation_errors": problems,
            "engineered": {k: readable[k] for k in
                           ("bmi", "pulse_pressure", "age_group", "high_risk_flag")},
            "feature_vector": vector,
            "predictions": {},
        }

        for choice in MODEL_CHOICES:
            outcome = ml_engine.predict(vector, choice)
            band = ml_engine.get_risk_level(outcome["probability"])
            entry["predictions"][choice] = {
                # Rounded to 10 decimals: enough to catch any real change,
                # tolerant of meaningless floating-point noise.
                "probability": round(outcome["probability"], 10),
                "predicted_class": outcome["predicted_class"],
                "model_used": outcome["model_used"],
                "models_at_risk": outcome["models_at_risk"],
                "total_models": outcome["total_models"],
                "risk_label": band["label"],
                "votes": {name: {"probability": round(vote["probability"], 10),
                                 "predicted": vote["predicted"]}
                          for name, vote in sorted(outcome["votes"].items())},
            }

        record["samples"].append(entry)

    return record


def save():
    """Capture the current behaviour and write it to baseline_predictions.json."""
    record = capture()
    with open(BASELINE_FILE, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print(f"Baseline written to {os.path.basename(BASELINE_FILE)}")
    show(record)


def show(record):
    """Print a one-line summary per sample."""
    print()
    print(f"{'SAMPLE':<38}{'BMI':>7}{'PP':>5}{'ENSEMBLE':>10}  BAND")
    print("-" * 74)
    for entry in record["samples"]:
        ens = entry["predictions"]["Ensemble Voting (All Models)"]
        print(f"{entry['label']:<38}"
              f"{entry['engineered']['bmi']:>7.2f}"
              f"{entry['engineered']['pulse_pressure']:>5}"
              f"{ens['probability']:>9.4%}  {ens['risk_label']}")
    print("-" * 74)


def _differences(old, new, path=""):
    """Walk two nested structures and yield every place they disagree."""
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old:
                yield f"{path}/{key}: ADDED"
            elif key not in new:
                yield f"{path}/{key}: REMOVED"
            else:
                yield from _differences(old[key], new[key], f"{path}/{key}")
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            yield f"{path}: length {len(old)} -> {len(new)}"
        else:
            for i, (a, b) in enumerate(zip(old, new)):
                yield from _differences(a, b, f"{path}[{i}]")
    elif old != new:
        yield f"{path}: {old!r} -> {new!r}"


def compare():
    """
    Re-run the samples and compare against the saved baseline.

    Returns 0 if everything matches, 1 if anything differs.
    """
    if not os.path.exists(BASELINE_FILE):
        print(f"No baseline found. Run:  python {os.path.basename(__file__)} --save")
        sys.exit(1)

    with open(BASELINE_FILE) as f:
        original = json.load(f)

    current = capture()
    diffs = list(_differences(original, current))

    show(current)
    print()
    if not diffs:
        print("PASS - all 5 sample predictions are byte-for-byte identical "
              "to the baseline.")
        return 0

    print(f"FAIL - {len(diffs)} difference(s) found:")
    for line in diffs[:40]:
        print(f"  {line}")
    if len(diffs) > 40:
        print(f"  ... and {len(diffs) - 40} more")
    return 1


if __name__ == "__main__":
    if "--save" in sys.argv:
        save()
    else:
        sys.exit(compare())
