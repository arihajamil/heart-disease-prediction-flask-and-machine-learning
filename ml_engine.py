"""
Heart Disease Prediction - Machine Learning Engine
========================================
Everything to do with making a prediction lives here:

  * loading the scaler and the five trained models ONCE at startup
  * turning raw form input into the 15 features the models expect
  * validating that the input is medically possible
  * running the prediction and reporting how many models agree

The feature formulas below must stay identical to the ones in
train_models.py. If they ever drift apart the models silently receive
different numbers in real use than they saw during training.
"""

import os
import json
import pickle
import logging

from config import Config

log = logging.getLogger(__name__)

# ── Module-level cache, filled once by init_engine() ──────────────
_SCALER = None
_MODELS = {}
_FEATURE_NAMES = []
_LOAD_WARNINGS = []
_MODELS_DIR = None

# Which pickle holds which classifier - defined once, in config.py
MODEL_FILES = Config.MODEL_FILES

# Short labels used in tables and charts
MODEL_SHORT = {
    "Logistic Regression": "LR",
    "Support Vector Machine (SVM)": "SVM",
    "Decision Tree": "DT",
    "Random Forest": "RF",
    "XGBoost": "XGB",
    Config.ENSEMBLE_NAME: "ENS",
}

# Fallback order if models/features.json cannot be read. The authoritative
# order still comes from that file, which training writes.
DEFAULT_FEATURES = Config.FEATURE_ORDER

# Friendly names for charts and tables
FEATURE_LABELS = {
    "age": "Age", "gender": "Gender", "height": "Height", "weight": "Weight",
    "ap_hi": "Systolic BP", "ap_lo": "Diastolic BP",
    "cholesterol": "Cholesterol", "gluc": "Glucose",
    "smoke": "Smoker", "alco": "Alcohol", "active": "Physically Active",
    "bmi": "BMI", "pulse_pressure": "Pulse Pressure",
    "age_group": "Age Group", "high_risk_flag": "High Risk Flag",
}


# ══════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════
def init_engine(models_dir):
    """
    Load the scaler and every model that will load.

    Each model is loaded separately so one missing or damaged file does
    not disable all the others - a missing xgboost library used to make
    the whole prediction page report "no models available".
    """
    global _SCALER, _MODELS, _FEATURE_NAMES, _LOAD_WARNINGS, _MODELS_DIR
    _MODELS_DIR = models_dir
    _MODELS = {}
    _LOAD_WARNINGS = []

    # -- feature list (the order is the contract with the scaler) --
    feature_path = os.path.join(models_dir, "features.json")
    try:
        with open(feature_path) as f:
            _FEATURE_NAMES = json.load(f)
    except Exception:
        _FEATURE_NAMES = list(DEFAULT_FEATURES)
        _LOAD_WARNINGS.append("features.json missing - using the built-in default order.")

    # -- scaler --
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    try:
        with open(scaler_path, "rb") as f:
            _SCALER = pickle.load(f)
    except FileNotFoundError:
        _SCALER = None
        _LOAD_WARNINGS.append("scaler.pkl not found - please train the models first.")
        return
    except Exception as e:
        _SCALER = None
        _LOAD_WARNINGS.append(f"scaler.pkl could not be read ({type(e).__name__}).")
        return

    # -- models --
    for name, filename in MODEL_FILES.items():
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            _LOAD_WARNINGS.append(f"{name}: file not found (skipped).")
            continue
        try:
            with open(path, "rb") as f:
                _MODELS[name] = pickle.load(f)
        except ModuleNotFoundError as e:
            _LOAD_WARNINGS.append(
                f"{name}: needs the '{e.name}' library (pip install {e.name}). Skipped.")
        except Exception as e:
            _LOAD_WARNINGS.append(f"{name}: could not be loaded ({type(e).__name__}). Skipped.")

    log.info(f"ML engine ready: {len(_MODELS)}/{len(MODEL_FILES)} models loaded.")


def get_load_warnings():
    """
    Any problems hit while loading models, as a list of plain sentences.

    The diagnosis page shows these so a missing model is visible rather
    than silently reducing the ensemble.
    """
    return list(_LOAD_WARNINGS)


def is_ready():
    """True when at least one model and the scaler are available."""
    return _SCALER is not None and len(_MODELS) > 0


def available_models():
    """Names of the models that loaded successfully, filtered by config.json."""
    config = load_config()
    return [name for name in _MODELS if config.get(name, True)]


def get_feature_names():
    """
    The 15 feature names in the exact order the scaler expects.

    Read from models/features.json at startup, which training writes.
    """
    return list(_FEATURE_NAMES)


# ══════════════════════════════════════════════════════════════════
# CONFIG / RESULTS  (read from disk, never crash the caller)
# ══════════════════════════════════════════════════════════════════
def _read_json(path, default):
    """Read a JSON file, returning `default` instead of raising on any problem."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def load_config():
    """Which models are switched on. Defaults to all enabled."""
    default = {name: True for name in MODEL_FILES}
    if not _MODELS_DIR:
        return default
    return _read_json(os.path.join(_MODELS_DIR, "config.json"), default)


def load_results():
    """Evaluation metrics produced by train_models.py."""
    if not _MODELS_DIR:
        return {}
    return _read_json(os.path.join(_MODELS_DIR, "results.json"), {})


def best_model_name():
    """
    The trained classifier with the highest ROC AUC, or None.

    The ensemble entry in results.json is skipped on purpose: "best model"
    means the best of the five saved classifiers, and the ensemble is an
    average of them rather than a competitor. Its own numbers still appear
    on the Model Performance page.
    """
    results = load_results()
    candidates = [name for name in results if name in MODEL_FILES]
    if not candidates:
        return None
    return max(candidates, key=lambda name: results[name].get("auc", 0))


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  - must match train_models.py exactly
# ------------------------------------------------------------------
# THE 15 FEATURES, IN THE ONLY ORDER THAT IS CORRECT
#
#   #   name             meaning                          where from
#   --  ---------------  -------------------------------  ----------
#    0  age              whole years                      form
#    1  gender           0 = female, 1 = male             form
#    2  height           centimetres                      form
#    3  weight           kilograms                        form
#    4  ap_hi            systolic  blood pressure (mmHg)  form
#    5  ap_lo            diastolic blood pressure (mmHg)  form
#    6  cholesterol      1 normal, 2 above, 3 well above  form
#    7  gluc             1 normal, 2 above, 3 well above  form
#    8  smoke            0 = no, 1 = yes                  form
#    9  alco             0 = no, 1 = yes                  form
#   10  active           0 = no, 1 = yes                  form
#   11  bmi              weight / height_m^2, clipped     derived
#   12  pulse_pressure   ap_hi - ap_lo                    derived
#   13  age_group        0-4 band, see the bins below     derived
#   14  high_risk_flag   raised cholesterol AND ap_hi>=140  derived
#
# The first 11 are typed in by the user; the last 4 are computed here.
# The order is a POSITIONAL contract with models/scaler.pkl - the scaler
# stores one mean and one standard deviation per slot and has no idea
# what the columns are called. Swap two entries and every value after
# that point is scaled with the wrong statistics, so the app would still
# produce a confident-looking number that is simply wrong.
#
# The formulas below are copied from research/train_models.py. They must
# stay identical: the models learned "BMI around 26 is normal" from the
# training figures, so if this file computed BMI a different way the same
# patient would land somewhere else entirely in the model's view.
# ══════════════════════════════════════════════════════════════════
def compute_engineered_features(age, height, weight, ap_hi, ap_lo, cholesterol):
    """
    Build the four derived features from the raw measurements.

    Returns: (bmi, pulse_pressure, age_group, high_risk_flag)
    """
    # BMI = weight in kg divided by height in metres, squared
    bmi = round(weight / ((height / 100) ** 2), 2)
    # Clipped to the range the models actually saw during training
    bmi = max(Config.BMI_CLIP_MIN, min(Config.BMI_CLIP_MAX, bmi))

    # Pulse pressure = the gap between the two blood pressure readings
    pulse_pressure = ap_hi - ap_lo

    # Age bands - matches pandas cut(bins=[0,30,45,60,75,120]) in training
    if age <= 30:
        age_group = 0
    elif age <= 45:
        age_group = 1
    elif age <= 60:
        age_group = 2
    elif age <= 75:
        age_group = 3
    else:
        age_group = 4

    # Raised cholesterol combined with hypertensive systolic pressure
    high_risk_flag = int(cholesterol >= 2 and ap_hi >= 140)

    return bmi, pulse_pressure, age_group, high_risk_flag


def engineer_features(form_data):
    """
    Turn a submitted form into the 15 numbers the models expect.

    `form_data` is any dict-like object (request.form works directly).
    Raises ValueError if a field is missing or not a number - the caller
    turns that into a friendly flash message.

    Returns: (feature_vector, readable_dict)
    """
    def _int(key):
        """Read one form field as a whole number, or raise ValueError."""
        raw = form_data.get(key, "")
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"'{key}' is required.")
        return int(float(raw))

    def _float(key):
        """Read one form field as a decimal number, or raise ValueError."""
        raw = form_data.get(key, "")
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"'{key}' is required.")
        return float(raw)

    age = _int("age")
    gender = _int("gender")
    height = _int("height")
    weight = _float("weight")
    ap_hi = _int("ap_hi")
    ap_lo = _int("ap_lo")
    cholesterol = _int("cholesterol")
    gluc = _int("gluc")
    smoke = _int("smoke")
    alco = _int("alco")
    active = _int("active")

    bmi, pulse_pressure, age_group, high_risk_flag = compute_engineered_features(
        age, height, weight, ap_hi, ap_lo, cholesterol)

    # Order must match models/features.json exactly - the scaler matches
    # values by POSITION, not by name.
    vector = [age, gender, height, weight, ap_hi, ap_lo,
              cholesterol, gluc, smoke, alco, active,
              bmi, pulse_pressure, age_group, high_risk_flag]

    readable = {
        "age": age, "gender": gender, "height": height, "weight": weight,
        "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol,
        "gluc": gluc, "smoke": smoke, "alco": alco, "active": active,
        "bmi": bmi, "pulse_pressure": pulse_pressure,
        "age_group": age_group, "high_risk_flag": high_risk_flag,
    }
    return vector, readable


# ══════════════════════════════════════════════════════════════════
# SERVER-SIDE VALIDATION  - never trust the browser
# ══════════════════════════════════════════════════════════════════
# Medically plausible ranges - defined once, in config.py
FIELD_RANGES = Config.FIELD_RANGES
CHOICE_FIELDS = Config.CHOICE_FIELDS


def validate_form(form_data):
    """
    Check every submitted value before it reaches a model.

    Every rule here is also applied in the browser by diagnosis.js, but a
    form can always be posted directly, so THIS is the real gate.

    Returns: (is_valid, list_of_error_messages)
    """
    import re as _re
    errors = []

    # -- patient name: letters, spaces, hyphens and apostrophes only --
    name = (form_data.get("patient_name") or "").strip()
    if not name:
        errors.append("Patient name is required.")
    elif len(name) < 2:
        errors.append("Patient name must be at least 2 characters.")
    elif len(name) > 60:
        errors.append("Patient name cannot be longer than 60 characters.")
    elif not _re.fullmatch(r"[A-Za-z .'\-]+", name):
        errors.append("Patient name may only contain letters, spaces, "
                      "hyphens and apostrophes.")

    # -- patient reference: optional, but must be a proper PT- reference --
    # House format is PT- followed by 3 to 12 digits, e.g. PT-1001.
    ref = (form_data.get("patient_id") or "").strip()
    if ref:
        if not ref.upper().startswith("PT-"):
            errors.append("Patient ID must start with 'PT-', for example PT-1001.")
        elif not _re.fullmatch(r"PT-\d{3,12}", ref.upper()):
            errors.append("Patient ID must be 'PT-' followed by 3 to 12 digits, "
                          "for example PT-1001.")

    # -- clinical notes: length only, the text itself is free-form --
    notes = form_data.get("notes") or ""
    if len(notes) > 500:
        errors.append("Clinical notes cannot be longer than 500 characters.")

    # -- each measurement present, numeric and inside its range --
    numbers = {}
    for field, (low, high, label) in FIELD_RANGES.items():
        raw = str(form_data.get(field, "")).strip()
        if raw == "":
            errors.append(f"{label} is required.")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue
        if value != value or value in (float("inf"), float("-inf")):
            errors.append(f"{label} is not a valid number.")
            continue
        if not (low <= value <= high):
            errors.append(f"{label} must be between {low} and {high}.")
            continue
        numbers[field] = value

    # Age, height and blood pressure are always whole numbers in practice
    for whole_field, label in [("age", "Age"), ("height", "Height"),
                               ("ap_hi", "Systolic blood pressure"),
                               ("ap_lo", "Diastolic blood pressure")]:
        if whole_field in numbers and numbers[whole_field] != int(numbers[whole_field]):
            errors.append(f"{label} must be a whole number.")

    # -- dropdowns must hold one of their allowed values --
    for field, allowed in CHOICE_FIELDS.items():
        raw = form_data.get(field, "")
        try:
            value = int(float(raw))
        except (TypeError, ValueError):
            errors.append(f"'{field}' has an invalid value.")
            continue
        if value not in allowed:
            errors.append(f"'{field}' has an invalid value.")

    # Stop here if anything above failed - the checks below need real numbers
    if errors:
        return False, errors

    ap_hi, ap_lo = numbers["ap_hi"], numbers["ap_lo"]
    height, weight = numbers["height"], numbers["weight"]

    # -- systolic must exceed diastolic in a living patient --
    if ap_hi <= ap_lo:
        errors.append(
            f"Systolic blood pressure ({int(ap_hi)}) must be higher than "
            f"diastolic ({int(ap_lo)}). The two readings may be swapped.")
        return False, errors

    # -- pulse pressure has to be physiologically plausible --
    pulse_pressure = ap_hi - ap_lo
    if pulse_pressure < Config.PULSE_PRESSURE_MIN:
        errors.append(f"Pulse pressure is only {int(pulse_pressure)} mmHg. "
                      f"A gap below {Config.PULSE_PRESSURE_MIN} mmHg is not plausible.")
    elif pulse_pressure > Config.PULSE_PRESSURE_MAX:
        errors.append(f"Pulse pressure is {int(pulse_pressure)} mmHg. "
                      f"A gap above {Config.PULSE_PRESSURE_MAX} mmHg is not plausible.")

    # -- BMI outside this range means height or weight was mistyped --
    bmi = weight / ((height / 100) ** 2)
    if bmi < Config.BMI_PLAUSIBLE_MIN:
        errors.append(f"The height and weight give a BMI of {bmi:.1f}, "
                      f"below the survivable range. Please re-check them.")
    elif bmi > Config.BMI_PLAUSIBLE_MAX:
        errors.append(f"The height and weight give a BMI of {bmi:.1f}, "
                      f"above the plausible range. Please re-check them.")

    return (len(errors) == 0), errors


# ══════════════════════════════════════════════════════════════════
# PREDICTION
# ------------------------------------------------------------------
# HOW ONE ASSESSMENT ACTUALLY RUNS
#
# 1. SCALING.  The raw numbers are on wildly different scales - age is
#    around 50, height around 170, smoke is 0 or 1. Logistic Regression
#    and SVM would let height dominate simply because its numbers are
#    bigger. Training therefore fitted a StandardScaler that rewrites
#    every column as "how many standard deviations from the mean".
#
#    That scaler is REUSED here, never refitted. It is the same object
#    that was pickled during training, so a patient is measured against
#    the training population - exactly what the models learned on. Fitting
#    a fresh scaler on one patient would be meaningless (a single row has
#    no spread) and would feed the models numbers they have never seen.
#
# 2. THE MODELS.  Each classifier was unpickled once at startup by
#    init_engine(), so a prediction costs no disk access. predict_proba()
#    returns [P(class 0), P(class 1)]; we keep index [1], the probability
#    of cardiovascular disease being PRESENT, as a number from 0 to 1.
#
# 3. THE THRESHOLD.  A probability is not a decision. Every model - and
#    the ensemble - calls a patient "at risk" at or above
#    Config.POSITIVE_THRESHOLD (0.40, the SCREENING threshold).
#
#    Why 0.40 and not the textbook 0.50: in a screening tool a missed
#    disease (false negative) is far more costly than an unnecessary
#    check-up (false positive), so the cut-off is deliberately lowered
#    to catch more real cases. The full rationale lives with the number
#    itself, in config.py.
#
#    The threshold is applied HERE, uniformly, to every model's
#    probability - deliberately NOT via each model's own predict(),
#    which is hard-wired to 0.50 inside scikit-learn. Applying one
#    shared cut-off in one place means the ensemble verdict, every
#    individual vote, and the "N of 5 models predict At Risk" count all
#    use the same rule, and changing the config value really does
#    change the decision everywhere.
#
#    0.40 also equals Config.RISK_LOW_MAX, so the yes/no verdict and
#    the coloured Low/Moderate/High band can never contradict each
#    other: "at risk" always lands in Moderate or High.
#
# 4. OUTPUT.  Every enabled model votes, and the ensemble probability is
#    the plain average of their probabilities. Keeping the individual
#    votes lets the interface show agreement - five models out of five
#    saying "at risk" is a very different message from three out of five.
# ══════════════════════════════════════════════════════════════════
def predict(feature_vector, model_choice="Ensemble Voting (All Models)"):
    """
    Run the prediction.

    Returns a dict with:
        probability      final risk probability (0-1)
        predicted_class  0 or 1
        model_used       label stored in the database
        votes            {model_name: {"probability": .., "predicted": ..}}
        models_at_risk   how many models said "at risk"
        total_models     how many models ran
    Raises RuntimeError if the engine is not ready.
    """
    if not is_ready():
        raise RuntimeError("No trained models are available.")

    enabled = available_models()
    if not enabled:
        raise RuntimeError("Every model has been switched off in the configuration.")

    # The scaler was fitted on a named DataFrame, so feed it one here too.
    # Passing a bare list still works but makes scikit-learn warn on every
    # single prediction, which floods the server log.
    try:
        import pandas as pd
        frame = pd.DataFrame([feature_vector], columns=_FEATURE_NAMES)
        scaled = _SCALER.transform(frame)
    except Exception:
        scaled = _SCALER.transform([feature_vector])

    votes = {}
    for name in enabled:
        model = _MODELS[name]
        probability = float(model.predict_proba(scaled)[0][1])
        votes[name] = {
            "probability": probability,
            # One shared screening cut-off for every model, applied here
            # rather than via model.predict() (which is hard-wired to 0.50
            # inside scikit-learn). See the THRESHOLD note above.
            "predicted": 1 if probability >= Config.POSITIVE_THRESHOLD else 0,
            "short": MODEL_SHORT.get(name, name[:3]),
        }

    if model_choice.startswith("Ensemble") or model_choice not in votes:
        # Average the probabilities of every enabled model
        probability = sum(vote["probability"] for vote in votes.values()) / len(votes)
        predicted_class = 1 if probability >= Config.POSITIVE_THRESHOLD else 0
        model_used = Config.ENSEMBLE_NAME
    else:
        probability = votes[model_choice]["probability"]
        predicted_class = votes[model_choice]["predicted"]
        model_used = model_choice

    return {
        "probability": probability,
        "predicted_class": predicted_class,
        "model_used": model_used,
        "votes": votes,
        "models_at_risk": sum(1 for vote in votes.values() if vote["predicted"] == 1),
        "total_models": len(votes),
    }


# ══════════════════════════════════════════════════════════════════
# PRESENTATION HELPERS
# ══════════════════════════════════════════════════════════════════
def get_risk_level(probability):
    """Turn a probability into a band, a colour class and a sentence."""
    if probability < Config.RISK_LOW_MAX:
        return {"label": "LOW RISK", "css": "risk-low", "colour": "#10b981",
                "emoji": "😊",          # smiling face
                "note": "Great news - no significant cardiovascular risk pattern detected."}
    if probability < Config.RISK_MODERATE_MAX:
        return {"label": "MODERATE RISK", "css": "risk-moderate", "colour": "#f59e0b",
                "emoji": "😐",          # neutral face
                "note": "Some risk indicators present. Lifestyle review recommended."}
    return {"label": "HIGH RISK", "css": "risk-high", "colour": "#ef4444",
            "emoji": "😟",              # worried face
            "note": "Strong risk indicators detected. Clinical review advised."}


def get_bmi_category(bmi):
    """Standard World Health Organisation BMI bands."""
    if bmi < 18.5:
        return {"label": "Underweight", "colour": "#3b82f6", "emoji": "🧍"}
    if bmi < 25:
        return {"label": "Normal weight", "colour": "#10b981", "emoji": "👍"}
    if bmi < 30:
        return {"label": "Overweight", "colour": "#f59e0b", "emoji": "⚠"}
    return {"label": "Obese", "colour": "#ef4444", "emoji": "🔴"}


def get_agreement_summary(models_at_risk, total_models):
    """Describe how strongly the models agree with each other."""
    if not total_models:
        return {"text": "No models ran.", "css": "agree-weak"}
    share = models_at_risk / total_models
    if share >= 0.8 or share <= 0.2:
        return {"text": "✅ Strong agreement between the models",
                "css": "agree-strong"}
    if 0.4 <= share <= 0.6:
        return {"text": "⚠ The models disagree - treat this with caution",
                "css": "agree-weak"}
    return {"text": "🤔 Moderate agreement", "css": "agree-moderate"}


def get_lifestyle_tips(risk_label):
    """General, non-personalised advice matched to the risk band."""
    if risk_label == "HIGH RISK":
        return [
            "See a doctor promptly for a full cardiovascular assessment (ECG, blood panel).",
            "Monitor blood pressure daily and record the readings for your doctor.",
            "Cut added salt to under 5 g per day; replace fried food with grilled or baked.",
            "If you smoke, stopping is the single largest risk reduction available.",
            "Begin exercise only with medical clearance, then build up gradually.",
            "Reduce alcohol sharply, or stop.",
        ]
    if risk_label == "MODERATE RISK":
        return [
            "Book a routine check-up within the next few months.",
            "Eat more vegetables, fruit, whole grains and oily fish; less red and processed meat.",
            "Aim for 150 minutes of moderate activity per week - "
            "a brisk 30-minute walk, five days.",
            "Losing even 5% of body weight measurably lowers blood pressure.",
            "If you smoke, this is the highest-value change you can make.",
            "Target 7-8 hours of sleep; poor sleep raises blood pressure.",
        ]
    return [
        "Keep up the current routine - the indicators look healthy.",
        "Maintain at least 150 minutes of moderate activity per week.",
        "Stay with a balanced diet; keep salt and saturated fat low.",
        "An annual blood pressure and cholesterol check is still worthwhile.",
        "Avoid smoking and keep alcohol within recommended limits.",
    ]


def get_feature_importances(model_name="Random Forest"):
    """
    Feature importances for the model-performance page.

    Used instead of computing SHAP on every request, which would make the
    page far too slow. Returns a list of {feature, importance} sorted high
    to low, or an empty list if the model does not expose importances.
    """
    model = _MODELS.get(model_name)
    if model is None:
        for fallback in ("XGBoost", "Random Forest", "Decision Tree"):
            model = _MODELS.get(fallback)
            if model is not None:
                model_name = fallback
                break
    if model is None:
        return [], None

    values = None
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = [abs(c) for c in model.coef_[0]]
    elif hasattr(model, "calibrated_classifiers_"):
        try:
            base = model.calibrated_classifiers_[0].estimator
            values = [abs(c) for c in base.coef_[0]]
        except Exception:
            return [], None

    if values is None or len(values) != len(_FEATURE_NAMES):
        return [], None

    total = sum(values) or 1
    rows = [{"feature": FEATURE_LABELS.get(name, name),
             "importance": round(float(value) / total, 4)}
            for name, value in zip(_FEATURE_NAMES, values)]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows, model_name
