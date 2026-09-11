"""
Heart Disease Prediction - Application configuration
==========================================
Central place for every setting the app needs. Values that matter for
security are read from environment variables, with a development fallback
so the project still runs straight after cloning.
"""

import os
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # ── Security ──────────────────────────────────────────────────
    # In production set a real value:   set HDP_SECRET_KEY=...
    # The fallback keeps development working but is NOT secret.
    SECRET_KEY = os.environ.get(
        "HDP_SECRET_KEY",
        "dev-only-insecure-key-change-me-in-production"
    )

    # Session cookie hardening
    SESSION_COOKIE_HTTPONLY = True        # JavaScript cannot read the cookie
    SESSION_COOKIE_SAMESITE = "Lax"       # blocks most cross-site request forgery
    SESSION_COOKIE_SECURE = os.environ.get("HDP_HTTPS", "0") == "1"
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)

    # ── Paths ─────────────────────────────────────────────────────
    BASE_DIR = BASE_DIR
    DB_PATH = os.path.join(BASE_DIR, "heart_disease.db")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    # The raw dataset lives with the training code in research/
    DATA_PATH = os.path.join(BASE_DIR, "research", "heart.csv")
    STATIC_IMG_DIR = os.path.join(BASE_DIR, "static", "img")

    # ── Risk banding ──────────────────────────────────────────────
    # Where the coloured Low / Moderate / High badge switches over.
    # ml_engine.get_risk_level() reads these, so changing a number here
    # really does change the banding everywhere.
    #
    # These bands are PRESENTATION heuristics, not clinical guidelines:
    # they choose the colour and wording shown to the user. What makes
    # them defensible is that the Low/Moderate boundary is deliberately
    # the same number as POSITIVE_THRESHOLD below, so the badge and the
    # yes/no verdict can never contradict each other.
    RISK_LOW_MAX = 0.40           # below this  -> Low
    RISK_MODERATE_MAX = 0.70      # below this  -> Moderate, above -> High

    # ── Decision threshold ────────────────────────────────────────
    # A model calls a patient "at risk" at or above this probability.
    #
    # 0.40 rather than the textbook 0.50, and the choice is deliberate.
    # Heart Disease Prediction is a SCREENING tool: its job is to flag people who
    # should see a doctor, not to diagnose. In screening the two kinds
    # of mistake are not equally costly -
    #
    #   * a false positive sends a healthy person for a check-up they
    #     did not need (a wasted appointment);
    #   * a false negative sends a sick person home reassured (a missed
    #     disease, potentially fatal).
    #
    # At the standard 0.50 cut-off the models miss roughly 31% of the
    # patients who really have the disease (recall ~0.69 on the held-out
    # test set). Lowering the cut-off to 0.40 trades some precision for
    # substantially higher recall - more false alarms, fewer missed
    # cases - which is the correct trade for a screening context.
    #
    # 0.40 also equals RISK_LOW_MAX above, so "at risk" and the coloured
    # band now agree by construction: every "at risk" patient is shown
    # Moderate or High, and every "not at risk" patient is shown Low.
    # (Before this change a patient at 0.45 was classified "not at risk"
    # yet displayed a MODERATE badge - a genuine contradiction.)
    #
    # train_models.py reports every model's metrics at BOTH cut-offs
    # (0.50 for comparability with the literature, and this screening
    # threshold for how the app actually behaves) - see results.json.
    POSITIVE_THRESHOLD = 0.40

    # ── The 15 model features, in the order the scaler expects ────
    # This order is a CONTRACT with models/scaler.pkl. The scaler matches
    # values by POSITION, not by name, so reordering this list silently
    # feeds every value into the wrong slot. Only change it by retraining.
    FEATURE_ORDER = [
        # 11 raw measurements taken straight from the form
        "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active",
        # 4 features the app derives from those - see
        # ml_engine.compute_engineered_features()
        "bmi", "pulse_pressure", "age_group", "high_risk_flag",
    ]

    # Name under which the averaged-probability ensemble is reported in
    # models/results.json and on the Model Performance page. It is not a
    # trained model (no pickle) - ml_engine.predict() builds it at run time
    # by averaging the five probabilities - so it is deliberately NOT in
    # MODEL_FILES below.
    ENSEMBLE_NAME = "Ensemble Voting"

    # Which pickle holds which trained classifier
    MODEL_FILES = {
        "Logistic Regression": "logistic_regression.pkl",
        "Support Vector Machine (SVM)": "svm.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
    }

    # ── Medically plausible input ranges ──────────────────────────
    # Enforced by ml_engine.validate_form() on the server, and mirrored in
    # static/js/diagnosis.js for instant feedback in the browser.
    # field -> (minimum, maximum, label shown in the error message)
    FIELD_RANGES = {
        "age": (18, 120, "Age"),
        "height": (100, 250, "Height"),
        "weight": (20, 200, "Weight"),
        "ap_hi": (60, 250, "Systolic blood pressure"),
        "ap_lo": (40, 200, "Diastolic blood pressure"),
    }

    # Dropdowns: only these codes are ever valid
    CHOICE_FIELDS = {
        "gender": {0, 1},            # 0 = female, 1 = male
        "cholesterol": {1, 2, 3},    # 1 = normal, 2 = above, 3 = well above
        "gluc": {1, 2, 3},
        "smoke": {0, 1},
        "alco": {0, 1},
        "active": {0, 1},
    }

    # Cross-field limits that a simple min/max cannot express
    PULSE_PRESSURE_MIN = 20       # systolic minus diastolic, in mmHg
    PULSE_PRESSURE_MAX = 100
    BMI_PLAUSIBLE_MIN = 12        # outside this, height or weight is mistyped
    BMI_PLAUSIBLE_MAX = 60
    BMI_CLIP_MIN = 10             # the range the models were trained on
    BMI_CLIP_MAX = 70

    # Free-text limits on the assessment form
    PATIENT_NAME_MAX = 60
    PATIENT_ID_PATTERN = r"PT-\d{3,12}"   # house format, e.g. PT-1001
    NOTES_MAX = 500

    # ── Display labels ────────────────────────────────────────────
    # The 1-3 clinical scale used for cholesterol and glucose
    LEVEL_LABELS = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}

    # ── Application behaviour ─────────────────────────────────────

    # Rows shown per page in the audit log
    AUDIT_PAGE_SIZE = 25

    # Uploads (dataset replacement on the admin page)
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024      # 32 MB ceiling on any upload


class DevConfig(Config):
    DEBUG = True


class ProdConfig(Config):
    DEBUG = False


def get_config():
    """Pick a config class from the HDP_ENV environment variable."""
    if os.environ.get("HDP_ENV", "development").lower().startswith("prod"):
        return ProdConfig
    return DevConfig
