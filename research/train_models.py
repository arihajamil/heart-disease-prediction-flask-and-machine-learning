"""
Heart Disease Prediction FYP - Professional Model Training Pipeline
======================================================
Preprocessing Steps:
  1. Load & initial inspection
  2. Age conversion (days -> years)
  3. Duplicate row removal
  4. Physiologically impossible value removal (domain rules)
  5. IQR-based outlier clipping (Winsorization)
  6. Median imputation for missing values
  7. Feature engineering (derived clinical features)
  8. High-correlation feature pruning
  9. Train / Test split (stratified, 80/20)
 10. StandardScaler normalization
 11. Train 5 classifiers with class_weight='balanced', then score the
     averaged-probability ensemble the app actually uses
 12. Evaluate at two cut-offs (0.50 standard, 0.40 screening) plus
     calibration (Brier score + reliability curve) - see _evaluate()
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# This script lives in research/ but writes into the application's models/
# and static/img/ folders. Rather than recomputing those paths from __file__
# (which breaks the moment the file moves), they come from config.py - the
# single place every path in the project is defined.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config                                      # noqa: E402

# Key under which the averaged-probability ensemble is reported in
# results.json. Defined in config.py so ml_engine and the admin pages use
# the very same string.
ENSEMBLE_NAME = Config.ENSEMBLE_NAME

BASE_DIR   = Config.BASE_DIR
MODELS_DIR = Config.MODELS_DIR
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------
# HELPER - model evaluation
# -------------------------------------------------------------
def _evaluate(model, X_test_scaled, y_test, model_name,
              train_time_s=0.0, pred_time_ms=0.0):
    """
    Score one fitted model on the held-out test set.

    Thin wrapper: pulls the probabilities out of the model and hands them
    to _evaluate_proba(), which does all the work. Splitting it this way
    lets the ensemble - which has no model object of its own, only an
    average of five probability vectors - be scored by exactly the same
    code and reported in exactly the same format.
    """
    proba = model.predict_proba(X_test_scaled)[:, 1]
    return _evaluate_proba(proba, y_test, model_name,
                           train_time_s, pred_time_ms)


def _evaluate_proba(proba, y_test, model_name,
                    train_time_s=0.0, pred_time_ms=0.0):
    """
    Score one vector of P(disease) against the held-out test set.

    Metrics are reported at TWO decision cut-offs:

      * 0.50 - the textbook default, kept so the headline numbers stay
        comparable with published results on this dataset;
      * Config.POSITIVE_THRESHOLD (0.40) - the screening cut-off the app
        actually uses, where missing a real case costs more than a false
        alarm. These land under the "screening" key in results.json.

    Calibration is also measured (Brier score + a 10-bin reliability
    curve), because the app shows the raw probability to the user - a
    displayed "72% risk" is only honest if roughly 72 of every 100 such
    patients really have the disease.
    """
    # 0.50 is what scikit-learn's own predict() uses, so applying it to
    # the probabilities here gives identical headline numbers for every
    # saved model - and makes the ensemble comparable on the same footing.
    preds = (proba >= 0.5).astype(int)
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, proba)
    f1    = f1_score(y_test, preds)
    prec  = precision_score(y_test, preds)
    rec   = recall_score(y_test, preds)
    cm    = confusion_matrix(y_test, preds).tolist()

    # ── Metrics at the app's screening threshold ──────────────────
    screen_t     = Config.POSITIVE_THRESHOLD
    preds_screen = (proba >= screen_t).astype(int)
    screen = {
        "threshold":   screen_t,
        "accuracy":    round(accuracy_score(y_test, preds_screen), 6),
        "precision":   round(precision_score(y_test, preds_screen), 6),
        "recall":      round(recall_score(y_test, preds_screen), 6),
        "f1":          round(f1_score(y_test, preds_screen), 6),
        "conf_matrix": confusion_matrix(y_test, preds_screen).tolist(),
    }

    # ── Calibration: is a predicted 72% really a 72%? ─────────────
    # Brier score = mean squared error between predicted probability and
    # the 0/1 outcome (lower is better; 0.25 is what always-guessing-50%
    # scores). The reliability curve bins patients by predicted
    # probability and compares each bin's prediction with the observed
    # disease rate - a well-calibrated model hugs the diagonal.
    brier = brier_score_loss(y_test, proba)
    frac_pos, mean_pred = calibration_curve(y_test, proba,
                                            n_bins=10, strategy="uniform")
    cal_data = {
        "mean_predicted":    [round(float(v), 5) for v in mean_pred],
        "fraction_positive": [round(float(v), 5) for v in frac_pos],
    }

    # ── ROC Curve data (downsampled to 100 pts for compact JSON) ──
    fpr, tpr, _ = roc_curve(y_test, proba)
    step = max(1, len(fpr) // 100)
    roc_data = {"fpr": [round(float(v), 5) for v in fpr[::step]],
                "tpr": [round(float(v), 5) for v in tpr[::step]]}

    # ── Precision-Recall Curve data ───────────────────────────────
    pr_p, pr_r, _ = precision_recall_curve(y_test, proba)
    avg_prec      = average_precision_score(y_test, proba)
    step_pr = max(1, len(pr_p) // 100)
    pr_data = {"precision": [round(float(v), 5) for v in pr_p[::step_pr]],
               "recall":    [round(float(v), 5) for v in pr_r[::step_pr]],
               "avg_precision": round(float(avg_prec), 6)}

    print(f"  [{model_name}] Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  "
          f"Prec={prec:.4f}  Rec={rec:.4f}  Brier={brier:.4f}  "
          f"Rec@{screen_t:.2f}={screen['recall']:.4f}  "
          f"TrainTime={train_time_s:.2f}s  PredTime={pred_time_ms:.1f}ms/1k")
    return {
        "accuracy":         round(acc,  6),
        "auc":              round(auc,  6),
        "f1":               round(f1,   6),
        "precision":        round(prec, 6),
        "recall":           round(rec,  6),
        "brier":            round(float(brier), 6),
        "screening":        screen,
        "calibration":      cal_data,
        "training_time_s":  round(train_time_s, 4),
        "pred_time_ms":     round(pred_time_ms, 4),
        "conf_matrix":      cm,
        "report":           classification_report(y_test, preds, output_dict=True),
        "roc_curve":        roc_data,
        "pr_curve":         pr_data,
    }

# -------------------------------------------------------------
# HELPER - K-Fold Cross Validation (Stratified, 5-fold)
# -------------------------------------------------------------
KFOLD_N = 5

def _kfold_cv(model, X_raw, y, model_name):
    """
    Run StratifiedKFold CV and return per-fold + mean/std stats.

    IMPORTANT - this receives UNSCALED data on purpose.

    Previously the data was scaled once up front and the already-fitted
    scaler was reused for every fold. That leaks information: each fold's
    validation rows were standardised using statistics calculated from the
    other folds, which flatters the scores.

    Wrapping the scaler and the model in a Pipeline means StandardScaler is
    re-fitted from scratch inside every fold, using only that fold's training
    rows. This is the statistically correct way to cross-validate and gives
    an honest estimate of how the model generalises.
    """
    skf     = StratifiedKFold(n_splits=KFOLD_N, shuffle=True, random_state=42)
    scoring = {
        "accuracy":  "accuracy",
        "roc_auc":   "roc_auc",
        "f1":        "f1",
        "precision": "precision",
        "recall":    "recall",
    }
    # Scaler is re-fitted inside each fold - no leakage between folds
    cv_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", clone(model)),
    ])
    cv_res = cross_validate(cv_pipeline, X_raw, y,
                            cv=skf, scoring=scoring,
                            return_train_score=False, n_jobs=-1)

    # Round each fold score so the console table stays readable
    def _fmt(arr): return [round(float(v), 6) for v in arr]

    result = {
        "k": KFOLD_N,
        "folds": {
            "accuracy":  _fmt(cv_res["test_accuracy"]),
            "auc":       _fmt(cv_res["test_roc_auc"]),
            "f1":        _fmt(cv_res["test_f1"]),
            "precision": _fmt(cv_res["test_precision"]),
            "recall":    _fmt(cv_res["test_recall"]),
        },
        "mean": {
            "accuracy":  round(float(cv_res["test_accuracy"].mean()),  6),
            "auc":       round(float(cv_res["test_roc_auc"].mean()),   6),
            "f1":        round(float(cv_res["test_f1"].mean()),        6),
            "precision": round(float(cv_res["test_precision"].mean()), 6),
            "recall":    round(float(cv_res["test_recall"].mean()),    6),
        },
        "std": {
            "accuracy":  round(float(cv_res["test_accuracy"].std()),  6),
            "auc":       round(float(cv_res["test_roc_auc"].std()),   6),
            "f1":        round(float(cv_res["test_f1"].std()),        6),
            "precision": round(float(cv_res["test_precision"].std()), 6),
            "recall":    round(float(cv_res["test_recall"].std()),    6),
        },
    }
    print(f"    K-Fold {KFOLD_N} [{model_name}]  "
          f"Acc={result['mean']['accuracy']:.4f}(±{result['std']['accuracy']:.4f})  "
          f"AUC={result['mean']['auc']:.4f}(±{result['std']['auc']:.4f})")
    return result


# -------------------------------------------------------------
# STEP 1 - load dataset
# -------------------------------------------------------------

def _load(data_path):
    """Read the raw CSV exactly as supplied, before any cleaning."""
    df = pd.read_csv(data_path, sep=None, engine="python")
    print(f"  Loaded: {df.shape[0]:,} rows ? {df.shape[1]} cols")
    return df


# -------------------------------------------------------------
# STEP 2 - basic housekeeping
# -------------------------------------------------------------
def _basic_clean(df):
    # Drop useless id/index columns
    """
    First pass over the raw file: drop the id column, convert age from days
    to whole years, and shift cholesterol/gluc from 0-2 to the 1-3 scale.

    That shift is the important one. The published file stores these two
    columns 0-indexed, while every downstream rule (and the web form) uses
    1 = normal, 2 = above normal, 3 = well above normal. Filtering on the
    1-3 values without shifting first discarded about 91% of the rows.
    """
    drop_cols = [c for c in ["Unnamed: 0", "id"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Age: days -> years
    if "age" in df.columns and df["age"].max() > 1000:
        df["age"] = (df["age"] / 365.25).round(0).astype(int)
        print("  Age converted from days to years.")

    # Cholesterol / glucose: normalise to the clinical 1-3 scale.
    #
    # This dataset stores them 0-indexed (0=normal, 1=above, 2=well above),
    # but the standard cardiovascular encoding - and the diagnosis form in
    # app.py - both use 1=normal, 2=above, 3=well above.
    #
    # Without this shift the domain filter below (which keeps only values
    # 1, 2 and 3) silently deleted every patient with normal cholesterol or
    # normal glucose: 58,242 rows, almost all of them healthy people. That
    # single mismatch cut the dataset from 70,000 rows to 6,583 and skewed
    # the class balance from 51% positive to 66% positive.
    for col in ["cholesterol", "gluc"]:
        if col in df.columns and df[col].min() == 0:
            df[col] = df[col] + 1
            print(f"  '{col}' shifted from 0-indexed to the clinical 1-3 scale.")

    return df


# -------------------------------------------------------------
# STEP 3 - remove duplicate rows
# -------------------------------------------------------------
def _remove_duplicates(df):
    """Drop exact duplicate patients so the same person cannot be in both train and test."""
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    removed = before - len(df)
    print(f"  Duplicates removed: {removed:,}  (remaining: {len(df):,})")
    return df


# -------------------------------------------------------------
# STEP 4 - domain / physiological rules
# -------------------------------------------------------------
def _domain_filter(df):
    """
    Keep only physiologically possible rows.

    These are not statistical outlier rules - each bound is a value a human
    body cannot take, so a row outside them is a recording error.
    """
    before = len(df)
    masks = []

    if "age" in df.columns:
        masks.append((df["age"] >= 1) & (df["age"] <= 120))

    # Blood pressure (ap_hi / ap_lo) - cardiovascular dataset columns
    if "ap_hi" in df.columns:
        masks.append((df["ap_hi"] >= 60) & (df["ap_hi"] <= 250))
    if "ap_lo" in df.columns:
        masks.append((df["ap_lo"] >= 40) & (df["ap_lo"] <= 200))
    if "ap_hi" in df.columns and "ap_lo" in df.columns:
        masks.append(df["ap_hi"] > df["ap_lo"])   # systolic must be > diastolic

    # Height & weight (cm / kg)
    if "height" in df.columns:
        masks.append((df["height"] >= 100) & (df["height"] <= 250))
    if "weight" in df.columns:
        masks.append((df["weight"] >= 20)  & (df["weight"] <= 300))

    # Cholesterol / glucose encoded as 1-3 (not outside range)
    for col in ["cholesterol", "gluc"]:
        if col in df.columns:
            masks.append(df[col].isin([1, 2, 3]))

    if masks:
        combined = masks[0]
        for mask in masks[1:]:
            combined = combined & mask
        df = df[combined].reset_index(drop=True)

    removed = before - len(df)
    print(f"  Domain-invalid rows removed: {removed:,}  (remaining: {len(df):,})")
    return df


# -------------------------------------------------------------
# STEP 5 - IQR-based Winsorization (outlier clipping)
# -------------------------------------------------------------
def _winsorize_iqr(df, num_cols, factor=1.5):
    """
    Pull extreme-but-possible values back to the 1.5xIQR fences.

    Clipping rather than deleting: the patient is real, only the reading is
    implausible, so the row is kept and the number is capped.
    """
    clipped_info = {}
    for col in num_cols:
        if col not in df.columns:
            continue
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_out > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            clipped_info[col] = int(n_out)
    if clipped_info:
        print(f"  IQR-clipped outliers: {clipped_info}")
    else:
        print("  No outliers clipped (all values within IQR bounds).")
    return df


# -------------------------------------------------------------
# STEP 6 - Median imputation
# -------------------------------------------------------------
def _impute_median(df, num_cols):
    """Fill any remaining gaps with the column median, which no outlier can drag."""
    missing = df[num_cols].isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values found - imputation skipped.")
    else:
        for col in missing.index:
            med = df[col].median()
            # Assign back directly. The old df[col].fillna(..., inplace=True)
            # is chained assignment, which pandas 3 turns into a silent no-op.
            df[col] = df[col].fillna(med)
            print(f"  Imputed '{col}': {missing[col]} NaNs -> median={med:.2f}")
    return df


# -------------------------------------------------------------
# STEP 7 - Feature engineering
# -------------------------------------------------------------
def _feature_engineering(df):
    """
    Derive the 4 extra features from the 11 measured ones.

    These must match ml_engine.compute_engineered_features() exactly, or the
    app would feed the models something different from what they learned.
    """
    new_feats = []

    # BMI  (if height in cm, weight in kg)
    if "height" in df.columns and "weight" in df.columns:
        df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(2)
        # Clip to sane range
        df["bmi"] = df["bmi"].clip(10, 70)
        new_feats.append("bmi")

    # Pulse pressure  (systolic - diastolic)
    if "ap_hi" in df.columns and "ap_lo" in df.columns:
        df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
        new_feats.append("pulse_pressure")

    # Age groups (binned)
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 30, 45, 60, 75, 120],
            labels=[0, 1, 2, 3, 4],
        ).astype(int)
        new_feats.append("age_group")

    # Combined risk flag  (high cholesterol AND hypertensive)
    if "cholesterol" in df.columns and "ap_hi" in df.columns:
        df["high_risk_flag"] = (
            (df["cholesterol"] >= 2) & (df["ap_hi"] >= 140)
        ).astype(int)
        new_feats.append("high_risk_flag")

    if new_feats:
        print(f"  Engineered features added: {new_feats}")
    return df


# -------------------------------------------------------------
# STEP 8 - Drop highly correlated features  (|r| >= 0.90)
# -------------------------------------------------------------
def _drop_high_correlation(df, feature_cols, threshold=0.90):
    """
    Remove one of any pair of near-identical columns.

    Two columns carrying the same information add no signal but do make the
    linear models unstable.
    """
    corr_matrix = df[feature_cols].corr().abs()
    upper_tri   = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_tri.columns
               if any(upper_tri[col] >= threshold)]
    if to_drop:
        print(f"  High-correlation features dropped (|r|>={threshold}): {to_drop}")
        feature_cols = [c for c in feature_cols if c not in to_drop]
    else:
        print(f"  No features dropped by correlation threshold ({threshold}).")
    return feature_cols


# -------------------------------------------------------------
# HELPER - SHAP summary image (written once, at training time)
# -------------------------------------------------------------
STATIC_IMG_DIR = Config.STATIC_IMG_DIR


def _generate_shap_image(model, X_train_scaled, feature_cols, log):
    """
    Save a SHAP summary plot to static/img/shap_summary.png.

    Doing this here rather than in the web app keeps page loads fast: SHAP
    takes tens of seconds on this dataset, which would be unusable on a
    request. Any failure is reported and ignored - a missing picture must
    never stop the models being saved.
    """
    out_path = os.path.join(STATIC_IMG_DIR, "shap_summary.png")
    try:
        os.makedirs(STATIC_IMG_DIR, exist_ok=True)

        import matplotlib
        matplotlib.use("Agg")          # no display needed on a server
        import matplotlib.pyplot as plt
        import shap

        # A small sample is plenty for a global summary and keeps it quick
        sample = pd.DataFrame(X_train_scaled, columns=feature_cols)
        sample = sample.sample(min(500, len(sample)), random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        # Newer SHAP returns (rows, features, classes) for classifiers;
        # older versions return a list of one array per class.
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, sample, feature_names=feature_cols,
                          show=False, plot_size=None)
        plt.title("SHAP Feature Impact - Random Forest", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close("all")

        log(f"  SHAP summary saved -> static/img/shap_summary.png")

    except ImportError:
        log("  (shap or matplotlib not installed - SHAP image skipped)")
    except Exception as e:
        log(f"  WARNING: SHAP image could not be generated: {type(e).__name__}: {e}")


# -------------------------------------------------------------
# MAIN - train_all()
# -------------------------------------------------------------
def train_all(data_path=None):
    """
    The full pipeline, start to finish.

        load -> clean -> deduplicate -> domain filter -> winsorize
        -> impute -> engineer features -> drop correlated
        -> stratified split -> fit the scaler on TRAIN ONLY
        -> train 5 classifiers -> evaluate -> save

    Fitting the scaler after the split is what keeps the test set honest:
    if it saw the test rows, their statistics would leak into training.

    Every model uses random_state=42, so re-running this script reproduces
    the same .pkl files byte for byte.
    """
    t0 = time.time()
    report_lines = []

    # Print progress and keep a copy, so the same text can be saved as a report
    def log(msg):
        print(msg)
        report_lines.append(msg)

    # -- resolve dataset --------------------------------------
    if data_path is None:
        custom    = os.path.join(BASE_DIR, "custom_dataset.csv")
        data_path = custom if os.path.exists(custom) else Config.DATA_PATH
    log(f"\n" + "=" * 60)
    log(f" Heart Disease Prediction - Professional Preprocessing Pipeline")
    log("=" * 60)
    # Log the dataset location RELATIVE to the project, never absolute.
    # This line ends up verbatim in models/preprocess_report.txt, which is
    # committed and shipped - an absolute path would leak the developer's
    # drive and folder layout into every training report.
    try:
        shown_path = os.path.relpath(data_path, BASE_DIR)
    except ValueError:          # different drive on Windows - fall back
        shown_path = os.path.basename(data_path)
    log(f"\n[STEP 1] Loading dataset: {shown_path}")
    df = _load(data_path)
    rows_original = len(df)

    log(f"\n[STEP 2] Basic housekeeping")
    df = _basic_clean(df)

    log(f"\n[STEP 3] Removing duplicate rows")
    df = _remove_duplicates(df)

    log(f"\n[STEP 4] Applying physiological domain filters")
    df = _domain_filter(df)

    # Determine target column
    target_col   = "cardio" if "cardio" in df.columns else "target"
    feature_cols = [c for c in df.columns if c != target_col]

    # Convert to numeric
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df[target_col].astype(int)

    log(f"\n[STEP 5] IQR Winsorization on numeric features")
    continuous_cols = [c for c in feature_cols
                       if df[c].nunique() > 10]          # skip binary/ordinal
    df = _winsorize_iqr(df, continuous_cols)

    log(f"\n[STEP 6] Median imputation for missing values")
    df = _impute_median(df, feature_cols)

    log(f"\n[STEP 7] Feature engineering")
    df = _feature_engineering(df)
    feature_cols = [c for c in df.columns if c != target_col]   # refresh list

    log(f"\n[STEP 8] Correlation-based feature pruning")
    feature_cols = _drop_high_correlation(df, feature_cols, threshold=0.90)

    # -- Final dataset summary --------------------------------
    X = df[feature_cols]
    y = df[target_col].astype(int)

    log("\n" + "-" * 60)
    log(" Dataset after preprocessing:")
    log(f"   Rows  : {rows_original:,} -> {len(df):,}")
    log(f"   Features used : {len(feature_cols)} -> {feature_cols}")
    class_counts = y.value_counts().to_dict()
    log(f"   Class distribution: {class_counts}")
    imbalance_ratio = max(class_counts.values()) / max(min(class_counts.values()), 1)
    balance_note = "balanced model weights applied" if imbalance_ratio > 1.5 else "balanced"
    log(f"   Class imbalance ratio: {imbalance_ratio:.2f}x ({balance_note})")
    log("-" * 60)

    # -- Train / Test split -----------------------------------
    log(f"\n[STEP 9] Train/Test split (80/20, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # -- Scaling ----------------------------------------------
    log(f"\n[STEP 10] StandardScaler normalization")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "features.json"), "w") as f:
        json.dump(feature_cols, f)
    log("  Scaler & feature list saved.")

    # -- Model training ---------------------------------------
    log(f"\n[STEP 11] Training classifiers")
    results = {}

    # 1. Logistic Regression
    log("  [1/5] Logistic Regression")
    lr = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=42
    )
    _t0 = time.time(); lr.fit(X_train_s, y_train); _train_t = time.time() - _t0
    _t0 = time.time(); lr.predict_proba(X_test_s[:1000]); _pred_t = (time.time()-_t0)*1000
    with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "wb") as f:
        pickle.dump(lr, f)
    results["Logistic Regression"] = _evaluate(lr, X_test_s, y_test, "LR",
                                               _train_t, _pred_t)

    # 2. SVM
    log("  [2/5] Support Vector Machine")
    svm_base = LinearSVC(max_iter=3000, C=1.0, class_weight="balanced", random_state=42)
    svm = CalibratedClassifierCV(svm_base, cv=3)
    _t0 = time.time(); svm.fit(X_train_s, y_train); _train_t = time.time() - _t0
    _t0 = time.time(); svm.predict_proba(X_test_s[:1000]); _pred_t = (time.time()-_t0)*1000
    with open(os.path.join(MODELS_DIR, "svm.pkl"), "wb") as f:
        pickle.dump(svm, f)
    results["Support Vector Machine (SVM)"] = _evaluate(svm, X_test_s, y_test, "SVM",
                                                        _train_t, _pred_t)

    # 3. Decision Tree
    log("  [3/5] Decision Tree")
    dt = DecisionTreeClassifier(
        max_depth=7, min_samples_split=20,
        class_weight="balanced", random_state=42
    )
    _t0 = time.time(); dt.fit(X_train_s, y_train); _train_t = time.time() - _t0
    _t0 = time.time(); dt.predict_proba(X_test_s[:1000]); _pred_t = (time.time()-_t0)*1000
    with open(os.path.join(MODELS_DIR, "decision_tree.pkl"), "wb") as f:
        pickle.dump(dt, f)
    results["Decision Tree"] = _evaluate(dt, X_test_s, y_test, "DT",
                                         _train_t, _pred_t)

    # 4. Random Forest
    log("  [4/5] Random Forest")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        min_samples_split=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    _t0 = time.time(); rf.fit(X_train_s, y_train); _train_t = time.time() - _t0
    _t0 = time.time(); rf.predict_proba(X_test_s[:1000]); _pred_t = (time.time()-_t0)*1000
    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "wb") as f:
        pickle.dump(rf, f)
    results["Random Forest"] = _evaluate(rf, X_test_s, y_test, "RF",
                                         _train_t, _pred_t)

    # 5. XGBoost / GradientBoosting
    log("  [5/5] XGBoost")
    neg  = int((y_train == 0).sum())
    pos  = int((y_train == 1).sum())
    scale_pw = neg / max(pos, 1)          # XGBoost handles imbalance via scale_pos_weight

    if XGB_AVAILABLE:
        # Note: use_label_encoder was removed in XGBoost 2.0, so it is no
        # longer passed here. Our labels are already 0/1 integers.
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pw,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        log("    (XGBoost not installed -> using GradientBoostingClassifier)")
        xgb = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
        )
    _t0 = time.time(); xgb.fit(X_train_s, y_train); _train_t = time.time() - _t0
    _t0 = time.time(); xgb.predict_proba(X_test_s[:1000]); _pred_t = (time.time()-_t0)*1000
    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "wb") as f:
        pickle.dump(xgb, f)
    results["XGBoost"] = _evaluate(xgb, X_test_s, y_test, "XGB", _train_t, _pred_t)
    results["XGBoost"]["is_fallback"] = not XGB_AVAILABLE

    # 6. Ensemble - the verdict the application actually shows
    # Not a sixth trained model: ml_engine.predict() averages the five
    # probabilities above and applies the screening threshold to the mean.
    # The same average is computed here so the number the app relies on
    # is reported alongside the models that produce it. There is no
    # cross-validation entry for it (nothing to re-fit) and no pickle.
    log("  [+]   Ensemble Voting (mean of the five probabilities)")
    ens_members = [lr, svm, dt, rf, xgb]
    _t0 = time.time()
    ens_proba = np.mean([m.predict_proba(X_test_s)[:, 1] for m in ens_members],
                        axis=0)
    _pred_t = (time.time() - _t0) * 1000 * (1000 / max(len(X_test_s), 1))
    results[ENSEMBLE_NAME] = _evaluate_proba(ens_proba, y_test, "ENS",
                                             0.0, _pred_t)
    results[ENSEMBLE_NAME]["is_ensemble"] = True

    # -- Step 12: K-Fold Cross Validation ----------------------
    log(f"\n[STEP 12] {KFOLD_N}-Fold Stratified Cross Validation on the training split")
    log("  (Scaler is re-fitted inside every fold, so no information leaks")
    log("   between folds. The 20% test set is excluded to keep it independent.)")

    # Cross-validate on the TRAINING split only, and pass it unscaled so each
    # fold fits its own scaler (see _kfold_cv). Keeping the 20% test set out of
    # CV means the holdout score stays a genuinely independent check.
    X_cv = X_train
    y_cv = y_train

    cv_models = {
        "Logistic Regression":          lr,
        "Support Vector Machine (SVM)": svm,
        "Decision Tree":                dt,
        "Random Forest":                rf,
        "XGBoost":                      xgb,
    }
    for mname, mobj in cv_models.items():
        log(f"  CV: {mname}")
        try:
            results[mname]["kfold_cv"] = _kfold_cv(mobj, X_cv, y_cv.values, mname[:3])
        except Exception as cv_err:
            log(f"    WARNING: CV failed for {mname}: {cv_err}")
            results[mname]["kfold_cv"] = None

    # -- Step 13: SHAP summary image ---------------------------
    # Drawn ONCE here and saved as a PNG. The web page then just shows the
    # image, instead of recomputing SHAP on every request (which takes tens
    # of seconds and would make the page unusable).
    log(f"\n[STEP 13] Generating SHAP summary image")
    _generate_shap_image(rf, X_train_s, feature_cols, log)

    # -- Save results -----------------------------------------

    duration = round(time.time() - t0, 2)
    with open(os.path.join(MODELS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save preprocessing report
    with open(os.path.join(MODELS_DIR, "preprocess_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    log("\n" + "=" * 60)
    log(f" All models trained in {duration}s")
    log(" Results  -> models/results.json")
    log(" Report   -> models/preprocess_report.txt")
    log("=" * 60 + "\n")

    return results, duration


# -------------------------------------------------------------
if __name__ == "__main__":
    train_all()