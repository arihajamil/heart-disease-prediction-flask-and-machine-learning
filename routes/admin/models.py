"""
Heart Disease Prediction - Administrator: model performance and training
=============================================================
The evaluation dashboard, and retraining the five classifiers on
a background thread so the interface stays responsive.

SHAP is deliberately NOT computed per request - the summary image
is written once during training and simply displayed here.
"""

import os
import json
import time
import threading

from flask import render_template, request, redirect, url_for, flash, current_app

import auth_db
import ml_engine
from routes.admin import admin_bp, log, _training_state, _training_lock
from routes.security import role_required, current_user



# ══════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
@admin_bp.route("/model-performance")
@role_required("admin")
def model_performance():
    """
    The evaluation dashboard for all five classifiers.

    Reads models/results.json, which training wrote - nothing is computed
    here. Feature importances come straight off the trained model, and the
    SHAP picture is a PNG produced during training, so the page stays fast.
    """
    results = ml_engine.load_results()
    if not results:
        return render_template("admin/model_performance.html",
                               rows=[], best=None, chart=None,
                               calibration=[], has_screening=False,
                               screen_threshold=None,
                               importances=[], importance_model=None,
                               shap_image=None)

    best = ml_engine.best_model_name()

    rows = []
    calibration = []
    for name, data in results.items():
        cv = data.get("kfold_cv") or {}
        matrix = data.get("conf_matrix", [[0, 0], [0, 0]])
        tn, fp = matrix[0][0], matrix[0][1]
        fn, tp = matrix[1][0], matrix[1][1]
        # Written by newer training runs only, so every read is guarded -
        # a results.json from before these metrics existed must still render.
        screening = data.get("screening") or {}
        rows.append({
            "name": name,
            "short": ml_engine.MODEL_SHORT.get(name, name[:3]),
            "accuracy": data.get("accuracy", 0),
            "precision": data.get("precision", 0),
            "recall": data.get("recall", 0),
            "f1": data.get("f1", 0),
            "auc": data.get("auc", 0),
            "brier": data.get("brier"),
            "screen_recall": screening.get("recall"),
            "screen_precision": screening.get("precision"),
            "screen_threshold": screening.get("threshold"),
            "train_time": data.get("training_time_s", 0),
            "pred_time": data.get("pred_time_ms", 0),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": tp / (tp + fn) if (tp + fn) else 0,
            "specificity": tn / (tn + fp) if (tn + fp) else 0,
            "cv_auc": (cv.get("mean") or {}).get("auc"),
            "cv_std": (cv.get("std") or {}).get("auc"),
            "cv_accuracy": (cv.get("mean") or {}).get("accuracy"),
            "is_best": name == best,
            "is_ensemble": bool(data.get("is_ensemble")),
        })
        cal = data.get("calibration") or {}
        if cal.get("mean_predicted"):
            calibration.append({
                "name": name,
                "short": ml_engine.MODEL_SHORT.get(name, name[:3]),
                "mean_predicted": cal["mean_predicted"],
                "fraction_positive": cal["fraction_positive"],
            })
    # Classifiers first, by AUC; the ensemble row last so the five models
    # it is built from are read before it.
    rows.sort(key=lambda r: (r["is_ensemble"], -r["auc"]))
    has_screening = any(r["screen_recall"] is not None for r in rows)
    screen_threshold = next((r["screen_threshold"] for r in rows
                             if r["screen_threshold"] is not None), None)

    # Feature importances are read from an already-trained model, which is
    # instant. SHAP is NOT computed per request - a summary image is written
    # once during training instead (see below).
    importances, importance_model = ml_engine.get_feature_importances("Random Forest")

    shap_name = "shap_summary.png"
    shap_path = os.path.join(current_app.config["STATIC_IMG_DIR"], shap_name)
    shap_image = shap_name if os.path.exists(shap_path) else None

    return render_template(
        "admin/model_performance.html",
        rows=rows, best=best,
        chart={"models": rows},
        calibration=calibration,
        has_screening=has_screening,
        screen_threshold=screen_threshold,
        importances=importances[:12],
        importance_model=importance_model,
        shap_image=shap_image,
        test_size=sum(sum(row) for row in
                      results[list(results)[0]].get("conf_matrix", [[0, 0], [0, 0]])),
    )

# ══════════════════════════════════════════════════════════════════
# MODEL TRAINING  (background thread)
# ══════════════════════════════════════════════════════════════════
def _run_training(app, user_id, username):
    """
    Executed on a worker thread so the browser is never left hanging.
    Progress is published through _training_state, which the page polls.
    """
    started = time.time()
    try:
        from research import train_models
        results, duration = train_models.train_all()

        with _training_lock:
            _training_state.update({
                "running": False, "status": "success",
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": duration,
                "message": f"Training finished in {duration}s. "
                           f"{sum(1 for r in results.values() if not r.get('is_ensemble'))} "
                           f"models updated.",
            })

        # Reload the freshly written pickles so predictions use them at once
        with app.app_context():
            ml_engine.init_engine(app.config["MODELS_DIR"])

        auth_db.log_training_run(user_id, "success", duration, json.dumps(results))
        auth_db.log_activity(user_id, username, "Model Training",
                             f"Training completed in {duration}s.")

    except Exception as e:
        duration = round(time.time() - started, 1)
        log.error(f"Training failed: {e}", exc_info=True)

        # Turn the raw Python error into something a person can act on
        text = str(e)
        if "MemoryError" in text or isinstance(e, MemoryError):
            friendly = "The machine ran out of memory. Try a smaller dataset."
        elif isinstance(e, KeyError):
            friendly = ("The dataset is missing a column the pipeline needs. It must "
                        "contain the standard measurements plus 'cardio' or 'target'.")
        elif isinstance(e, FileNotFoundError):
            friendly = "The dataset file could not be found. Please check heart.csv exists."
        elif isinstance(e, ValueError):
            friendly = ("The dataset contains values the pipeline could not process - "
                        "check for text in numeric columns or an empty target column.")
        else:
            friendly = f"Training stopped unexpectedly ({type(e).__name__})."

        with _training_lock:
            _training_state.update({
                "running": False, "status": "failed",
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": duration, "message": friendly,
            })

        try:
            auth_db.log_training_run(user_id, "failed", duration, "{}")
            auth_db.log_activity(user_id, username, "Model Training",
                                 f"Training failed: {friendly}")
        except Exception:
            pass


@admin_bp.route("/training")
@role_required("admin")
def training():
    """Show training status, dataset health and the history of past runs."""
    try:
        runs = auth_db.get_training_runs()
    except Exception as e:
        log.error(f"Training history failed: {e}", exc_info=True)
        runs = []

    with _training_lock:
        state = dict(_training_state)

    dataset_path = current_app.config["DATA_PATH"]
    dataset_ok = os.path.exists(dataset_path)

    return render_template(
        "admin/training.html",
        runs=runs, state=state,
        dataset_ok=dataset_ok,
        dataset_name=os.path.basename(dataset_path),
        models_loaded=len(ml_engine.available_models()),
        warnings=ml_engine.get_load_warnings(),
    )


@admin_bp.route("/training/run", methods=["POST"])
@role_required("admin")
def run_training():
    """
    Start a retraining run on a background thread.

    The dataset is checked first, so an obvious problem is reported at once
    rather than several minutes later. Running it on a thread keeps the
    interface usable while the models rebuild.
    """
    actor = current_user()

    with _training_lock:
        if _training_state["running"]:
            flash("Training is already running. Please wait for it to finish.", "warning")
            return redirect(url_for("admin.training"))

    # Check the dataset before starting, so an obvious problem is reported
    # immediately rather than several minutes later on a worker thread.
    dataset_path = current_app.config["DATA_PATH"]
    if not os.path.exists(dataset_path):
        flash(f"Training cannot start: {os.path.basename(dataset_path)} was not found.",
              "danger")
        return redirect(url_for("admin.training"))

    try:
        import pandas as pd
        head = pd.read_csv(dataset_path, nrows=5, sep=None, engine="python")
        if "cardio" not in head.columns and "target" not in head.columns:
            flash("Training cannot start: the dataset has no 'cardio' or 'target' column.",
                  "danger")
            return redirect(url_for("admin.training"))
    except Exception:
        flash("Training cannot start: the dataset could not be read as a CSV.", "danger")
        return redirect(url_for("admin.training"))

    with _training_lock:
        _training_state.update({
            "running": True, "status": "running",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": None, "duration": None,
            "message": "Training is running in the background…",
        })

    worker = threading.Thread(
        target=_run_training,
        args=(current_app._get_current_object(), actor["id"], actor["username"]),
        daemon=True,
    )
    worker.start()

    flash("Training started. This page refreshes on its own until it finishes.", "info")
    return redirect(url_for("admin.training"))


@admin_bp.route("/training/status")
@role_required("admin")
def training_status():
    """Small JSON endpoint the training page polls."""
    with _training_lock:
        return dict(_training_state)
