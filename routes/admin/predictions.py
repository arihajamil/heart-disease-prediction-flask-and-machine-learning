"""
Heart Disease Prediction - Administrator: prediction management
====================================================
Every assessment saved by every user, with filtering, single
deletion and a guarded bulk clear.
"""

from flask import render_template, request, redirect, url_for, flash

import auth_db
import ml_engine
from routes.admin import admin_bp, log
from routes.security import role_required, current_user



# ══════════════════════════════════════════════════════════════════
# PREDICTIONS
# ══════════════════════════════════════════════════════════════════
@admin_bp.route("/predictions")
@role_required("admin")
def predictions():
    """List every saved assessment, filterable by verdict, model or name."""
    try:
        records = auth_db.get_predictions()
    except Exception as e:
        log.error(f"Prediction list failed: {e}", exc_info=True)
        flash("Prediction records could not be loaded.", "danger")
        records = []

    verdict = request.args.get("verdict", "all")
    model_filter = request.args.get("model", "all")
    search = (request.args.get("q") or "").strip().lower()

    models_seen = sorted({r["model_used"] for r in records if r["model_used"]})

    view = []
    for r in records:
        if verdict == "high" and r["predicted_class"] != 1:
            continue
        if verdict == "low" and r["predicted_class"] != 0:
            continue
        if model_filter != "all" and r["model_used"] != model_filter:
            continue
        if search and search not in str(r.get("patient_name", "")).lower() \
                and search not in str(r.get("doctor_name", "")).lower():
            continue
        view.append({**r, "risk": ml_engine.get_risk_level(r["probability"])})

    return render_template(
        "admin/predictions.html",
        rows=view, models_seen=models_seen,
        verdict=verdict, model_filter=model_filter, search=search,
        total=len(records),
        high=sum(1 for r in records if r["predicted_class"] == 1),
        low=sum(1 for r in records if r["predicted_class"] == 0),
    )


@admin_bp.route("/predictions/delete/<int:pred_id>", methods=["POST"])
@role_required("admin")
def delete_prediction(pred_id):
    """Delete one assessment and record who removed it."""
    actor = current_user()
    try:
        auth_db.delete_prediction(pred_id, actor["username"])
        flash(f"Prediction #{pred_id} deleted.", "success")
    except Exception as e:
        log.error(f"Prediction delete failed: {e}", exc_info=True)
        flash("That record could not be deleted.", "danger")
    return redirect(url_for("admin.predictions"))


@admin_bp.route("/predictions/clear", methods=["POST"])
@role_required("admin")
def clear_predictions():
    """
    Delete every assessment in the system.

    Guarded by a typed confirmation - the admin must enter DELETE ALL - so
    an accidental click cannot wipe the data.
    """
    actor = current_user()
    # Typed confirmation guards an irreversible action
    if (request.form.get("confirm") or "").strip().upper() != "DELETE ALL":
        flash("Type DELETE ALL to confirm. Nothing was removed.", "warning")
        return redirect(url_for("admin.predictions"))
    try:
        auth_db.clear_all_predictions(actor["username"])
        flash("Every prediction record has been removed.", "success")
    except Exception as e:
        log.error(f"Clear predictions failed: {e}", exc_info=True)
        flash("The records could not be cleared.", "danger")
    return redirect(url_for("admin.predictions"))
