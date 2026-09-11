"""
Heart Disease Prediction - Patient routes
===============================
The core clinical journey: dashboard, the risk-assessment form, the
prediction itself, personal history, downloadable reports, profile and help.

Patients see only their own records. Doctors and administrators see the
wider view, exactly as they did in the original application.
"""

import logging
from datetime import datetime

from flask import (Blueprint, render_template, request, redirect, url_for,
                   session, flash, abort, Response)

import auth_db
import ml_engine
from config import Config
from routes import (login_required, current_user, is_admin, is_doctor,
                    ROLE_PATIENT, ROLE_DOCTOR)

log = logging.getLogger(__name__)

patient_bp = Blueprint("patient", __name__)

# Readable labels for the 1-3 clinical scale - defined once, in config.py
LEVEL_LABELS = Config.LEVEL_LABELS


def _scope_user_id(user):
    """
    Patients only ever see their own records; clinical staff see everything.
    This is the single authorisation rule the whole app relies on.
    """
    return user["id"] if user["role"] == ROLE_PATIENT else None


def _patient_ref(user):
    """
    The Patient ID the app assigns to a patient account, e.g. PT-0007.

    Derived from the account id so it never changes and never needs typing.
    Doctors and admins assess other people, so they get None here and
    enter the details themselves.
    """
    if user["role"] != ROLE_PATIENT:
        return None
    return f"PT-{user['id']:04d}"


# ══════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════
@patient_bp.route("/dashboard")
@login_required
def dashboard():
    """
    The landing page after sign-in, adapted to the user's role.

    A patient sees their own totals; clinical staff see system-wide ones.
    The difference comes entirely from _scope_user_id().
    """
    user = current_user()
    try:
        predictions = auth_db.get_predictions(user_id=_scope_user_id(user))
    except Exception as e:
        log.error(f"Dashboard could not load predictions: {e}", exc_info=True)
        flash("Your records could not be loaded right now.", "danger")
        predictions = []

    high_risk = sum(1 for p in predictions if p["predicted_class"] == 1)
    low_risk = len(predictions) - high_risk

    # Most recent assessment, used for the "last result" tile
    last = predictions[0] if predictions else None
    last_risk = ml_engine.get_risk_level(last["probability"]) if last else None

    # Headline accuracy of the best model, for the "model accuracy" tile
    results = ml_engine.load_results()
    best_name = ml_engine.best_model_name()
    best_accuracy = results.get(best_name, {}).get("accuracy") if best_name else None

    # Staff dashboards also show user counts
    all_users = []
    if user["role"] != ROLE_PATIENT:
        try:
            all_users = auth_db.get_all_users()
        except Exception as e:
            log.error(f"Dashboard could not load users: {e}", exc_info=True)

    # Recent activity feed, staff only
    recent_logs = []
    if is_admin(user):
        try:
            recent_logs = auth_db.get_system_logs(8)
        except Exception:
            recent_logs = []

    # Small chart of the last 10 risk scores
    trend = [{"label": (p["timestamp"] or "")[:16],
              "value": round(p["probability"] * 100, 1)}
             for p in list(reversed(predictions[:10]))]

    return render_template(
        "shared/dashboard.html",
        predictions=predictions[:10],
        total_predictions=len(predictions),
        high_risk=high_risk, low_risk=low_risk,
        last=last, last_risk=last_risk,
        best_model=best_name, best_accuracy=best_accuracy,
        all_users=all_users,
        doctors=sum(1 for account in all_users if account["role"] == ROLE_DOCTOR),
        patients=sum(1 for account in all_users if account["role"] == ROLE_PATIENT),
        recent_logs=recent_logs,
        trend=trend,
        engine_ready=ml_engine.is_ready(),
    )


# ══════════════════════════════════════════════════════════════════
# DIAGNOSIS FORM
# ══════════════════════════════════════════════════════════════════
@patient_bp.route("/diagnosis")
@login_required
def diagnosis():
    """Show the empty risk-assessment form."""
    if not ml_engine.is_ready():
        flash("No trained models are available. Please ask an administrator "
              "to run the training pipeline.", "warning")

    return render_template(
        "patient/diagnosis.html",
        models=ml_engine.available_models(),
        warnings=ml_engine.get_load_warnings(),
        engine_ready=ml_engine.is_ready(),
        form={},
        patient_ref=_patient_ref(current_user()),
    )


# ══════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════
@patient_bp.route("/predict", methods=["POST"])
@login_required
def predict():
    """
    Run one risk assessment - the core of the application.

    The order matters and is deliberate:
      1. validate the form on the SERVER (the browser check can be bypassed)
      2. turn 11 raw inputs into the 15 features the models expect
      3. scale them with the SAME scaler that was fitted during training
      4. run every enabled model, then combine or pick one
      5. save the result, then render it

    A database failure at step 5 is reported but does not discard the
    result already computed - the user can still download the report.
    """
    user = current_user()

    if not ml_engine.is_ready():
        flash("No trained models are available.", "danger")
        return redirect(url_for("patient.diagnosis"))

    form = request.form.to_dict()

    # -- Patient accounts never type these: the ID is assigned by the app
    #    and the name is the account name. Whatever was posted is replaced.
    patient_ref = _patient_ref(user)
    if patient_ref:
        form["patient_id"] = patient_ref
        form["patient_name"] = (user.get("fullname") or "").strip()

    # -- Server-side validation. The browser also checks, but a form can
    #    always be submitted directly, so this is the real gate. ------
    is_valid, errors = ml_engine.validate_form(form)
    if not is_valid:
        for message in errors:
            flash(message, "warning")
        return render_template("patient/diagnosis.html",
                               models=ml_engine.available_models(),
                               warnings=ml_engine.get_load_warnings(),
                               engine_ready=True, form=form,
                               patient_ref=patient_ref)

    # -- Turn the form into the 15 model features ------------------
    try:
        vector, readable = ml_engine.engineer_features(form)
    except ValueError as e:
        flash(str(e), "warning")
        return render_template("patient/diagnosis.html",
                               models=ml_engine.available_models(),
                               warnings=ml_engine.get_load_warnings(),
                               engine_ready=True, form=form,
                               patient_ref=patient_ref)

    # -- Run the models --------------------------------------------
    model_choice = form.get("model_choice", "Ensemble Voting (All Models)")
    try:
        outcome = ml_engine.predict(vector, model_choice)
    except Exception as e:
        log.error(f"Prediction failed: {e}", exc_info=True)
        flash("The assessment could not be completed. The saved models may be "
              "out of date - please ask an administrator to re-run training.", "danger")
        return redirect(url_for("patient.diagnosis"))

    # -- Save it ----------------------------------------------------
    patient_name = (form.get("patient_name") or user.get("fullname") or "").strip()
    patient_id_ref = (form.get("patient_id") or "").strip()
    notes = (form.get("notes") or "").strip()

    saved_id = None
    try:
        saved_id = auth_db.add_prediction(
            user["id"], readable["age"], readable["gender"], readable["height"],
            readable["weight"], readable["ap_hi"], readable["ap_lo"],
            readable["cholesterol"], readable["gluc"], readable["smoke"],
            readable["alco"], readable["active"],
            outcome["predicted_class"], outcome["probability"],
            outcome["model_used"], patient_name, notes,
            patient_id=patient_id_ref)
        flash("Assessment complete and saved to your history.", "success")
    except Exception as e:
        log.error(f"Could not save prediction: {e}", exc_info=True)
        flash("The result below is valid but could not be saved. "
              "Please download the report.", "warning")

    risk = ml_engine.get_risk_level(outcome["probability"])

    return render_template(
        "patient/result.html",
        outcome=outcome,
        risk=risk,
        agreement=ml_engine.get_agreement_summary(
            outcome["models_at_risk"], outcome["total_models"]),
        tips=ml_engine.get_lifestyle_tips(risk["label"]),
        bmi_category=ml_engine.get_bmi_category(readable["bmi"]),
        readable=readable,
        level_labels=LEVEL_LABELS,
        patient_name=patient_name,
        patient_id_ref=patient_id_ref,
        notes=notes,
        saved_id=saved_id,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ══════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════
@patient_bp.route("/history")
@login_required
def history():
    """
    The signed-in user's assessment history, with optional filters.

    A patient sees only their own rows; clinical staff see everything.
    """
    user = current_user()
    try:
        predictions = auth_db.get_predictions(user_id=_scope_user_id(user))
    except Exception as e:
        log.error(f"History failed to load: {e}", exc_info=True)
        flash("Your history could not be loaded right now.", "danger")
        predictions = []

    # Optional filters
    verdict = request.args.get("verdict", "all")
    if verdict == "high":
        predictions = [p for p in predictions if p["predicted_class"] == 1]
    elif verdict == "low":
        predictions = [p for p in predictions if p["predicted_class"] == 0]

    search = (request.args.get("q") or "").strip().lower()
    if search:
        predictions = [p for p in predictions
                       if search in str(p.get("patient_name", "")).lower()
                       or search in str(p.get("model_used", "")).lower()]

    rows = []
    for p in predictions:
        rows.append({**p, "risk": ml_engine.get_risk_level(p["probability"])})

    # Oldest first, so the chart reads left to right through time
    chart = [{"label": (p["timestamp"] or "")[:10],
              "value": round(p["probability"] * 100, 1)}
             for p in list(reversed(predictions))][-30:]

    return render_template(
        "patient/history.html",
        rows=rows, verdict=verdict, search=search,
        chart=chart,
        is_personal=(user["role"] == ROLE_PATIENT),
    )


# ══════════════════════════════════════════════════════════════════
# REPORT DOWNLOAD
# ══════════════════════════════════════════════════════════════════
@patient_bp.route("/report/<int:pred_id>")
@login_required
def report(pred_id):
    """
    Download one assessment as a formatted plain-text report.

    Checks ownership first: you may fetch your own report, and clinical
    staff may fetch anyone's. Anyone else gets 403.
    """
    user = current_user()
    try:
        record = auth_db.get_prediction_by_id(pred_id)
    except Exception as e:
        log.error(f"Report lookup failed for {pred_id}: {e}", exc_info=True)
        abort(500)

    if not record:
        abort(404)

    # Ownership check: your own record, or you are clinical staff
    owns_it = record["user_id"] == user["id"]
    staff = is_doctor(user) or is_admin(user)
    if not owns_it and not staff:
        abort(403)

    risk = ml_engine.get_risk_level(record["probability"])
    bmi = round(record["weight"] / ((record["height"] / 100) ** 2), 2)
    bmi_cat = ml_engine.get_bmi_category(bmi)
    pulse_pressure = int(record["ap_hi"] - record["ap_lo"])

    text = f"""==================================================
HEART DISEASE PREDICTION - RISK ASSESSMENT REPORT
==================================================
Report ID  : {record['id']}
Date/Time  : {record['timestamp']}
Prepared By: {record.get('doctor_name') or record.get('doctor_username')}
Patient ID : {record.get('patient_id') or 'Not recorded'}
Patient    : {record.get('patient_name') or 'Not recorded'}

CLINICAL INDICATORS:
--------------------
Age              : {int(record['age'])} years
Gender           : {'Male' if record['gender'] == 1 else 'Female'}
Height           : {int(record['height'])} cm
Weight           : {record['weight']} kg
BMI              : {bmi} ({bmi_cat['label']})
Systolic BP      : {int(record['ap_hi'])} mmHg
Diastolic BP     : {int(record['ap_lo'])} mmHg
Pulse Pressure   : {pulse_pressure} mmHg
Cholesterol      : {LEVEL_LABELS.get(record['cholesterol'], record['cholesterol'])}
Glucose          : {LEVEL_LABELS.get(record['gluc'], record['gluc'])}
Smoker           : {'Yes' if record['smoke'] else 'No'}
Alcohol Use      : {'Yes' if record['alco'] else 'No'}
Physically Active: {'Yes' if record['active'] else 'No'}

AI PREDICTION:
--------------
Model Used      : {record['model_used']}
Risk Probability: {record['probability']:.2%}
Risk Level      : {risk['label']}
Interpretation  : {risk['note']}

RISK BANDS USED:
----------------
  LOW       below 40%
  MODERATE  40% to 70%
  HIGH      above 70%

Clinical Notes: {record.get('notes') or 'None'}

==================================================
IMPORTANT - PLEASE READ
==================================================
This is an EDUCATIONAL TOOL, NOT A MEDICAL DIAGNOSIS.

Heart Disease Prediction estimates cardiovascular risk from statistical
patterns found in population data. It has not examined the
patient, it does not know their medical history, and it can
be wrong in both directions.

Do not start, stop or change any treatment based on this
report. Always consult a qualified doctor before acting on
any result shown here.
==================================================
Report generated by Heart Disease Prediction (FYP 2026)
=================================================="""

    filename = f"heart_disease_report_{record['id']}_" \
               f"{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

    return Response(
        text,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ══════════════════════════════════════════════════════════════════
# PROFILE
# ══════════════════════════════════════════════════════════════════
@patient_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """
    View and update the signed-in user's own details.

    A new password is re-hashed through the same PBKDF2 helper used at
    registration. The copy held in the session is refreshed so the sidebar
    updates immediately.
    """
    user = current_user()

    if request.method == "POST":
        fullname = (request.form.get("fullname") or "").strip()
        email = (request.form.get("email") or "").strip()
        new_password = request.form.get("new_password") or ""
        confirm = request.form.get("confirm_password") or ""

        errors = []
        if not fullname:
            errors.append("Your full name is required.")
        if not email or "@" not in email:
            errors.append("A valid email address is required.")
        if new_password:
            if len(new_password) < 8:
                errors.append("The new password must be at least 8 characters.")
            if new_password != confirm:
                errors.append("The two passwords do not match.")

        if errors:
            for message in errors:
                flash(message, "warning")
            return redirect(url_for("patient.profile"))

        try:
            # update_user_profile re-hashes with the existing PBKDF2 helper
            auth_db.update_user_profile(user["id"], fullname, email,
                                        new_password if new_password else None)
        except Exception as e:
            log.error(f"Profile update failed: {e}", exc_info=True)
            flash("Your profile could not be saved. Please try again.", "danger")
            return redirect(url_for("patient.profile"))

        # Refresh the copy held in the session
        refreshed = dict(session["user"])
        refreshed["fullname"] = fullname
        refreshed["email"] = email
        session["user"] = refreshed

        flash("Profile updated." + (" Your password was changed." if new_password else ""),
              "success")
        return redirect(url_for("patient.profile"))

    # Read the freshest copy from the database for display
    try:
        record = auth_db.get_user_by_id(user["id"]) or user
    except Exception:
        record = user

    return render_template("patient/profile.html", record=record)


# ══════════════════════════════════════════════════════════════════
# HELP
# ══════════════════════════════════════════════════════════════════
@patient_bp.route("/help")
@login_required
def help_page():
    """Static reference page explaining the inputs, models and risk bands."""
    return render_template("patient/help.html")
