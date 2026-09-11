"""
Heart Disease Prediction - Doctor routes
==============================
Clinical oversight: the list of patients, one patient's full history, and
aggregate reporting across every assessment in the system.

Administrators can reach these pages too, since their role is a superset
of the doctor's.
"""

import logging
from collections import Counter, defaultdict

from flask import Blueprint, render_template, abort, request, flash

import auth_db
import ml_engine
from routes import role_required, ROLE_PATIENT, ROLE_DOCTOR

log = logging.getLogger(__name__)

doctor_bp = Blueprint("doctor", __name__, url_prefix="/doctor")


@doctor_bp.route("/patients")
@role_required("doctor", "admin")
def patients():
    """Every patient account, with how many assessments each one has."""
    try:
        users = auth_db.get_all_users()
        counts = auth_db.count_predictions_by_user()
        predictions = auth_db.get_predictions()
    except Exception as e:
        log.error(f"Patient list failed to load: {e}", exc_info=True)
        flash("The patient list could not be loaded right now.", "danger")
        users, counts, predictions = [], {}, []

    # Highest risk score seen per user, so staff can triage at a glance
    worst = defaultdict(float)
    latest = {}
    for p in predictions:
        uid = p["user_id"]
        worst[uid] = max(worst[uid], p["probability"])
        if uid not in latest:
            latest[uid] = p["timestamp"]

    search = (request.args.get("q") or "").strip().lower()

    rows = []
    for user in users:
        # Anyone who is not staff counts as a patient, plus anyone who has
        # recorded assessments (older accounts were created as Doctors).
        if user["role"] not in (ROLE_PATIENT, ROLE_DOCTOR) and not counts.get(user["id"]):
            continue
        if search and search not in (user["fullname"] or "").lower() \
                and search not in user["username"].lower():
            continue

        highest = worst.get(user["id"], 0.0)
        rows.append({
            "id": user["id"],
            "username": user["username"],
            "fullname": user["fullname"],
            "email": user["email"],
            "role": user["role"],
            "is_banned": user["is_banned"],
            "count": counts.get(user["id"], 0),
            "highest": highest,
            "highest_risk": ml_engine.get_risk_level(highest) if highest else None,
            "last_seen": latest.get(user["id"], None),
        })

    rows.sort(key=lambda r: (-r["highest"], -r["count"]))

    return render_template(
        "doctor/patients.html",
        rows=rows, search=search,
        total_patients=len(rows),
        total_assessments=sum(r["count"] for r in rows),
        at_risk=sum(1 for r in rows if r["highest"] >= 0.70),
    )


@doctor_bp.route("/patients/<int:user_id>")
@role_required("doctor", "admin")
def patient_detail(user_id):
    """One patient's complete assessment history."""
    # Keep abort() calls OUTSIDE the try block. Flask signals HTTP errors by
    # raising an exception, so a broad "except Exception" around abort(404)
    # would swallow it and report a 500 instead.
    try:
        person = auth_db.get_user_by_id(user_id)
        records = auth_db.get_predictions(user_id=user_id) if person else []
    except Exception as e:
        log.error(f"Patient detail failed for {user_id}: {e}", exc_info=True)
        abort(500)

    if not person:
        abort(404)

    rows = [{**r, "risk": ml_engine.get_risk_level(r["probability"])} for r in records]

    high = sum(1 for r in records if r["predicted_class"] == 1)
    average = (sum(r["probability"] for r in records) / len(records)) if records else 0

    # Oldest first so the chart reads left to right
    chart = [{"label": (r["timestamp"] or "")[:10],
              "value": round(r["probability"] * 100, 1)}
             for r in list(reversed(records))]

    return render_template(
        "doctor/patient_detail.html",
        person=person, rows=rows, chart=chart,
        total=len(records), high=high, low=len(records) - high,
        average=average,
        average_risk=ml_engine.get_risk_level(average) if records else None,
    )


@doctor_bp.route("/reports")
@role_required("doctor", "admin")
def reports():
    """Aggregate clinical reporting: risk distribution and volume over time."""
    try:
        records = auth_db.get_predictions()
    except Exception as e:
        log.error(f"Clinical reports failed to load: {e}", exc_info=True)
        flash("Report data could not be loaded right now.", "danger")
        records = []

    # -- optional date filter --
    date_from = (request.args.get("from") or "").strip()
    date_to = (request.args.get("to") or "").strip()
    if date_from:
        records = [r for r in records if (r["timestamp"] or "")[:10] >= date_from]
    if date_to:
        records = [r for r in records if (r["timestamp"] or "")[:10] <= date_to]

    high = sum(1 for r in records if r["predicted_class"] == 1)
    low = len(records) - high
    average = (sum(r["probability"] for r in records) / len(records)) if records else 0

    # -- risk banding across every record --
    bands = Counter()
    for r in records:
        bands[ml_engine.get_risk_level(r["probability"])["label"]] += 1
    band_chart = [
        {"label": "Low", "value": bands.get("LOW RISK", 0), "colour": "#10b981"},
        {"label": "Moderate", "value": bands.get("MODERATE RISK", 0), "colour": "#f59e0b"},
        {"label": "High", "value": bands.get("HIGH RISK", 0), "colour": "#ef4444"},
    ]

    # -- assessments per day, split high vs low --
    by_day = defaultdict(lambda: {"high": 0, "low": 0})
    for r in records:
        day = (r["timestamp"] or "")[:10]
        if not day:
            continue
        if r["predicted_class"] == 1:
            by_day[day]["high"] += 1
        else:
            by_day[day]["low"] += 1
    timeline = [{"label": day, "high": tally["high"], "low": tally["low"]}
                for day, tally in sorted(by_day.items())][-30:]

    # -- which models are actually being used --
    model_counts = Counter(r["model_used"] for r in records)
    model_chart = [{"label": name, "value": count}
                   for name, count in model_counts.most_common()]

    return render_template(
        "doctor/reports.html",
        total=len(records), high=high, low=low, average=average,
        band_chart=band_chart, timeline=timeline, model_chart=model_chart,
        date_from=date_from, date_to=date_to,
        has_data=bool(records),
    )
