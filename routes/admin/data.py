"""
Heart Disease Prediction - Administrator: dataset and analytics
====================================================
Inspecting the training dataset, and the system-wide charts
showing how the platform is being used.
"""

import os
from collections import Counter, defaultdict

from flask import render_template, current_app, flash

import auth_db
import ml_engine
from routes.admin import admin_bp, log
from routes.security import role_required, ROLE_DOCTOR, ROLE_PATIENT, ALL_ROLES



# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════
@admin_bp.route("/dataset")
@role_required("admin")
def dataset():
    """
    Show what is inside the training dataset.

    Row count, class balance, per-column types and a ten-row preview, so an
    examiner can see the data the models learned from.
    """
    path = current_app.config["DATA_PATH"]
    info = {"exists": os.path.exists(path), "name": os.path.basename(path)}

    if not info["exists"]:
        return render_template("admin/dataset.html", info=info,
                               columns=[], preview=[], stats=None)

    info["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 2)

    try:
        import pandas as pd
        frame = pd.read_csv(path)
    except Exception as e:
        log.error(f"Dataset could not be read: {e}", exc_info=True)
        flash("The dataset file could not be read as a CSV.", "danger")
        return render_template("admin/dataset.html", info=info,
                               columns=[], preview=[], stats=None)

    target = "cardio" if "cardio" in frame.columns else (
        "target" if "target" in frame.columns else None)

    stats = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "target": target,
        "positive": int(frame[target].sum()) if target else 0,
        "negative": int((frame[target] == 0).sum()) if target else 0,
        "balance": round(float(frame[target].mean()) * 100, 1) if target else 0,
        "missing": int(frame.isnull().sum().sum()),
    }

    columns = [{"name": c,
                "dtype": str(frame[c].dtype),
                "nulls": int(frame[c].isnull().sum()),
                "unique": int(frame[c].nunique())}
               for c in frame.columns]

    preview = frame.head(10).to_dict(orient="records")

    balance_chart = [
        {"label": "No disease", "value": stats["negative"], "colour": "#10b981"},
        {"label": "Disease", "value": stats["positive"], "colour": "#ef4444"},
    ] if target else []

    return render_template("admin/dataset.html", info=info, stats=stats,
                           columns=columns, preview=preview,
                           preview_cols=list(frame.columns),
                           balance_chart=balance_chart)


# ══════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════
@admin_bp.route("/analytics")
@role_required("admin")
def analytics():
    """
    System-wide usage charts.

    Assessments per day, the high/low split, risk bands and users by role.
    All the counting happens here; the browser only draws the numbers.
    """
    try:
        records = auth_db.get_predictions()
        everyone = auth_db.get_all_users()
    except Exception as e:
        log.error(f"Analytics failed: {e}", exc_info=True)
        flash("Analytics data could not be loaded.", "danger")
        records, everyone = [], []

    high = sum(1 for r in records if r["predicted_class"] == 1)

    # Assessments per day, split by verdict
    by_day = defaultdict(lambda: {"high": 0, "low": 0})
    for r in records:
        day = (r["timestamp"] or "")[:10]
        if not day:
            continue
        by_day[day]["high" if r["predicted_class"] == 1 else "low"] += 1
    timeline = [{"label": day, "high": tally["high"], "low": tally["low"]}
                for day, tally in sorted(by_day.items())][-30:]

    role_counts = Counter(account["role"] for account in everyone)
    role_chart = [{"label": role, "value": role_counts.get(role, 0)}
                  for role in ALL_ROLES if role_counts.get(role, 0)]

    model_counts = Counter(r["model_used"] for r in records)
    model_chart = [{"label": n, "value": c} for n, c in model_counts.most_common()]

    bands = Counter(ml_engine.get_risk_level(r["probability"])["label"] for r in records)
    band_chart = [
        {"label": "Low", "value": bands.get("LOW RISK", 0), "colour": "#10b981"},
        {"label": "Moderate", "value": bands.get("MODERATE RISK", 0), "colour": "#f59e0b"},
        {"label": "High", "value": bands.get("HIGH RISK", 0), "colour": "#ef4444"},
    ]

    return render_template(
        "admin/analytics.html",
        total=len(records), high=high, low=len(records) - high,
        users=len(everyone),
        doctors=role_counts.get(ROLE_DOCTOR, 0),
        patients=role_counts.get(ROLE_PATIENT, 0),
        timeline=timeline, role_chart=role_chart,
        model_chart=model_chart, band_chart=band_chart,
        split={"high": high, "low": len(records) - high},
        has_data=bool(records),
    )
