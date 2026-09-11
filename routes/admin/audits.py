"""
Heart Disease Prediction - Administrator: activity log
===========================================
The audit trail. Every state-changing action in the system writes
a row here through auth_db.log_activity().
"""

from flask import render_template, request, redirect, url_for, flash, current_app

import auth_db
from routes.admin import admin_bp, log
from routes.security import role_required, current_user, ROLE_ADMIN



# ══════════════════════════════════════════════════════════════════
# AUDIT LOG
# ══════════════════════════════════════════════════════════════════
@admin_bp.route("/audits")
@role_required("admin")
def audits():
    """
    The activity log, filterable by action or user and split into pages.

    Every state-changing action in the app writes a row here.
    """
    try:
        entries = auth_db.get_system_logs(1000)
    except Exception as e:
        log.error(f"Audit log failed: {e}", exc_info=True)
        flash("The activity log could not be loaded.", "danger")
        entries = []

    action_filter = request.args.get("action", "all")
    search = (request.args.get("q") or "").strip().lower()

    actions = sorted({e["action"] for e in entries if e["action"]})

    filtered = []
    for e in entries:
        if action_filter != "all" and e["action"] != action_filter:
            continue
        if search and search not in str(e.get("username", "")).lower() \
                and search not in str(e.get("details", "")).lower():
            continue
        filtered.append(e)

    # -- pagination --
    per_page = current_app.config.get("AUDIT_PAGE_SIZE", 25)
    try:
        page = max(1, int(request.args.get("page", 1)))
    except (TypeError, ValueError):
        page = 1

    total_pages = max(1, (len(filtered) + per_page - 1) // per_page)
    page = min(page, total_pages)
    start = (page - 1) * per_page
    rows = filtered[start:start + per_page]

    return render_template(
        "admin/audits.html",
        rows=rows, actions=actions,
        action_filter=action_filter, search=search,
        page=page, total_pages=total_pages, total=len(filtered),
        per_page=per_page,
    )


@admin_bp.route("/audits/clear", methods=["POST"])
@role_required("admin")
def clear_audits():
    # Every administrator can do this now that there is a single admin tier.
    """Purge the activity log, keeping the original initialisation entry."""
    actor = current_user()
    try:
        auth_db.clear_system_logs(actor["username"])
        flash("The activity log has been purged.", "success")
    except Exception as e:
        log.error(f"Audit purge failed: {e}", exc_info=True)
        flash("The log could not be cleared.", "danger")
    return redirect(url_for("admin.audits"))
