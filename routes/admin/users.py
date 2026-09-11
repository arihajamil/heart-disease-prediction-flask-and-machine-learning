"""
Heart Disease Prediction - Administrator: user and doctor accounts
=========================================================
Listing every account, changing roles, suspending and deleting,
and creating clinician logins.

Two safety rules run through this module: nobody may act on their
own account, and the last active administrator is protected from
removal so the system can never be left with no way in.
"""

from collections import Counter

from flask import render_template, request, redirect, url_for, flash

import auth_db
from routes.admin import admin_bp, log, _can_manage
from routes.security import role_required, current_user, ROLE_DOCTOR, ROLE_ADMIN, ALL_ROLES



# ══════════════════════════════════════════════════════════════════
# USERS
# ══════════════════════════════════════════════════════════════════
@admin_bp.route("/users")
@role_required("admin")
def users():
    """
    List every account, with optional role and text filters.

    Also counts how many assessments each user has recorded.
    """
    try:
        rows = auth_db.get_all_users()
        counts = auth_db.count_predictions_by_user()
    except Exception as e:
        log.error(f"User list failed: {e}", exc_info=True)
        flash("The user list could not be loaded.", "danger")
        rows, counts = [], {}

    role_filter = request.args.get("role", "all")
    search = (request.args.get("q") or "").strip().lower()

    view = []
    for u in rows:
        if role_filter != "all" and u["role"] != role_filter:
            continue
        if search and search not in u["username"].lower() \
                and search not in (u["fullname"] or "").lower():
            continue
        view.append({**u, "count": counts.get(u["id"], 0)})

    by_role = Counter(u["role"] for u in rows)

    return render_template(
        "admin/users.html",
        rows=view, all_roles=ALL_ROLES,
        role_filter=role_filter, search=search,
        total=len(rows), by_role=by_role,
        banned=sum(1 for account in rows if account["is_banned"]),
    )


@admin_bp.route("/users/<int:user_id>/role", methods=["POST"])
@role_required("admin")
def change_role(user_id):
    """
    Change one user's role.

    Promoting anyone TO Admin is always allowed. Demoting is blocked if the
    target is the last active administrator, because that would leave the
    system with nobody able to manage it.
    """
    actor = current_user()
    new_role = request.form.get("role", "")

    if new_role not in ALL_ROLES:
        flash("That is not a valid role.", "warning")
        return redirect(url_for("admin.users"))

    target = auth_db.get_user_by_id(user_id)
    if not target:
        flash("That user no longer exists.", "warning")
        return redirect(url_for("admin.users"))

    # Demoting the last administrator would lock everyone out, so that
    # specific change is blocked. Promoting anyone TO Admin is always fine.
    if new_role != ROLE_ADMIN:
        allowed, reason = _can_manage(target, action="demoted")
        if not allowed:
            flash(reason, "danger")
            return redirect(url_for("admin.users"))
    elif target["id"] == actor["id"]:
        flash("You cannot change your own role.", "danger")
        return redirect(url_for("admin.users"))

    if target["role"] == new_role:
        flash(f"{target['username']} already has the {new_role} role.", "info")
        return redirect(url_for("admin.users"))

    try:
        auth_db.update_user_role(user_id, new_role, actor["username"])
        flash(f"{target['username']} is now a {new_role}.", "success")
    except Exception as e:
        log.error(f"Role change failed: {e}", exc_info=True)
        flash("The role could not be changed.", "danger")

    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/ban", methods=["POST"])
@role_required("admin")
def ban(user_id):
    """
    Suspend or reinstate an account, whichever it is not already.

    Reinstating is always safe; suspending runs the last-admin check first.
    """
    actor = current_user()
    target = auth_db.get_user_by_id(user_id)
    if not target:
        flash("That user no longer exists.", "warning")
        return redirect(url_for("admin.users"))

    # Reinstating is always safe; only suspending needs the last-admin check
    if not target["is_banned"]:
        allowed, reason = _can_manage(target, action="suspended")
        if not allowed:
            flash(reason, "danger")
            return redirect(url_for("admin.users"))

    try:
        if target["is_banned"]:
            auth_db.unban_user(user_id, actor["username"])
            flash(f"{target['username']} has been reinstated.", "success")
        else:
            auth_db.ban_user(user_id, actor["username"])
            flash(f"{target['username']} has been suspended.", "success")
    except Exception as e:
        log.error(f"Ban toggle failed: {e}", exc_info=True)
        flash("That account could not be updated.", "danger")

    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/delete", methods=["POST"])
@role_required("admin")
def delete_user(user_id):
    """Permanently delete an account, subject to the same safety rules."""
    actor = current_user()
    target = auth_db.get_user_by_id(user_id)
    if not target:
        flash("That user no longer exists.", "warning")
        return redirect(url_for("admin.users"))

    allowed, reason = _can_manage(target, action="deleted")
    if not allowed:
        flash(reason, "danger")
        return redirect(url_for("admin.users"))

    try:
        auth_db.delete_user(user_id, actor["username"])
        flash(f"Account '{target['username']}' has been permanently deleted.", "success")
    except Exception as e:
        log.error(f"User delete failed: {e}", exc_info=True)
        flash("That account could not be deleted.", "danger")

    return redirect(url_for("admin.users"))


# ══════════════════════════════════════════════════════════════════
# DOCTORS
# ══════════════════════════════════════════════════════════════════
@admin_bp.route("/doctors", methods=["GET", "POST"])
@role_required("admin")
def doctors():
    """
    List doctor accounts and create new ones.

    Creation reuses validate_registration(), the same rules the public
    register page applies, so an account made here cannot be weaker or a
    duplicate.
    """
    actor = current_user()

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        fullname = (request.form.get("fullname") or "").strip()
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""

        # Reuse the same field rules the public register page applies, so a
        # doctor created here cannot be weaker or duplicated than a patient
        # who signs up. The public page only permits Patient; this admin-only
        # route explicitly widens the allowed set to Doctor.
        from routes.auth_routes import validate_registration
        errors = validate_registration(username, fullname, email,
                                       password, password,
                                       role=ROLE_DOCTOR,
                                       allowed_roles=(ROLE_DOCTOR,))

        if errors:
            for message in errors:
                flash(message, "warning")
        else:
            try:
                new_id, error = auth_db.register_user(
                    username, password, ROLE_DOCTOR, fullname, email)
                if error == "taken":
                    flash("That username is already taken.", "warning")
                else:
                    auth_db.log_activity(actor["id"], actor["username"],
                                         "Create Doctor",
                                         f"Doctor account '{username}' created.")
                    flash(f"Doctor account '{username}' created.", "success")
                    return redirect(url_for("admin.doctors"))
            except Exception as e:
                log.error(f"Doctor creation failed: {e}", exc_info=True)
                flash("The account could not be created.", "danger")

    try:
        everyone = auth_db.get_all_users()
        counts = auth_db.count_predictions_by_user()
    except Exception as e:
        log.error(f"Doctor list failed: {e}", exc_info=True)
        everyone, counts = [], {}

    rows = [{**account, "count": counts.get(account["id"], 0)}
            for account in everyone if account["role"] == ROLE_DOCTOR]

    return render_template("admin/doctors.html", rows=rows,
                           active=sum(1 for r in rows if not r["is_banned"]),
                           suspended=sum(1 for r in rows if r["is_banned"]))
