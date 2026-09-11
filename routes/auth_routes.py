"""
Heart Disease Prediction - Authentication routes
======================================
Sign in, register and sign out. The logged-in user's row is stored in the
Flask session cookie, which is signed with SECRET_KEY so it cannot be
tampered with.
"""

import re
import logging

from flask import (Blueprint, render_template, request, redirect,
                   url_for, session, flash)

import auth_db
from routes import (current_user, ROLE_PATIENT, login_required,
                    login_blocked_seconds, record_login_failure,
                    clear_login_failures)

log = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)


def _landing_page_for(role):
    """Everyone starts at the dashboard; the dashboard adapts to the role."""
    return url_for("patient.dashboard")


# ══════════════════════════════════════════════════════════════════
# REGISTRATION RULES
# ------------------------------------------------------------------
# The same rules are checked in the browser by register.js so the user
# gets instant feedback, but these server-side checks are the ones that
# actually decide - a form can always be posted directly.
# ══════════════════════════════════════════════════════════════════
# Roles a visitor may grant THEMSELVES on the public register page.
#
# Patient only, and the restriction is the point. A Doctor account can
# read every patient's assessment history (see _scope_user_id() in
# patient_routes.py), so letting an anonymous visitor self-select
# "Doctor" would hand full patient-data access to anyone who fills in a
# form - a privacy breach, not just a policy problem. Doctor accounts
# are therefore created only by an administrator (Admin -> Doctors),
# and Administrator itself is likewise never self-service.
SELF_SERVICE_ROLES = (ROLE_PATIENT,)


def validate_registration(username, fullname, email, password, confirm,
                          role=ROLE_PATIENT, allowed_roles=SELF_SERVICE_ROLES):
    """
    Return a list of problems. An empty list means the form is good.

    `allowed_roles` exists because these rules are shared: the public
    register page calls this with the default (Patient only), while the
    admin "create doctor" page passes allowed_roles=(ROLE_DOCTOR,). The
    field rules stay identical either way, so an account created by an
    administrator can never be weaker than a self-registered one.
    """
    errors = []

    # -- role: must be one the calling page is allowed to offer --
    if role not in allowed_roles:
        errors.append("This registration page creates patient accounts only. "
                      "Doctor accounts are set up by an administrator.")

    # -- username: letters, numbers and underscores --
    if not username:
        errors.append("A username is required.")
    elif len(username) < 3:
        errors.append("The username must be at least 3 characters.")
    elif len(username) > 20:
        errors.append("The username cannot be longer than 20 characters.")
    elif not re.fullmatch(r"[A-Za-z0-9_]+", username):
        errors.append("The username may only contain letters, numbers and underscores.")
    elif not username[0].isalpha():
        errors.append("The username must start with a letter.")

    # -- full name: letters and ordinary name punctuation only --
    if not fullname:
        errors.append("Your full name is required.")
    elif len(fullname) < 2:
        errors.append("Your full name must be at least 2 characters.")
    elif len(fullname) > 60:
        errors.append("Your full name cannot be longer than 60 characters.")
    elif not re.fullmatch(r"[A-Za-z .'\-]+", fullname):
        errors.append("Your full name may only contain letters, spaces, "
                      "hyphens and apostrophes.")

    # -- email --
    if not email:
        errors.append("An email address is required.")
    elif len(email) > 120:
        errors.append("That email address is too long.")
    elif not re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", email):
        errors.append("Please enter a valid email address, for example name@example.com.")

    # -- password: must mix letters and numbers --
    if len(password) < 8:
        errors.append("The password must be at least 8 characters.")
    elif len(password) > 72:
        errors.append("The password cannot be longer than 72 characters.")
    else:
        if not re.search(r"[A-Za-z]", password):
            errors.append("The password must contain at least one letter.")
        if not re.search(r"\d", password):
            errors.append("The password must contain at least one number.")
        if re.search(r"\s", password):
            errors.append("The password cannot contain spaces.")
        if username and password.lower() == username.lower():
            errors.append("The password cannot be the same as your username.")

    if password != confirm:
        errors.append("The two passwords do not match.")

    # -- stop the same person registering more than once --
    # Checked last, so obvious format problems are reported first and we
    # only hit the database once the rest of the form looks sane.
    if not errors:
        try:
            if auth_db.username_exists(username):
                errors.append("That username is already taken. Please choose another.")
            if auth_db.email_exists(email):
                errors.append("An account already exists with that email address. "
                              "Please sign in instead, or use a different address.")
        except Exception as e:
            log.error(f"Duplicate check failed: {e}", exc_info=True)
            errors.append("The account could not be checked right now. "
                          "Please try again shortly.")

    return errors


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    # Already signed in? Go straight through.
    """
    Show the sign-in form and process it.

    On success the user's row (minus the password hash) is placed in the
    session. The session is cleared first so a brand new session id is
    issued on every privilege change, which defeats session fixation.
    """
    if current_user():
        return redirect(_landing_page_for(current_user()["role"]))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            flash("Please enter both a username and a password.", "warning")
            return render_template("auth/login.html", username=username)

        # -- Brute-force guard: too many recent failures -> wait. Checked
        #    BEFORE the password so a locked attacker learns nothing new,
        #    and the expensive PBKDF2 hash is never even computed. --------
        wait = login_blocked_seconds(username)
        if wait:
            minutes = max(1, -(-wait // 60))          # ceiling division
            try:
                auth_db.log_activity(None, username, "Login Blocked",
                                     f"Rate limit hit ({request.remote_addr}). "
                                     f"Locked for ~{minutes} more minute(s).")
            except Exception:
                pass                                  # auditing is best-effort
            flash(f"Too many failed sign-in attempts. Please wait about "
                  f"{minutes} minute{'s' if minutes != 1 else ''} and try again.",
                  "danger")
            return render_template("auth/login.html", username=username)

        try:
            user, status = auth_db.validate_login(username, password)
        except Exception as e:
            log.error(f"Login failed for {username!r}: {e}", exc_info=True)
            flash("The sign-in service is unavailable. Please try again shortly.", "danger")
            return render_template("auth/login.html", username=username)

        if status == "ok":
            clear_login_failures(username)            # slate wiped on success
            # Never keep the password hash in the session cookie
            safe_user = {k: v for k, v in user.items() if k != "password_hash"}
            session.clear()                  # new session id on privilege change
            session["user"] = safe_user
            session.permanent = True

            try:
                auth_db.log_activity(user["id"], user["username"], "Login",
                                     f"Authenticated as {user['role']}.")
            except Exception:
                pass                          # auditing must not block sign-in

            flash(f"Welcome back, {safe_user.get('fullname') or username}.", "success")

            # Honour ?next= but only for local paths, never an external URL.
            # Three checks, all needed:
            #   startswith("/")      - relative paths and absolute URLs are out
            #   not startswith("//") - "//evil.com" is protocol-relative
            #   no backslash at all  - browsers normalise "\" to "/", so
            #                          "/\evil.com" becomes "//evil.com" and
            #                          would slip past the first two checks
            next_page = request.args.get("next", "")
            if (next_page.startswith("/") and not next_page.startswith("//")
                    and "\\" not in next_page):
                return redirect(next_page)
            return redirect(_landing_page_for(safe_user["role"]))

        # Every unsuccessful attempt counts toward the rate limit, except a
        # database outage - punishing users for a server problem would be
        # wrong, and an attacker gains nothing from a dead database anyway.
        if status != "db_error":
            record_login_failure(username)

        if status == "banned":
            flash("This account has been suspended. Please contact an administrator.", "danger")
        elif status == "db_error":
            flash("The database is unavailable. Please try again shortly.", "danger")
        else:
            flash("Invalid username or password.", "danger")

        return render_template("auth/login.html", username=username)

    return render_template("auth/login.html", username="")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """
    Public self-registration - Patient accounts only.

    The posted role is never read at all: every account created here is a
    Patient, full stop. Doctor accounts can see every patient's records,
    so they are created only by an administrator (Admin -> Doctors), and
    administrative access is likewise never available from this form. An
    attacker posting role=Doctor or role=Admin directly still gets a
    Patient account, because the value is hard-coded server-side.
    """
    if current_user():
        return redirect(_landing_page_for(current_user()["role"]))

    form = {}
    if request.method == "POST":
        # The role is fixed, not chosen. Even a hand-crafted POST carrying
        # role=Doctor is ignored - the form field simply is not read.
        chosen_role = ROLE_PATIENT

        form = {
            "username": (request.form.get("username") or "").strip(),
            "fullname": (request.form.get("fullname") or "").strip(),
            "email": (request.form.get("email") or "").strip(),
        }
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""

        errors = validate_registration(
            form["username"], form["fullname"], form["email"], password, confirm,
            role=chosen_role)

        if errors:
            for message in errors:
                flash(message, "warning")
            return render_template("auth/register.html", form=form)

        try:
            user_id, error = auth_db.register_user(
                form["username"], password, chosen_role,
                form["fullname"], form["email"])
        except Exception as e:
            log.error(f"Registration failed: {e}", exc_info=True)
            flash("Registration is unavailable right now. Please try again shortly.", "danger")
            return render_template("auth/register.html", form=form)

        if error == "taken":
            flash("That username is already taken. Please choose another.", "warning")
            return render_template("auth/register.html", form=form)

        flash("Account created. You can sign in now.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/register.html", form=form)


@auth_bp.route("/logout")
@login_required
def logout():
    """Record the sign-out, clear the session and return to the login page."""
    user = current_user()
    try:
        auth_db.log_activity(user["id"], user["username"], "Logout", "User session ended.")
    except Exception:
        pass
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))
