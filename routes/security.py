"""
Heart Disease Prediction - Security layer
===============================
Everything that decides WHO may do WHAT:

  * the three roles the system recognises
  * @login_required and @role_required, applied to every protected route
  * CSRF protection for every state-changing request
  * login rate limiting, so passwords cannot be brute-forced online

Kept separate from the application factory so the rules are in one small
file that can be read end to end.
"""

import time
import secrets
import threading
from functools import wraps

from flask import session, redirect, url_for, flash, request, abort


# ══════════════════════════════════════════════════════════════════
# ROLE MODEL
# ------------------------------------------------------------------
# The database stores the role as plain text. The original system used
# Doctor / Admin / SuperAdmin; a Patient role is added here for public
# self-registration. No schema change is required and every existing
# account keeps working exactly as before.
# ══════════════════════════════════════════════════════════════════
ROLE_PATIENT = "Patient"
ROLE_DOCTOR = "Doctor"
ROLE_ADMIN = "Admin"

# There is a SINGLE administrator tier. Admin holds every privilege in the
# system: user management, dataset, analytics, model training and the audit
# log. The old SuperAdmin role has been merged into Admin, and auth_db
# converts any leftover SuperAdmin account automatically on startup.
ADMIN_ROLES = (ROLE_ADMIN,)

# Every role the system recognises, in order of increasing privilege
ALL_ROLES = [ROLE_PATIENT, ROLE_DOCTOR, ROLE_ADMIN]


def current_user():
    """Return the logged-in user dict from the session, or None."""
    return session.get("user")


def is_admin(user=None):
    """True if this user (or the signed-in one) holds the Admin role."""
    user = user or current_user()
    return bool(user) and user.get("role") in ADMIN_ROLES


def is_doctor(user=None):
    """True if this user (or the signed-in one) holds the Doctor role."""
    user = user or current_user()
    return bool(user) and user.get("role") == ROLE_DOCTOR


# ══════════════════════════════════════════════════════════════════
# DECORATORS
# ══════════════════════════════════════════════════════════════════
def login_required(view):
    """Send anonymous visitors to the login page, remembering where they wanted to go."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user():
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def role_required(*allowed_roles):
    """
    Restrict a route to one or more roles.

    Used as @role_required("admin") or @role_required("doctor", "admin").
    A logged-in user with the wrong role gets a proper 403 page, never a crash.
    """
    # Translate the friendly names used at the call site into stored role values
    expanded = set()
    for name in allowed_roles:
        key = name.lower()
        if key == "admin":
            expanded.update(ADMIN_ROLES)
        elif key == "doctor":
            expanded.add(ROLE_DOCTOR)
        elif key == "patient":
            expanded.add(ROLE_PATIENT)
        else:
            expanded.add(name)

    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = current_user()
            if not user:
                flash("Please sign in to continue.", "warning")
                return redirect(url_for("auth.login", next=request.path))
            if user.get("role") not in expanded:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator


# ══════════════════════════════════════════════════════════════════
# LOGIN RATE LIMITING
# ------------------------------------------------------------------
# PBKDF2 makes each password guess slow, but nothing else stopped an
# attacker from simply guessing forever. This limiter does: after
# MAX_FAILURES wrong passwords against the same username OR from the
# same IP address within WINDOW seconds, further attempts are refused
# for LOCKOUT seconds. A correct login clears the counters.
#
# Both keys matter. Limiting per-username stops one account being
# hammered from many machines; limiting per-IP stops one machine
# spraying common passwords across many accounts.
#
# The counters live in process memory (a dict guarded by a lock, since
# Flask serves requests on threads). That is the right weight for this
# application: it needs no new table, resets on restart, and a single
# SQLite-backed instance is single-process anyway. A multi-process
# deployment would move this state into the database or a shared cache
# - the three functions below are the only place that change touches.
# ══════════════════════════════════════════════════════════════════
LOGIN_MAX_FAILURES = 5        # wrong passwords allowed ...
LOGIN_WINDOW = 15 * 60        # ... within this many seconds ...
LOGIN_LOCKOUT = 10 * 60       # ... before this long a lock applies

_login_failures = {}          # key -> [timestamps of recent failures]
_login_lock = threading.Lock()


def _login_keys(username):
    """The two counter keys one attempt touches: the account and the IP."""
    ip = request.remote_addr or "unknown"
    return (f"user:{(username or '').lower()}", f"ip:{ip}")


def login_blocked_seconds(username):
    """
    Seconds until this username/IP pair may try again, or 0 if not blocked.

    A key is blocked when it has accumulated LOGIN_MAX_FAILURES failures
    within the window; the lock expires LOGIN_LOCKOUT seconds after the
    most recent failure.
    """
    now = time.time()
    remaining = 0
    with _login_lock:
        for key in _login_keys(username):
            stamps = [t for t in _login_failures.get(key, [])
                      if now - t < LOGIN_WINDOW]
            if stamps:
                _login_failures[key] = stamps      # prune old entries
            else:
                # Nothing left in the window: drop the key entirely, so an
                # attacker spraying thousands of made-up usernames cannot
                # grow this dict without bound.
                _login_failures.pop(key, None)
            if len(stamps) >= LOGIN_MAX_FAILURES:
                wait = int(LOGIN_LOCKOUT - (now - stamps[-1]))
                remaining = max(remaining, wait)
    return max(remaining, 0)


def record_login_failure(username):
    """Count one wrong password against both the username and the IP."""
    now = time.time()
    with _login_lock:
        for key in _login_keys(username):
            _login_failures.setdefault(key, []).append(now)


def clear_login_failures(username):
    """A successful sign-in wipes the slate for this username and IP."""
    with _login_lock:
        for key in _login_keys(username):
            _login_failures.pop(key, None)


# ══════════════════════════════════════════════════════════════════
# CSRF PROTECTION
# ------------------------------------------------------------------
# A random token is stored in the session and echoed back in a hidden
# field on every form. A forged request from another site cannot read the
# session, so it cannot supply the right token. This is the same idea as
# Flask-WTF but needs no extra dependency.
# ══════════════════════════════════════════════════════════════════
def generate_csrf_token():
    """
    Return this session's CSRF token, creating it on first use.

    Exposed to every template so forms can embed it in a hidden field.
    """
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(32)
    return session["_csrf_token"]


def _check_csrf():
    """Reject any POST that does not carry the session's CSRF token."""
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return
    expected = session.get("_csrf_token")
    supplied = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
    if not expected or not supplied or not secrets.compare_digest(expected, supplied):
        abort(400, description="Security token missing or invalid. Please reload and try again.")
