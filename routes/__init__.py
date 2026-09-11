"""
Heart Disease Prediction - Flask application factory
==========================================
Creates the app, wires up the four blueprints, and provides the shared
security helpers (login/role decorators and CSRF protection) that every
protected route uses.
"""

import os
import logging

from flask import Flask, redirect, url_for, render_template, request

from config import get_config

# The security layer lives in routes/security.py. It is re-exported here so
# that every existing "from routes import login_required" keeps working.
from routes.security import (                                    # noqa: E402,F401
    ROLE_PATIENT, ROLE_DOCTOR, ROLE_ADMIN,
    ADMIN_ROLES, ALL_ROLES,
    current_user, is_admin, is_doctor,
    login_required, role_required,
    generate_csrf_token, _check_csrf,
    login_blocked_seconds, record_login_failure, clear_login_failures,
)


# ══════════════════════════════════════════════════════════════════
# APPLICATION FACTORY
# ══════════════════════════════════════════════════════════════════
def create_app(config_object=None):
    """
    Build and return the Flask application.

    The factory pattern keeps setup in one place and lets the tests build a
    fresh app whenever they need one. In order it: loads config, starts
    logging, prepares the database, loads the models once, switches on CSRF
    checking, exposes helpers to every template, registers the four
    blueprints, and installs the error pages.
    """
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "templates"),
        static_folder=os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "static"),
    )
    app.config.from_object(config_object or get_config())

    # ── Logging: errors go to a file, never to the user ───────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    app.logger.setLevel(logging.INFO)

    # ── Database: create tables / seed accounts on first run ──────
    try:
        import auth_db
        auth_db.init_db()
    except Exception as e:
        app.logger.error(f"Database initialisation failed: {e}")

    # ── Warm the models once at startup, not per request ──────────
    try:
        import ml_engine
        ml_engine.init_engine(app.config["MODELS_DIR"])
        for warning in ml_engine.get_load_warnings():
            app.logger.warning(f"Model loading: {warning}")
    except Exception as e:
        app.logger.error(f"Model engine failed to start: {e}")

    # ── CSRF on every state-changing request ──────────────────────
    app.before_request(_check_csrf)

    # ── Make helpers available inside every template ──────────────
    @app.context_processor
    def inject_globals():
        """
        Values every template can use without the view passing them in.

        This is why base.html can ask for current_user or is_admin on any page.
        """
        return {
            "csrf_token": generate_csrf_token,
            "current_user": current_user(),
            "is_admin": is_admin(),
            "is_doctor": is_doctor(),
            "ROLE_PATIENT": ROLE_PATIENT,
            "ROLE_DOCTOR": ROLE_DOCTOR,
            "ADMIN_ROLES": ADMIN_ROLES,
        }

    # ── Blueprints ────────────────────────────────────────────────
    from routes.auth_routes import auth_bp
    from routes.patient_routes import patient_bp
    from routes.doctor_routes import doctor_bp
    from routes.admin import admin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(patient_bp)
    app.register_blueprint(doctor_bp)
    app.register_blueprint(admin_bp)

    # ── Root ──────────────────────────────────────────────────────
    @app.route("/")
    def index():
        """Send visitors to their dashboard, or to the login page if signed out."""
        if current_user():
            return redirect(url_for("patient.dashboard"))
        return redirect(url_for("auth.login"))

    # ── Error pages ───────────────────────────────────────────────
    @app.errorhandler(400)
    def bad_request(e):
        """400 page - usually a missing or expired CSRF token."""
        return render_template("errors/400.html",
                               message=getattr(e, "description", None)), 400

    @app.errorhandler(403)
    def forbidden(e):
        """403 page - signed in, but this role may not see this page."""
        return render_template("errors/403.html"), 403

    @app.errorhandler(404)
    def not_found(e):
        """404 page - no route matches this URL."""
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(e):
        # Log the real problem, show the user something friendly
        """
        500 page.

        The real exception is written to the log; the visitor only ever sees a
        short apology, never a stack trace.
        """
        app.logger.error(f"Internal error on {request.path}: {e}", exc_info=True)
        return render_template("errors/500.html"), 500

    return app
