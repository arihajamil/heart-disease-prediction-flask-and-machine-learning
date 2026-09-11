"""
Heart Disease Prediction - Database layer
===============================
Every read and write to heart_disease.db happens here. No other file talks to
SQLite directly, so the storage details stay in one place.

Four tables:
    users          accounts, roles and PBKDF2 password hashes
    predictions    one row per saved risk assessment
    system_logs    the audit trail - every state-changing action
    training_runs  a record of each model retraining

Two things worth knowing:

  * Passwords are hashed with PBKDF2-HMAC-SHA256 and a per-user random
    salt. The plain password is never stored and cannot be recovered.
  * Every query is parameterised with ? placeholders. Values are never
    formatted into SQL strings, so the app has no injection surface.

init_db() runs on import: it creates any missing tables, applies the small
schema migrations, and seeds the demo accounts if the database is empty.
"""

import sqlite3
import hashlib
import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "heart_disease.db")

# ─────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────
# Passwords are hashed with PBKDF2-HMAC-SHA256 from the Python standard
# library. Two things make this much safer than a plain SHA-256 hash:
#   1. A random "salt" is added to every password, so two users who choose
#      the same password still get completely different stored values.
#      This defeats pre-computed "rainbow table" attacks.
#   2. The hash is repeated 260,000 times, which makes brute-force guessing
#      roughly 260,000 times slower for an attacker.
#
# Stored format:  pbkdf2$<iterations>$<salt_hex>$<hash_hex>
PBKDF2_ITERATIONS = 260000


def hash_password(password: str) -> str:
    """Hash a password with a fresh random salt. Never returns the same value twice."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt,
                                 PBKDF2_ITERATIONS)
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """
    Check a typed password against the stored hash.

    Also understands the OLD unsalted SHA-256 format so that accounts created
    before this upgrade can still log in. Those old hashes are automatically
    replaced with a secure one on the next successful login (see validate_login).
    """
    if not stored:
        return False

    # New salted format
    if stored.startswith("pbkdf2$"):
        try:
            _, iterations, salt_hex, hash_hex = stored.split("$")
            digest = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                         bytes.fromhex(salt_hex), int(iterations))
            # compare_digest avoids leaking timing information to an attacker
            return secrets.compare_digest(digest.hex(), hash_hex)
        except (ValueError, TypeError):
            return False

    # Legacy unsalted SHA-256 format (pre-upgrade accounts)
    legacy = hashlib.sha256(password.encode()).hexdigest()
    return secrets.compare_digest(legacy, stored)


def is_legacy_hash(stored: str) -> bool:
    """True if this password still uses the old, insecure SHA-256 format."""
    return bool(stored) and not stored.startswith("pbkdf2$")


def _connect():
    """
    Open the database with foreign keys switched ON.

    SQLite ignores ON DELETE CASCADE unless this pragma is set, so without it
    deleting a doctor would leave their prediction records behind as orphans.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """
    Create any missing tables, run the small schema migrations, and seed
    the two demo accounts if the database is empty.

    Safe to call repeatedly - every step checks before it acts. Called
    automatically when this module is imported.
    """
    conn = _connect()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        fullname TEXT,
        email TEXT,
        is_banned INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        age REAL, gender INTEGER, height REAL, weight REAL,
        ap_hi REAL, ap_lo REAL, cholesterol INTEGER, gluc INTEGER,
        smoke INTEGER, alco INTEGER, active INTEGER,
        predicted_class INTEGER, probability REAL, model_used TEXT,
        patient_id TEXT DEFAULT '',
        patient_name TEXT DEFAULT '', notes TEXT DEFAULT '',
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )""")

    c.execute("""
        CREATE TABLE IF NOT EXISTS system_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, username TEXT, action TEXT NOT NULL,
        details TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

    c.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        triggered_by INTEGER, status TEXT DEFAULT 'running',
        duration_s REAL, results_json TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

    # ── Simple migration for databases created before patient_id existed ──
    # CREATE TABLE IF NOT EXISTS does nothing to a table that already exists,
    # so any new column has to be added explicitly.
    existing_columns = [row[1] for row in c.execute("PRAGMA table_info(predictions)")]
    if "patient_id" not in existing_columns:
        c.execute("ALTER TABLE predictions ADD COLUMN patient_id TEXT DEFAULT ''")

    # ── Migration: the specialisation field has been removed ──────────
    # Older databases still carry the column. SQLite 3.35+ can drop it, so
    # the stored schema matches the code. Safe to run repeatedly.
    user_columns = [row[1] for row in c.execute("PRAGMA table_info(users)")]
    if "specialisation" in user_columns:
        try:
            c.execute("ALTER TABLE users DROP COLUMN specialisation")
        except sqlite3.OperationalError:
            pass          # very old SQLite - leave the unused column in place

    # ── Migration: SuperAdmin has been merged into Admin ──────────────
    # The system now has a single administrator tier holding every
    # privilege. Any account still carrying the old SuperAdmin role is
    # converted to Admin so it keeps working and loses nothing. This is
    # safe to run repeatedly - once converted there is nothing left to do.
    c.execute("SELECT COUNT(*) FROM users WHERE role='SuperAdmin'")
    legacy_count = c.fetchone()[0]
    if legacy_count:
        c.execute("UPDATE users SET role='Admin' WHERE role='SuperAdmin'")
        c.execute("""INSERT INTO system_logs (user_id,username,action,details)
                     VALUES (NULL,'system','Role Migration',?)""",
                  (f"{legacy_count} SuperAdmin account(s) converted to Admin - "
                   f"the two roles are now a single administrator tier.",))

    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        # Two seed accounts: one clinician and one administrator.
        # The administrator holds every privilege in the system.
        defaults = [
            ("doctor", hash_password("doctor123"), "Doctor",
             "Dr. John Smith", "jsmith@example.com"),
            ("admin",  hash_password("admin123"),  "Admin",
             "Admin User",     "admin@example.com"),
        ]
        c.executemany("""
            INSERT INTO users (username,password_hash,role,fullname,email)
            VALUES (?,?,?,?,?)
        """, defaults)
        c.execute("""
            INSERT INTO system_logs (user_id,username,action,details)
            VALUES (1,'system','DB Initialised','Default accounts seeded successfully.')
        """)
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────
def validate_login(username, password):
    """
    Check a username and password.

    Returns (user_dict, "ok") on success, or (None, reason) where reason is
    "invalid", "banned" or "db_error".

    The password is checked BEFORE the ban status, so a wrong password on a
    suspended account does not reveal that the account exists. An account
    still using the old unsalted hash is silently upgraded here, once the
    password is known to be correct.
    """
    try:
        conn = _connect()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        row = c.fetchone()
        conn.close()
    except sqlite3.Error:
        return None, "db_error"

    if not row:
        return None, "invalid"
    if not verify_password(password, row["password_hash"]):
        return None, "invalid"
    # Ban check happens after the password check so a wrong password on a
    # banned account does not reveal that the account exists.
    if row["is_banned"]:
        return None, "banned"

    # Automatic upgrade: if this account still uses the old unsalted SHA-256
    # hash, replace it with a secure salted one now that we know the password
    # is correct. The user notices nothing and never has to reset anything.
    if is_legacy_hash(row["password_hash"]):
        try:
            conn = _connect()
            conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                         (hash_password(password), row["id"]))
            conn.commit()
            conn.close()
        except sqlite3.Error:
            pass          # upgrade is best-effort; login still succeeds

    return dict(row), "ok"


def register_user(username, password, role, fullname, email):
    """
    Create a new account.

    Returns (new_id, None) on success, or (None, "taken") if the username
    already exists. The caller is expected to have validated the input.
    """
    conn = _connect()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO users (username,password_hash,role,fullname,email)
            VALUES (?,?,?,?,?)
        """, (username, hash_password(password), role, fullname, email))
        conn.commit()
        uid = c.lastrowid
        log_activity(uid, username, "Registration", f"New {role} account created.")
        return uid, None
    except sqlite3.IntegrityError:
        return None, "taken"
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
def log_activity(user_id, username, action, details):
    """
    Write one row to the audit trail.

    Called after every action that changes state, so an administrator can
    see who did what and when.
    """
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO system_logs (user_id,username,action,details)
        VALUES (?,?,?,?)
    """, (user_id, username, action, details))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Users CRUD
# ─────────────────────────────────────────────
def get_all_users():
    """Return every account as a list of dicts, ordered by id."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id,username,role,fullname,email,is_banned,created_at FROM users ORDER BY id")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_user_by_id(user_id):
    """Fetch a single user row as a dict, or None. Read-only."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""SELECT id,username,role,fullname,email,is_banned,created_at
                 FROM users WHERE id=?""", (user_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def get_prediction_by_id(pred_id):
    """
    Fetch one prediction, joined to the user who created it.
    Used by the report download, which checks ownership before sending.
    """
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""SELECT p.*, u.fullname AS doctor_name, u.username AS doctor_username,
                        u.role AS doctor_role
                 FROM predictions p JOIN users u ON p.user_id = u.id
                 WHERE p.id = ?""", (pred_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def email_exists(email, exclude_user_id=None):
    """
    True if any account already uses this email address.

    The users table only enforces a UNIQUE username, so without this check
    the same person could register over and over with a new username each
    time. Comparison is case-insensitive because email addresses are.
    """
    if not email:
        return False
    conn = _connect()
    c = conn.cursor()
    if exclude_user_id is None:
        c.execute("SELECT COUNT(*) FROM users WHERE LOWER(email)=LOWER(?)",
                  (email.strip(),))
    else:
        # Used when editing a profile, so the account does not clash with itself
        c.execute("SELECT COUNT(*) FROM users WHERE LOWER(email)=LOWER(?) AND id<>?",
                  (email.strip(), exclude_user_id))
    found = c.fetchone()[0] > 0
    conn.close()
    return found


def username_exists(username):
    """True if this username is already taken (case-insensitive)."""
    if not username:
        return False
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE LOWER(username)=LOWER(?)",
              (username.strip(),))
    found = c.fetchone()[0] > 0
    conn.close()
    return found


def count_active_admins():
    """
    How many Admin accounts are currently able to sign in.

    There is only one administrator tier now, so the system must never be
    left with zero usable admins - that would lock everybody out with no
    way back in. The routes use this before banning, deleting or demoting.
    """
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE role='Admin' AND is_banned=0")
    total = c.fetchone()[0]
    conn.close()
    return total


def count_predictions_by_user():
    """Return {user_id: number_of_predictions} for the doctor's patient list."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT user_id, COUNT(*) FROM predictions GROUP BY user_id")
    counts = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    return counts


def update_user_profile(user_id, fullname, email, new_password=None):
    """
    Save a user's name and email, and optionally set a new password.

    Passing new_password=None leaves the existing password untouched.
    """
    conn = _connect()
    c = conn.cursor()
    if new_password:
        c.execute("UPDATE users SET fullname=?,email=?,password_hash=? WHERE id=?",
                  (fullname, email, hash_password(new_password), user_id))
    else:
        c.execute("UPDATE users SET fullname=?,email=? WHERE id=?",
                  (fullname, email, user_id))
    conn.commit()
    log_activity(user_id, fullname, "Profile Update", "User profile updated.")
    conn.close()


def update_user_role(user_id, new_role, operator_username):
    """Change a user's role and record the change in the audit log."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    uname = row[0] if row else "unknown"
    c.execute("UPDATE users SET role=? WHERE id=?", (new_role, user_id))
    conn.commit()
    log_activity(None, operator_username, "Role Change",
                 f"User {uname} role changed to {new_role}.")
    conn.close()


def ban_user(user_id, operator_username):
    """Suspend an account so it can no longer sign in."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    uname = row[0] if row else "unknown"
    c.execute("UPDATE users SET is_banned=1 WHERE id=?", (user_id,))
    conn.commit()
    log_activity(None, operator_username, "Ban User", f"User {uname} has been banned.")
    conn.close()


def unban_user(user_id, operator_username):
    """Reinstate a suspended account."""
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    uname = row[0] if row else "unknown"
    c.execute("UPDATE users SET is_banned=0 WHERE id=?", (user_id,))
    conn.commit()
    log_activity(None, operator_username, "Unban User", f"User {uname} has been unbanned.")
    conn.close()


def delete_user(user_id, operator_username):
    """
    Permanently remove an account.

    Foreign keys are ON (see _connect), so ON DELETE CASCADE also removes
    that user's saved assessments rather than leaving them orphaned.
    """
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    uname = row[0] if row else "unknown"
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    log_activity(None, operator_username, "Delete User", f"Account '{uname}' permanently deleted.")
    conn.close()


# ─────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────
def add_prediction(user_id, age, gender, height, weight, ap_hi, ap_lo,
                   cholesterol, gluc, smoke, alco, active,
                   predicted_class, probability, model_used,
                   patient_name="", notes="", patient_id=""):
    """
    Save one completed risk assessment and return its new id.

    Stores both the inputs and the outcome, so a report can be regenerated
    later without re-running the model.
    """
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions
        (user_id,age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,
        smoke,alco,active,predicted_class,probability,model_used,
        patient_id,patient_name,notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (user_id, age, gender, height, weight, ap_hi, ap_lo,
          cholesterol, gluc, smoke, alco, active,
          predicted_class, probability, model_used,
          patient_id, patient_name, notes))
    conn.commit()
    pid = c.lastrowid
    c.execute("SELECT username FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    uname = row[0] if row else "unknown"
    log_activity(user_id, uname, "Prediction",
                 f"Risk={predicted_class} ({probability:.2%}) via {model_used}.")
    conn.close()
    return pid


def get_predictions(user_id=None):
    """
    Return saved assessments, newest first.

    Passing a user_id returns only that person's records - this is the one
    rule that stops a patient seeing anybody else's data. Passing None
    returns every record, which is what clinical staff and admins see.
    """
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if user_id:
        c.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY timestamp DESC", (user_id,))
    else:
        c.execute("""
            SELECT p.*, u.fullname as doctor_name, u.username as doctor_username
            FROM predictions p JOIN users u ON p.user_id=u.id
            ORDER BY p.timestamp DESC
        """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def delete_prediction(pred_id, operator_username):
    """Remove a single assessment and log who did it."""
    conn = _connect()
    c = conn.cursor()
    c.execute("DELETE FROM predictions WHERE id=?", (pred_id,))
    conn.commit()
    log_activity(None, operator_username, "Delete Prediction", f"Prediction ID {pred_id} removed.")
    conn.close()


def clear_all_predictions(operator_username):
    """Remove every assessment in the system. Irreversible."""
    conn = _connect()
    c = conn.cursor()
    c.execute("DELETE FROM predictions")
    conn.commit()
    log_activity(None, operator_username, "Clear Predictions", "All predictions purged.")
    conn.close()


# ─────────────────────────────────────────────
# Logs
# ─────────────────────────────────────────────
def get_system_logs(limit=200):
    """Return the most recent audit entries, newest first."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM system_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def clear_system_logs(operator_username):
    """
    Purge the audit trail.

    The original 'DB Initialised' row is deliberately kept so the log is
    never completely empty.
    """
    conn = _connect()
    c = conn.cursor()
    c.execute("DELETE FROM system_logs WHERE action != 'DB Initialised'")
    conn.commit()
    log_activity(None, operator_username, "Purge Logs", "System audit logs cleared.")
    conn.close()


# ─────────────────────────────────────────────
# Training history
# ─────────────────────────────────────────────
def log_training_run(triggered_by, status, duration_s, results_json):
    """Record the outcome of a model training run."""
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        INSERT INTO training_runs (triggered_by,status,duration_s,results_json)
        VALUES (?,?,?,?)
    """, (triggered_by, status, duration_s, results_json))
    conn.commit()
    conn.close()


def get_training_runs():
    """Return the last 20 training runs, newest first."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM training_runs ORDER BY timestamp DESC LIMIT 20")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# Initialise on import
init_db()
