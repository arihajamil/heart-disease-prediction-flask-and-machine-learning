"""
Heart Disease Prediction - Simple Test Suite (Flask edition)
==================================================
Run with:   python test_project.py

No pytest, no fixtures, no mocks - just plain assert-style checks so the
tests are easy to read and easy to explain in the FYP viva.

Covers:
  1. feature engineering          5. input validation
  2. password security            6. database
  3. model loading                7. Flask routes and access control
  4. end-to-end prediction
"""

import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

passed = 0
failed = 0
failures = []


def check(name, condition, detail=""):
    """Record one test result and print it immediately."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        failures.append(name)
        print(f"  FAIL  {name}   {detail}")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 1 - Feature engineering
# ------------------------------------------------------------------
# The four engineered features must be calculated the SAME way when we
# train the models and when we predict for a live patient. If they drift
# apart, the model silently receives wrong numbers.
# ══════════════════════════════════════════════════════════════════
print("\n[1] FEATURE ENGINEERING")

import ml_engine   # noqa: E402

ml_engine.init_engine(MODELS_DIR)


def engineer(age, height, weight, ap_hi, ap_lo, cholesterol):
    """Shortcut to the shared helper, so the tests exercise the real code."""
    return ml_engine.compute_engineered_features(
        age, height, weight, ap_hi, ap_lo, cholesterol)


bmi, pp, ag, hrf = engineer(45, 170, 70, 120, 80, 1)
check("BMI is calculated correctly (170cm/70kg = 24.22)", bmi == 24.22, f"got {bmi}")
check("Pulse pressure = systolic - diastolic (120-80 = 40)", pp == 40, f"got {pp}")
check("Age 45 falls in age_group 1", ag == 1, f"got {ag}")
check("high_risk_flag is 0 for normal cholesterol + normal BP", hrf == 0, f"got {hrf}")

# high_risk_flag should only fire when BOTH conditions are true
check("high_risk_flag = 1 when cholesterol>=2 AND ap_hi>=140",
      engineer(45, 170, 70, 150, 80, 2)[3] == 1)
check("high_risk_flag = 0 when only blood pressure is high",
      engineer(45, 170, 70, 150, 80, 1)[3] == 0)
check("high_risk_flag = 0 when only cholesterol is high",
      engineer(45, 170, 70, 120, 80, 2)[3] == 0)

# Age group boundaries must match pandas cut(bins=[0,30,45,60,75,120])
boundary_ages = [30, 31, 45, 46, 60, 61, 75, 76]
expected_groups = [0, 1, 1, 2, 2, 3, 3, 4]
actual_groups = [engineer(a, 170, 70, 120, 80, 1)[2] for a in boundary_ages]
check("Age group boundaries match the training pipeline",
      actual_groups == expected_groups, f"got {actual_groups}")

# BMI must be clipped so extreme inputs cannot reach the model
check("BMI is clipped to a maximum of 70", engineer(45, 100, 200.0, 120, 80, 1)[0] == 70)
check("BMI is clipped to a minimum of 10", engineer(45, 250, 20.0, 120, 80, 1)[0] == 10)

# The feature vector must have 15 values in the exact recorded order
vector = ml_engine.build_feature_vector = None  # placeholder, see below
form = {"age": 45, "gender": 1, "height": 170, "weight": 70, "ap_hi": 120,
        "ap_lo": 80, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}
vec, readable = ml_engine.engineer_features(form)
check("Feature vector has 15 values", len(vec) == 15, f"got {len(vec)}")
check("Feature vector order matches features.json",
      ml_engine.get_feature_names() ==
      ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol",
       "gluc", "smoke", "alco", "active", "bmi", "pulse_pressure",
       "age_group", "high_risk_flag"],
      f"got {ml_engine.get_feature_names()}")

# The training pipeline calculates these on a whole DataFrame at once (fast),
# while the app calculates them one patient at a time. This proves they agree.
try:
    import pandas as pd
    from research import train_models

    sample = pd.DataFrame({
        "age":         [25, 35, 50, 70, 80],
        "height":      [160, 175, 168, 180, 155],
        "weight":      [55.0, 80.0, 95.0, 70.0, 60.0],
        "ap_hi":       [110, 130, 150, 145, 120],
        "ap_lo":       [70, 85, 95, 90, 80],
        "cholesterol": [1, 2, 3, 2, 1],
    })
    engineered = train_models._feature_engineering(sample.copy())

    all_match = True
    for i in range(len(sample)):
        row = sample.iloc[i]
        expected = ml_engine.compute_engineered_features(
            row["age"], row["height"], row["weight"],
            row["ap_hi"], row["ap_lo"], row["cholesterol"])
        actual = (engineered["bmi"].iloc[i], engineered["pulse_pressure"].iloc[i],
                  engineered["age_group"].iloc[i], engineered["high_risk_flag"].iloc[i])
        if [round(float(v), 2) for v in expected] != [round(float(v), 2) for v in actual]:
            all_match = False

    check("Training pipeline and app produce IDENTICAL features", all_match,
          "the two implementations have drifted apart")
except ImportError:
    print("  SKIP  Feature parity test (scikit-learn not installed)")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 2 - Password security
# ══════════════════════════════════════════════════════════════════
print("\n[2] PASSWORD SECURITY")

import auth_db   # noqa: E402

hash_a = auth_db.hash_password("MySecret123")
hash_b = auth_db.hash_password("MySecret123")

check("Password is not stored in plain text", "MySecret123" not in str(hash_a))
check("Hash is a non-empty string", isinstance(hash_a, str) and len(hash_a) > 20)
check("Uses PBKDF2, not bare SHA-256", hash_a.startswith("pbkdf2$"))

# A salted hash produces a DIFFERENT string every time, even for the same
# password. This is what stops an attacker using a pre-computed table.
check("Same password gives two DIFFERENT hashes (salted)", hash_a != hash_b)
check("Correct password verifies successfully",
      auth_db.verify_password("MySecret123", hash_a))
check("Wrong password is rejected",
      not auth_db.verify_password("WrongPassword", hash_a))
check("Empty stored hash is rejected", not auth_db.verify_password("x", ""))

# Old accounts created before the upgrade must still be able to sign in
import hashlib   # noqa: E402
legacy = hashlib.sha256("OldPassword".encode()).hexdigest()
check("Legacy SHA-256 hashes still verify (migration path)",
      auth_db.verify_password("OldPassword", legacy))
check("Legacy hashes are correctly identified", auth_db.is_legacy_hash(legacy))
check("New hashes are NOT flagged as legacy", not auth_db.is_legacy_hash(hash_a))


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 3 - Model loading
# ══════════════════════════════════════════════════════════════════
print("\n[3] MODEL LOADING")

check("scaler.pkl exists", os.path.exists(os.path.join(MODELS_DIR, "scaler.pkl")))
check("features.json exists", os.path.exists(os.path.join(MODELS_DIR, "features.json")))
check("The ML engine reports itself ready", ml_engine.is_ready())
check("At least one model loaded", len(ml_engine.available_models()) >= 1)

for warning in ml_engine.get_load_warnings():
    print(f"  NOTE  {warning}")

# One broken or missing model file must not disable all the others
check("Engine reports load problems instead of hiding them",
      isinstance(ml_engine.get_load_warnings(), list))

results = ml_engine.load_results()
check("results.json loads", isinstance(results, dict) and len(results) > 0)
check("A best model can be identified", ml_engine.best_model_name() is not None)


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 4 - End-to-end prediction
# ══════════════════════════════════════════════════════════════════
print("\n[4] END-TO-END PREDICTION")

if ml_engine.is_ready():
    healthy = {"age": 30, "gender": 0, "height": 170, "weight": 62, "ap_hi": 110,
               "ap_lo": 75, "cholesterol": 1, "gluc": 1,
               "smoke": 0, "alco": 0, "active": 1}
    vec_healthy, _ = ml_engine.engineer_features(healthy)
    outcome = ml_engine.predict(vec_healthy)

    check("Prediction returns a probability between 0 and 1",
          0.0 <= outcome["probability"] <= 1.0, f"got {outcome['probability']}")
    check("Prediction returns class 0 or 1",
          outcome["predicted_class"] in (0, 1))
    check("Every enabled model voted",
          outcome["total_models"] == len(ml_engine.available_models()))
    check("A healthy 30-year-old scores below 50%",
          outcome["probability"] < 0.5, f"got {outcome['probability']:.1%}")

    unhealthy = {"age": 62, "gender": 1, "height": 170, "weight": 95, "ap_hi": 170,
                 "ap_lo": 100, "cholesterol": 3, "gluc": 3,
                 "smoke": 1, "alco": 1, "active": 0}
    vec_unhealthy, _ = ml_engine.engineer_features(unhealthy)
    bad_outcome = ml_engine.predict(vec_unhealthy)

    check("An at-risk 62-year-old scores higher than a healthy 30-year-old",
          bad_outcome["probability"] > outcome["probability"],
          f"{bad_outcome['probability']:.1%} vs {outcome['probability']:.1%}")

    # Single-model mode must work as well as the ensemble
    first_model = ml_engine.available_models()[0]
    single = ml_engine.predict(vec_healthy, first_model)
    check(f"Single-model mode works ({first_model})",
          single["model_used"] == first_model)
    check("Ensemble mode is labelled correctly",
          outcome["model_used"] == "Ensemble Voting")
else:
    check("Prediction test could run", False, "no models available")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 5 - Input validation (server side)
# ══════════════════════════════════════════════════════════════════
print("\n[5] INPUT VALIDATION")

good = {"patient_name": "Ali Khan", "patient_id": "PT-001", "notes": "",
        "age": 45, "gender": 1, "height": 170, "weight": 70, "ap_hi": 120,
        "ap_lo": 80, "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}

ok, errors = ml_engine.validate_form(good)
check("A normal patient passes validation", ok, str(errors))


def rejects(label, **changes):
    """Assert that the assessment form is refused when a field is changed to a bad value."""
    payload = dict(good)
    payload.update(changes)
    valid, _ = ml_engine.validate_form(payload)
    check(label, not valid, "the invalid input was accepted")


rejects("Systolic lower than diastolic is REJECTED", ap_hi=60, ap_lo=200)
rejects("Systolic equal to diastolic is REJECTED", ap_hi=120, ap_lo=120)
rejects("An impossible BMI is REJECTED", height=100, weight=195)
rejects("A child is REJECTED (models are adult-trained)", age=5)
rejects("Out-of-range blood pressure is REJECTED", ap_hi=400)
rejects("An invalid cholesterol code is REJECTED", cholesterol=9)
rejects("Non-numeric input is REJECTED", age="not-a-number")
rejects("Implausible pulse pressure is REJECTED", ap_hi=200, ap_lo=60)
rejects("A negative weight is REJECTED", weight=-70)
rejects("A decimal age is REJECTED", age=45.7)
rejects("An empty field is REJECTED", age="")
rejects("A missing patient name is REJECTED", patient_name="")
rejects("Digits in the patient name are REJECTED", patient_name="Ali123")
rejects("A one-character name is REJECTED", patient_name="A")
rejects("A patient ID without the PT- prefix is REJECTED", patient_id="1001")
rejects("A patient ID with the wrong prefix is REJECTED", patient_id="ID-1001")
rejects("Letters after PT- are REJECTED", patient_id="PT-ABCD")
rejects("Too few digits after PT- are REJECTED", patient_id="PT-12")
rejects("A bare 'PT-' is REJECTED", patient_id="PT-")
rejects("Spaces in the patient ID are REJECTED", patient_id="PT 1001")


def accepts(label, **changes):
    """Assert that the assessment form is accepted when a field holds a valid value."""
    payload = dict(good)
    payload.update(changes)
    valid, errs = ml_engine.validate_form(payload)
    check(label, valid, str(errs))


accepts("A correct PT- reference is accepted", patient_id="PT-1001")
accepts("A long PT- reference is accepted", patient_id="PT-123456789")
accepts("A lowercase pt- reference is accepted", patient_id="pt-1001")
accepts("A blank patient ID is accepted (optional)", patient_id="")

# Every documented boundary must be reachable with a compatible partner value
accepts("Age 18 (minimum) is accepted", age=18)
accepts("Age 120 (maximum) is accepted", age=120)
accepts("Height 100 cm (minimum) is accepted", height=100, weight=25)
accepts("Height 250 cm (maximum) is accepted", height=250, weight=150)
accepts("Weight 200 kg (maximum) is accepted", height=200, weight=200)
accepts("Systolic 250 (maximum) is accepted", ap_hi=250, ap_lo=170)
accepts("Diastolic 40 (minimum) is accepted", ap_hi=120, ap_lo=40)
rejects("Over-long clinical notes are REJECTED", notes="x" * 600)
rejects("An invalid gender code is REJECTED", gender=7)

# Risk bands now carry a mood emoji for the result page
for prob, band in [(0.20, "LOW RISK"), (0.55, "MODERATE RISK"), (0.85, "HIGH RISK")]:
    level = ml_engine.get_risk_level(prob)
    check(f"{band} has a mood emoji", bool(level.get("emoji")))
check("Low and high risk use DIFFERENT emojis",
      ml_engine.get_risk_level(0.20)["emoji"] != ml_engine.get_risk_level(0.85)["emoji"])
check("BMI categories carry an emoji", bool(ml_engine.get_bmi_category(22).get("emoji")))


# ── Registration rules ────────────────────────────────────────────
print("\n[5b] REGISTRATION VALIDATION")

try:
    from routes.auth_routes import validate_registration

    def reg(username="goodname1", fullname="Ali Khan", email="a@b.com",
            password="Pass1234", confirm=None):
        """Build a registration payload, overriding only the fields a test cares about."""
        return validate_registration(username, fullname, email, password,
                                     confirm if confirm is not None else password)

    check("A valid registration is accepted", not reg())
    check("A password with no number is REJECTED", reg(password="Password"))
    check("A password with no letter is REJECTED", reg(password="12345678"))
    check("A password under 8 characters is REJECTED", reg(password="Ab1"))
    check("A password containing a space is REJECTED", reg(password="Pass 123"))
    check("A password equal to the username is REJECTED",
          reg(username="Pass1234", password="Pass1234"))
    check("Mismatched passwords are REJECTED", reg(confirm="Different9"))
    check("Symbols in the username are REJECTED", reg(username="bad-name!"))
    check("A username starting with a digit is REJECTED", reg(username="9abcdef"))
    check("Digits in the full name are REJECTED", reg(fullname="Ali123"))
    check("An invalid email is REJECTED", reg(email="notanemail"))
    check("An email with no dot is REJECTED", reg(email="a@bcom"))

    # -- self-registration is PATIENT ONLY --
    # A Doctor account can read every patient's records, so the public
    # register page must never hand that role out. Doctor accounts are
    # created only by an administrator (Admin -> Doctors).
    from routes.auth_routes import SELF_SERVICE_ROLES

    def reg_role(role):
        """Build a registration payload for one particular role."""
        return validate_registration("goodname1", "Ali Khan", "a@b.com",
                                     "Pass1234", "Pass1234", role=role)

    check("Patient is the ONLY self-service role",
          SELF_SERVICE_ROLES == ("Patient",), str(SELF_SERVICE_ROLES))
    check("Registering as a Patient is accepted", not reg_role("Patient"))
    check("Registering as a Doctor is REJECTED (patient-data access)",
          reg_role("Doctor"))
    check("Registering as Admin is REJECTED (privilege escalation)",
          reg_role("Admin"))
    check("Registering as SuperAdmin is REJECTED", reg_role("SuperAdmin"))
    check("An unknown role is REJECTED", reg_role("Wizard"))

    # The admin "create doctor" page uses the same field rules but widens
    # the allowed roles - that path must keep working.
    check("Admin path CAN validate a Doctor account",
          not validate_registration("gooddoc1", "Dr Sara Ali", "s@b.com",
                                    "Pass1234", "Pass1234",
                                    role="Doctor", allowed_roles=("Doctor",)))

    # -- nobody may register twice --
    existing = auth_db.get_all_users()
    if existing:
        taken_username = existing[0]["username"]
        taken_email = next((u["email"] for u in existing if u["email"]), None)

        check("A duplicate username is REJECTED",
              validate_registration(taken_username, "Ali Khan", "brand.new@x.com",
                                    "Pass1234", "Pass1234"))
        check("A duplicate username in a different case is REJECTED",
              validate_registration(taken_username.upper(), "Ali Khan",
                                    "brand.new2@x.com", "Pass1234", "Pass1234"))
        if taken_email:
            check("A duplicate EMAIL is REJECTED (no registering twice)",
                  validate_registration("brandnewname", "Ali Khan", taken_email,
                                        "Pass1234", "Pass1234"))
            check("A duplicate email in a different case is REJECTED",
                  validate_registration("brandnewname2", "Ali Khan",
                                        taken_email.upper(), "Pass1234", "Pass1234"))
        check("auth_db.email_exists() finds a used address",
              auth_db.email_exists(taken_email) if taken_email else True)
        check("auth_db.email_exists() ignores an unused address",
              not auth_db.email_exists("definitely.not.used@nowhere.example"))
        check("auth_db.username_exists() finds a taken username",
              auth_db.username_exists(taken_username))
except ImportError as e:
    print(f"  SKIP  Registration rules ({e})")

# Risk banding thresholds used by the coloured gauge
check("0.20 is banded LOW", ml_engine.get_risk_level(0.20)["label"] == "LOW RISK")
check("0.55 is banded MODERATE", ml_engine.get_risk_level(0.55)["label"] == "MODERATE RISK")
check("0.85 is banded HIGH", ml_engine.get_risk_level(0.85)["label"] == "HIGH RISK")

# ── Screening threshold ───────────────────────────────────────────
# The decision cut-off is deliberately 0.40, not the textbook 0.50: in a
# screening tool a missed disease costs more than a false alarm. It must
# also EQUAL the Low/Moderate band boundary, so the yes/no verdict and
# the coloured badge can never contradict each other (a patient at 0.45
# used to be "not at risk" yet wear a MODERATE badge).
from config import Config as _Cfg   # noqa: E402
check("Screening threshold is 0.40", _Cfg.POSITIVE_THRESHOLD == 0.40,
      f"got {_Cfg.POSITIVE_THRESHOLD}")
check("Decision threshold EQUALS the Low/Moderate band boundary",
      _Cfg.POSITIVE_THRESHOLD == _Cfg.RISK_LOW_MAX,
      f"{_Cfg.POSITIVE_THRESHOLD} vs {_Cfg.RISK_LOW_MAX}")
check("A probability just under the threshold is banded LOW (consistent)",
      ml_engine.get_risk_level(_Cfg.POSITIVE_THRESHOLD - 0.001)["label"] == "LOW RISK")
check("A probability at the threshold is NOT banded LOW (consistent)",
      ml_engine.get_risk_level(_Cfg.POSITIVE_THRESHOLD)["label"] != "LOW RISK")

# WHO BMI categories
check("BMI 17 is Underweight", ml_engine.get_bmi_category(17)["label"] == "Underweight")
check("BMI 22 is Normal weight", ml_engine.get_bmi_category(22)["label"] == "Normal weight")
check("BMI 27 is Overweight", ml_engine.get_bmi_category(27)["label"] == "Overweight")
check("BMI 35 is Obese", ml_engine.get_bmi_category(35)["label"] == "Obese")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 6 - Database
# ══════════════════════════════════════════════════════════════════
print("\n[6] DATABASE")

db_path = os.path.join(BASE_DIR, "heart_disease.db")
if os.path.exists(db_path):
    connection = sqlite3.connect(db_path)
    tables = [row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    connection.close()

    for table_name in ["users", "predictions", "system_logs", "training_runs"]:
        check(f"Table '{table_name}' exists", table_name in tables)

    _, status_wrong = auth_db.validate_login("admin", "definitely_wrong")
    check("Wrong password is rejected", status_wrong == "invalid")

    _, status_missing = auth_db.validate_login("no_such_user_12345", "x")
    check("Unknown username is rejected", status_missing == "invalid")

    _, status_injection = auth_db.validate_login("admin' OR '1'='1", "x")
    check("SQL injection attempt is rejected", status_injection == "invalid")

    connection = sqlite3.connect(db_path)
    survived = connection.execute(
        "SELECT name FROM sqlite_master WHERE name='users'").fetchone()
    connection.close()
    check("Users table survived the injection attempt", survived is not None)
else:
    check("heart_disease.db exists", False)


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 7 - Flask routes and access control
# ══════════════════════════════════════════════════════════════════
print("\n[7] FLASK ROUTES AND ACCESS CONTROL")

try:
    import re
    import logging
    logging.disable(logging.ERROR)      # keep expected errors out of the output
    from routes import create_app

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    def csrf_from(path):
        """Pull the hidden CSRF token out of a rendered page so a POST can be signed."""
        page = client.get(path).get_data(as_text=True)
        found = re.search(r'name="csrf_token" value="([^"]+)"', page)
        return found.group(1) if found else ""

    # -- anonymous visitors are redirected to the login page --
    for guarded in ["/dashboard", "/diagnosis", "/history", "/profile"]:
        response = client.get(guarded)
        check(f"{guarded} requires sign-in", response.status_code == 302,
              f"got {response.status_code}")

    # -- the login page itself is public --
    check("/login is publicly reachable", client.get("/login").status_code == 200)

    # -- CSRF must block a form post with no token --
    no_token = client.post("/login", data={"username": "admin", "password": "admin123"})
    check("POST without a CSRF token is blocked", no_token.status_code == 400,
          f"got {no_token.status_code}")

    # -- sign in as a patient-level user and check role separation --
    token = csrf_from("/login")
    client.post("/login", data={"csrf_token": token,
                                "username": "doctor", "password": "doctor123"})

    check("A signed-in user reaches the dashboard",
          client.get("/dashboard").status_code == 200)
    check("A doctor reaches the patient list",
          client.get("/doctor/patients").status_code == 200)
    check("A doctor is BLOCKED from the admin area (403)",
          client.get("/admin/users").status_code == 403)

    client.get("/logout")

    # -- the single administrator tier reaches everything --
    token = csrf_from("/login")
    client.post("/login", data={"csrf_token": token,
                                "username": "admin", "password": "admin123"})
    for admin_path in ["/admin/users", "/admin/analytics", "/admin/model-performance",
                       "/admin/audits", "/admin/training", "/admin/dataset",
                       "/admin/doctors", "/admin/predictions"]:
        check(f"Admin reaches {admin_path}",
              client.get(admin_path).status_code == 200)

    check("A missing page returns 404", client.get("/no-such-page").status_code == 404)
    client.get("/logout")

    # ── Login rate limiting ───────────────────────────────────────
    # After LOGIN_MAX_FAILURES wrong passwords the account/IP pair must
    # be locked out, and a successful login must clear the counters.
    from routes.security import (login_blocked_seconds, record_login_failure,
                                 clear_login_failures, LOGIN_MAX_FAILURES)

    with app.test_request_context():
        probe = "ratelimit_probe_user"
        clear_login_failures(probe)
        check("A fresh account is not rate limited",
              login_blocked_seconds(probe) == 0)

        for _ in range(LOGIN_MAX_FAILURES - 1):
            record_login_failure(probe)
        check("One failure below the limit is still allowed",
              login_blocked_seconds(probe) == 0)

        record_login_failure(probe)          # the failure that trips the lock
        blocked_for = login_blocked_seconds(probe)
        check(f"{LOGIN_MAX_FAILURES} failures trigger a lockout",
              blocked_for > 0, f"blocked for {blocked_for}s")

        clear_login_failures(probe)
        check("A successful login clears the lockout",
              login_blocked_seconds(probe) == 0)

        # The store must not keep empty keys around - an attacker spraying
        # made-up usernames would otherwise grow it without bound.
        from routes import security as _sec_mod
        check("Cleared counters leave no key behind (no unbounded growth)",
              not any(probe in key for key in _sec_mod._login_failures))

    # The lockout must also bite over real HTTP - and because failures
    # count per-IP too, the test client (one IP) locks itself out.
    with app.test_request_context():
        clear_login_failures("admin")        # start clean for this client IP
    for _ in range(LOGIN_MAX_FAILURES):
        token = csrf_from("/login")
        client.post("/login", data={"csrf_token": token,
                                    "username": "admin",
                                    "password": "definitely_wrong"})
    token = csrf_from("/login")
    page = client.post("/login", data={"csrf_token": token,
                                       "username": "admin",
                                       "password": "admin123"}
                       ).get_data(as_text=True)
    check("The right password is refused DURING a lockout",
          "Too many failed sign-in attempts" in page)

    # Leave no lock behind for anything that runs after this group. The
    # helper keys on the request's IP, so wipe the store directly instead.
    from routes import security as _security
    _security._login_failures.clear()

except ImportError as e:
    print(f"  SKIP  Flask route tests ({e})")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 8 - Single administrator tier
# ------------------------------------------------------------------
# SuperAdmin has been merged into Admin. There is now one admin role
# holding every privilege, with a guard so the last one cannot be
# removed and lock everybody out.
# ══════════════════════════════════════════════════════════════════
print("\n[8] SINGLE ADMINISTRATOR TIER")

from routes import ROLE_ADMIN, ADMIN_ROLES, ALL_ROLES   # noqa: E402

check("Only one administrator role exists", ADMIN_ROLES == (ROLE_ADMIN,),
      f"got {ADMIN_ROLES}")
check("The role list is Patient, Doctor, Admin",
      ALL_ROLES == ["Patient", "Doctor", "Admin"], f"got {ALL_ROLES}")
check("No SuperAdmin constant remains",
      not hasattr(__import__("routes"), "ROLE_SUPERADMIN"))

# No account should still carry the old role
roles_in_db = {u["role"] for u in auth_db.get_all_users()}
check("No SuperAdmin accounts remain in the database",
      "SuperAdmin" not in roles_in_db, f"found roles {sorted(roles_in_db)}")
check("At least one active administrator exists",
      auth_db.count_active_admins() >= 1)

# The last-admin guard must refuse to remove the final administrator
try:
    from routes import admin as admin_routes   # _can_manage lives in routes/admin/__init__.py

    admins = [u for u in auth_db.get_all_users()
              if u["role"] == ROLE_ADMIN and not u["is_banned"]]
    if len(admins) >= 2:
        actor, target = admins[0], admins[1]
        app2 = create_app(); app2.config["TESTING"] = True
        with app2.test_request_context():
            from flask import session as flask_session
            flask_session["user"] = actor

            # With several admins present, managing another one is allowed
            allowed, _ = admin_routes._can_manage(target, "deleted")
            check("An admin CAN manage another admin", allowed)

            # Nobody may act on their own account
            allowed, reason = admin_routes._can_manage(actor, "deleted")
            check("An admin cannot act on their own account", not allowed)

            # Pretend only one admin is left and confirm the guard fires
            real_counter = auth_db.count_active_admins
            auth_db.count_active_admins = lambda: 1
            try:
                allowed, reason = admin_routes._can_manage(target, "deleted")
                check("The LAST admin is protected from deletion", not allowed, reason)
                check("The refusal explains why",
                      "only active administrator" in reason)
            finally:
                auth_db.count_active_admins = real_counter
    else:
        print("  SKIP  Last-admin guard (needs two admin accounts to test)")
except Exception as e:
    check("Last-admin guard test ran", False, str(e)[:70])


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f" RESULTS:  {passed} passed,  {failed} failed")
if failures:
    print("\n Failed tests:")
    for name in failures:
        print(f"   - {name}")
else:
    print(" All tests passed.")
print("=" * 60)
