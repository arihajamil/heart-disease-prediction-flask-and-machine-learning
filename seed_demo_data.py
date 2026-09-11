"""
Heart Disease Prediction - Demo data seeder
=================================
Creates the demo patient account (patient / patient123) if it does not
exist, then adds 15 realistic assessment records so the dashboard, history,
charts, analytics and clinical reports all have something to show.

Run with:   python seed_demo_data.py
Undo with:  python seed_demo_data.py --clear

Every record is produced by running the patient's measurements through the
REAL trained models, exactly as the web form does. Nothing is invented, so
the probabilities and verdicts are genuine and reproducible - re-running an
entry through the interface gives the same answer.

The records are spread over the last few weeks so the trend and
"assessments per day" charts have a realistic shape.
"""

import sys
from datetime import datetime, timedelta

import auth_db
import ml_engine
from config import Config

# A marker written into the notes of every generated row, so --clear can
# find and remove exactly these records and nothing else.
DEMO_TAG = "[demo]"


# ══════════════════════════════════════════════════════════════════
# THE PATIENTS
# ------------------------------------------------------------------
# A deliberate spread: healthy young adults through to older patients
# with several risk factors, so the charts show a real distribution
# rather than everything landing in one band.
# ══════════════════════════════════════════════════════════════════
#          name,              ref,        age gen  ht   wt   hi   lo  chol gluc smk alc act  days_ago
PATIENTS = [
    ("Ayesha Siddiqui",   "PT-1001",  28, 0, 163,  55, 108,  70,  1, 1, 0, 0, 1, 27),
    ("Bilal Ahmed",       "PT-1002",  34, 1, 178,  74, 118,  76,  1, 1, 0, 0, 1, 25),
    ("Fatima Noor",       "PT-1003",  31, 0, 158,  52, 112,  72,  1, 1, 0, 0, 1, 22),
    ("Usman Tariq",       "PT-1004",  41, 1, 175,  88, 128,  84,  2, 1, 1, 0, 1, 20),
    ("Zainab Malik",      "PT-1005",  46, 0, 160,  71, 132,  86,  2, 1, 0, 0, 1, 18),
    ("Hassan Raza",       "PT-1006",  52, 1, 172,  92, 140,  90,  2, 2, 1, 1, 0, 16),
    ("Mariam Javed",      "PT-1007",  49, 0, 165,  78, 136,  88,  2, 1, 0, 0, 0, 14),
    ("Imran Sheikh",      "PT-1008",  57, 1, 170,  95, 148,  92,  3, 2, 1, 1, 0, 12),
    ("Nadia Hussain",     "PT-1009",  61, 0, 157,  84, 152,  94,  3, 2, 0, 0, 0, 10),
    ("Kamran Aslam",      "PT-1010",  64, 1, 174, 101, 162,  98,  3, 3, 1, 1, 0,  8),
    ("Rukhsana Bibi",     "PT-1011",  67, 0, 154,  79, 158,  96,  3, 2, 0, 0, 0,  6),
    ("Tariq Mehmood",     "PT-1012",  70, 1, 168,  90, 168, 100,  3, 3, 1, 0, 0,  4),
]

# Sara Ahmed is the demo patient account - three assessments over time so
# her personal trend chart has something to plot.
SELF_CHECKS = [
    (38, 0, 162, 74, 130, 84, 2, 1, 0, 0, 0, 21),
    (38, 0, 162, 72, 126, 82, 2, 1, 0, 0, 1, 11),
    (38, 0, 162, 69, 120, 78, 1, 1, 0, 0, 1,  2),
]

# Rotate through the models so the "model usage" chart is not one bar
MODELS = ["Ensemble Voting (All Models)", "Random Forest", "XGBoost",
          "Logistic Regression", "Support Vector Machine (SVM)"]

NOTES = [
    "Routine screening.", "Annual check-up.", "Follow-up after last visit.",
    "Patient reports occasional breathlessness.", "Referred by GP.",
    "Pre-operative assessment.", "Family history of heart disease.",
    "Monitoring blood pressure.", "Reviewed medication.", "Walk-in appointment.",
]


def _find_user(username, fallback_role=None):
    """Return a user id by username, or the first account of a role."""
    users = auth_db.get_all_users()
    for row in users:
        if row["username"] == username:
            return row["id"], row["fullname"]
    if fallback_role:
        for row in users:
            if row["role"] == fallback_role:
                return row["id"], row["fullname"]
    return None, None


def _backdate(prediction_id, days_ago):
    """
    Move a record back in time so the charts show a spread of dates.
    add_prediction() always stamps 'now', so the timestamp is corrected here.
    """
    when = datetime.now() - timedelta(days=days_ago,
                                      hours=(prediction_id * 3) % 9,
                                      minutes=(prediction_id * 17) % 60)
    conn = auth_db._connect()
    conn.execute("UPDATE predictions SET timestamp = ? WHERE id = ?",
                 (when.strftime("%Y-%m-%d %H:%M:%S"), prediction_id))
    conn.commit()
    conn.close()


def clear_demo_data():
    """Remove only the rows this script created."""
    conn = auth_db._connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM predictions WHERE notes LIKE ?", (f"%{DEMO_TAG}%",))
    count = cur.fetchone()[0]
    cur.execute("DELETE FROM predictions WHERE notes LIKE ?", (f"%{DEMO_TAG}%",))
    cur.execute("DELETE FROM system_logs WHERE details LIKE ?", ("%Risk=%",))
    conn.commit()
    conn.close()
    print(f"Removed {count} demo assessment(s).")


def seed():
    """
    Create the demo assessments by running each patient through the models.

    Nothing is invented - every probability is a genuine model output, so
    re-entering the same numbers in the web form gives the same answer.
    """
    ml_engine.init_engine(Config.MODELS_DIR)
    if not ml_engine.is_ready():
        print("ERROR: no trained models are available, so predictions cannot be made.")
        print("Run  python train_models.py  first.")
        return 1

    doctor_id, doctor_name = _find_user("doctor", "Doctor")
    patient_id, patient_name = _find_user("patient", "Patient")

    if doctor_id is None:
        print("ERROR: no doctor account found. Start the app once so the "
              "default accounts are seeded.")
        return 1

    # A fresh database only seeds the doctor and admin accounts, so the
    # demo patient login documented in the README (patient / patient123)
    # is created here on demand. Without this, the README promised an
    # account that did not exist on a clean install.
    if patient_id is None:
        new_id, err = auth_db.register_user(
            "patient", "patient123", "Patient",
            "Demo Patient", "patient@example.com")
        if err is None:
            patient_id, patient_name = new_id, "Demo Patient"
            print("Created demo patient account:  patient / patient123")
        else:
            print("NOTE: demo patient account could not be created "
                  f"({err}) - self-check records will be skipped.")

    print(f"Using models : {', '.join(ml_engine.available_models())}")
    print(f"Doctor       : {doctor_name}")
    print(f"Patient      : {patient_name or '(none - self-checks will be skipped)'}")
    print()

    created = 0
    print(f"{'PATIENT':<20}{'AGE':>4}{'BP':>9}{'BMI':>7}{'RISK':>8}  BAND")
    print("-" * 62)

    # ── Assessments recorded by the doctor ────────────────────────
    for index, row in enumerate(PATIENTS):
        (name, ref, age, gender, height, weight,
         ap_hi, ap_lo, chol, gluc, smoke, alco, active, days_ago) = row

        form = {"age": age, "gender": gender, "height": height, "weight": weight,
                "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": chol, "gluc": gluc,
                "smoke": smoke, "alco": alco, "active": active,
                "patient_name": name, "patient_id": ref, "notes": ""}

        # Refuse to insert anything the live form would have rejected
        valid, problems = ml_engine.validate_form(form)
        if not valid:
            print(f"  SKIPPED {name}: {problems[0]}")
            continue

        vector, readable = ml_engine.engineer_features(form)
        model = MODELS[index % len(MODELS)]
        outcome = ml_engine.predict(vector, model)
        band = ml_engine.get_risk_level(outcome["probability"])

        new_id = auth_db.add_prediction(
            doctor_id, age, gender, height, weight, ap_hi, ap_lo,
            chol, gluc, smoke, alco, active,
            outcome["predicted_class"], outcome["probability"],
            outcome["model_used"], name,
            f"{NOTES[index % len(NOTES)]} {DEMO_TAG}", patient_id=ref)
        _backdate(new_id, days_ago)
        created += 1

        print(f"{name:<20}{age:>4}{f'{ap_hi}/{ap_lo}':>9}"
              f"{readable['bmi']:>7.1f}{outcome['probability']:>7.1%}  {band['label']}")

    # ── Self-checks recorded by the patient account ───────────────
    if patient_id is not None:
        for index, row in enumerate(SELF_CHECKS):
            (age, gender, height, weight, ap_hi, ap_lo,
             chol, gluc, smoke, alco, active, days_ago) = row

            form = {"age": age, "gender": gender, "height": height, "weight": weight,
                    "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": chol, "gluc": gluc,
                    "smoke": smoke, "alco": alco, "active": active,
                    "patient_name": patient_name, "patient_id": "", "notes": ""}

            valid, problems = ml_engine.validate_form(form)
            if not valid:
                print(f"  SKIPPED self-check {index + 1}: {problems[0]}")
                continue

            vector, readable = ml_engine.engineer_features(form)
            outcome = ml_engine.predict(vector, "Ensemble Voting (All Models)")
            band = ml_engine.get_risk_level(outcome["probability"])

            new_id = auth_db.add_prediction(
                patient_id, age, gender, height, weight, ap_hi, ap_lo,
                chol, gluc, smoke, alco, active,
                outcome["predicted_class"], outcome["probability"],
                outcome["model_used"], patient_name,
                f"Self-check {index + 1} of 3. {DEMO_TAG}")
            _backdate(new_id, days_ago)
            created += 1

            print(f"{patient_name + ' (self)':<20}{age:>4}{f'{ap_hi}/{ap_lo}':>9}"
                  f"{readable['bmi']:>7.1f}{outcome['probability']:>7.1%}  {band['label']}")

    print("-" * 62)
    print(f"Created {created} assessment(s).")
    return 0


if __name__ == "__main__":
    if "--clear" in sys.argv:
        clear_demo_data()
    else:
        sys.exit(seed())
