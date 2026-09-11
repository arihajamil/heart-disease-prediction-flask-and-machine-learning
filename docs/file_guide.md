# File Guide

Every file in Heart Disease Prediction, what it does, and what depends on it.

If you are looking for *where* something happens, this is the fastest way to find
it. If you want to know *why* it is arranged this way, read
[architecture.md](architecture.md) instead.

---

## Contents

1. [Entry point and configuration](#1-entry-point-and-configuration)
2. [The core modules](#2-the-core-modules)
3. [The routes package](#3-the-routes-package)
4. [Templates](#4-templates)
5. [Static assets](#5-static-assets)
6. [The trained model files](#6-the-trained-model-files)
7. [The research folder](#7-the-research-folder)
8. [Tests and verification](#8-tests-and-verification)
9. [Data and generated files](#9-data-and-generated-files)
10. [Dependency map](#10-dependency-map)

---

## 1. Entry point and configuration

### `run.py` — 32 lines
The development entry point. Calls `create_app()` and starts Flask's built-in
server on port 5000.

It reads three environment variables (`HDP_PORT`, `HDP_ENV`) and
prints a small banner so it is obvious the server started. Nothing else lives
here — keeping the entry point trivial means the application can also be served
by a production WSGI server (`waitress-serve "run:app"`) with no code change.

**Run it with:** `python run.py`

### `config.py` — 163 lines
The single source of truth for every tunable value in the project. Nothing else
in the codebase hard-codes a number that appears here.

| Section | What it holds |
|---|---|
| Security | `SECRET_KEY` from the environment, session-cookie hardening, 8-hour session lifetime |
| Paths | `DB_PATH`, `MODELS_DIR`, `DATA_PATH`, `STATIC_IMG_DIR` — all derived from `BASE_DIR`, so the project is portable |
| Risk banding & decision | `RISK_LOW_MAX` (0.40), `RISK_MODERATE_MAX` (0.70), `POSITIVE_THRESHOLD` (0.40 — the screening cut-off, equal to the Low/Moderate boundary so verdict and badge always agree) |
| `FEATURE_ORDER` | The 15 feature names in the exact order `scaler.pkl` expects |
| `MODEL_FILES` | Which pickle holds which classifier |
| `FIELD_RANGES` | Medically plausible bounds for age, height, weight and both blood pressures |
| `CHOICE_FIELDS` | The only valid codes for each dropdown |
| Cross-field limits | Pulse-pressure range, plausible BMI range, BMI clipping range |
| Display labels | The 1–3 clinical scale wording |

`get_config()` at the bottom picks `DevConfig` or `ProdConfig` from
`HDP_ENV`. The only difference is `DEBUG`.

> **Watch out:** in production mode Jinja stops auto-reloading templates. If you
> edit an HTML file and the browser keeps showing the old version, check which
> mode the server is running in.

---

## 2. The core modules

### `ml_engine.py` — 646 lines
Everything to do with producing a prediction. Imported by the route modules; it
imports nothing from them, so it can be used from a script or a test with no
Flask application present.

| Section | Functions |
|---|---|
| Startup | `init_engine`, `get_load_warnings`, `is_ready`, `available_models`, `get_feature_names` |
| Config / results | `_read_json`, `load_config`, `load_results`, `best_model_name` |
| Feature engineering | `compute_engineered_features`, `engineer_features` |
| Validation | `validate_form` |
| Prediction | `predict` |
| Presentation | `get_risk_level`, `get_bmi_category`, `get_agreement_summary`, `get_lifestyle_tips`, `get_feature_importances` |

Three long comment blocks in this file explain the parts an examiner is most
likely to ask about: the meaning and order of all 15 features, why the training
scaler is reused rather than refitted, and what `predict_proba` actually returns.

**Models are loaded once**, at application startup, into module-level variables
(`_SCALER`, `_MODELS`). A prediction therefore costs no disk access. Each model
is loaded inside its own `try`, so one damaged pickle or one missing library
disables that model alone instead of the whole page.

### `auth_db.py` — 617 lines
The only module that talks to SQLite. Every query in the project goes through a
function here, and every one of them is parameterised.

| Group | Functions |
|---|---|
| Passwords | `hash_password`, `verify_password` |
| Connection & schema | `_connect`, `init_db` |
| Authentication | `validate_login`, `register_user`, `is_legacy_hash` |
| Users | `get_all_users`, `get_user_by_id`, `email_exists`, `username_exists`, `count_active_admins`, `update_user_role`, `ban_user`, `unban_user`, `delete_user`, `update_user_profile` |
| Predictions | `add_prediction`, `get_predictions`, `get_prediction_by_id`, `count_predictions_by_user`, `delete_prediction`, `clear_all_predictions` |
| Logging | `log_activity`, `get_system_logs`, `clear_system_logs` |
| Training | `log_training_run`, `get_training_runs` |

Passwords use PBKDF2-HMAC-SHA256 with a 16-byte random salt per user and 260,000
iterations, stored as `pbkdf2_sha256$iterations$salt$hash`. Old plain SHA-256
hashes from an earlier version still verify and are silently upgraded on the next
successful sign-in, so no account was locked out by the change.

`init_db()` also runs the small schema migrations: adding the `patient_id`
column, folding any surviving `SuperAdmin` rows into `Admin`, and dropping the
removed `specialisation` column.

### `seed_demo_data.py` — 234 lines
Creates the demo patient account and fifteen sample assessments.

Every one of those fifteen probabilities is a genuine model output — the script
runs each fictional patient through the real pipeline rather than inventing a
number. Typing the same measurements into the web form gives the same answer,
which matters if an examiner decides to check.

---

## 3. The routes package

### `routes/__init__.py` — 142 lines
The application factory. It:

1. creates the Flask app and loads the config class,
2. calls `auth_db.init_db()` and `ml_engine.init_engine()` once,
3. registers the four blueprints,
4. registers the CSRF check that runs before every POST,
5. registers the 400 / 403 / 404 / 500 error handlers,
6. exposes `generate_csrf_token` to every template.

It re-exports the names from `security.py`, so older imports such as
`from routes import login_required` still work.

### `routes/security.py` — 212 lines
The role model and the access-control layer, extracted so it can be read on its
own.

- **Roles:** `ROLE_PATIENT`, `ROLE_DOCTOR`, `ROLE_ADMIN`, plus `ALL_ROLES`.
  There is one administrator tier holding every privilege.
- **Helpers:** `current_user`, `is_admin`, `is_doctor`.
- **Decorators:** `@login_required` redirects anonymous visitors to the login
  page, remembering where they were going. `@role_required("doctor", "admin")`
  returns a proper 403 page for a signed-in user with the wrong role.
- **CSRF:** a `secrets.token_hex(32)` token per session, compared with
  `secrets.compare_digest`. Same idea as Flask-WTF, no extra dependency.
- **Login rate limiting:** `login_blocked_seconds`, `record_login_failure`,
  `clear_login_failures`. Five wrong passwords against one username or from
  one IP within 15 minutes lock sign-in for 10 minutes; a correct login
  clears the counters, and keys with no recent failures are dropped
  entirely, so spraying made-up usernames cannot grow the store without
  bound. In-process and thread-safe — the right weight for a single-instance
  SQLite app, and these three functions are the only place a shared-store
  version would touch.

### `routes/auth_routes.py` — 294 lines
`/login`, `/register`, `/logout`. Also holds `validate_registration()`, the
shared rule set that both the public form and the admin doctor-creation page use,
so an account made by an administrator can never be weaker than a self-registered
one.

Registration is deliberately restricted: every account the public form creates
is a **Patient** — the server hard-codes the role and never reads a posted one,
because a Doctor account can see every patient's records and an administrator
can do anything at all. Doctor logins are created only on the admin Doctors
page, which reuses the same `validate_registration()` rules with
`allowed_roles=(ROLE_DOCTOR,)`. Sign-in is rate limited (see `security.py`),
and on success the session is cleared before the user is stored, so a new
session id is issued on every privilege change — that is what defeats session
fixation.

### `routes/patient_routes.py` — 448 lines
The main user journey: `/dashboard`, `/diagnosis`, `/predict`, `/history`,
`/report/<id>`, `/profile`, `/help`.

`predict()` is the heart of the application and runs in a fixed order: validate
on the server → engineer features → scale → run the models → save → render. A
database failure at the save step is reported but does not throw away the result
that was already computed.

One rule runs through the whole module:

```python
_scope_user_id(user)   # returns user["id"] for a patient, None for staff
```

`None` means "no filter", so the same query serves a patient looking at their own
history and a doctor looking at everyone's.

### `routes/doctor_routes.py` — 177 lines
`/doctor/patients`, `/doctor/patients/<id>`, `/doctor/reports`. Read-only
oversight for clinicians. Guarded by `@role_required("doctor", "admin")`, so
administrators can see these pages too.

### `routes/admin/` — 898 lines across six modules
One blueprint (`admin_bp`), split by subject. Every endpoint name is unchanged
from when this was a single file: `admin.users`, `admin.change_role` and the rest
all still resolve.

| Module | Lines | Responsibility |
|---|---:|---|
| `__init__.py` | 85 | Creates `admin_bp`, holds the shared background-training state, and defines `_can_manage()` — the last-admin guard |
| `users.py` | 230 | Account list, role changes, suspension, deletion, doctor creation |
| `predictions.py` | 94 | Every saved assessment, with filters, single delete and a guarded bulk clear |
| `data.py` | 138 | Dataset inspection and the system-wide analytics charts |
| `models.py` | 243 | Metric tables, confusion matrices, feature importance, SHAP image, background retraining |
| `audits.py` | 81 | The paginated activity log |

`_can_manage()` enforces two rules that appear in every state-changing route:
nobody may act on their own account, and the last active administrator cannot be
demoted, suspended or deleted. Without the second rule the system could be left
with nobody able to administer it.

The submodule imports sit at the **bottom** of `routes/admin/__init__.py`,
because `admin_bp` must exist before any module tries to decorate a function with
it.

---

## 4. Templates

`templates/base.html` (218 lines) is the layout every page extends: the sidebar,
the flash-message area and the footer disclaimer.

| Folder | Files |
|---|---|
| `auth/` | `login.html`, `register.html` |
| `patient/` | `diagnosis.html`, `result.html`, `history.html`, `profile.html`, `help.html` |
| `doctor/` | `patients.html`, `patient_detail.html`, `reports.html` |
| `admin/` | `users.html`, `doctors.html`, `predictions.html`, `dataset.html`, `analytics.html`, `model_performance.html`, `training.html`, `audits.html` |
| `shared/` | `dashboard.html` — one template, three role-dependent views |
| `errors/` | `400.html`, `403.html`, `404.html`, `500.html` |

> A note for anyone editing `base.html`: `request.endpoint` is `None` on a 404
> page, so any test against it must be written `(request.endpoint or '')`.
> Forgetting this turns every 404 into a 500.

---

## 5. Static assets

| File | Lines | Purpose |
|---|---:|---|
| `css/style.css` | 763 | The complete theme. Hand-written — no Bootstrap |
| `css/fonts.css` | 110 | `@font-face` rules for the self-hosted Inter font |
| `js/main.js` | 155 | Sidebar toggle, flash auto-dismiss, destructive-action confirmations |
| `js/diagnosis.js` | 276 | Live BMI as you type, range checks, blood-pressure sanity — mirrors the server rules |
| `js/register.js` | 141 | Password strength meter and live field validation (registration is patient-only, so there is no role switch) |
| `js/charts.js` | 311 | Thin wrappers around Chart.js for every chart in the app, including the calibration curves |
| `js/chart.min.js` | — | Chart.js 4.4.1, vendored |
| `fonts/inter-0.woff2`, `inter-1.woff2` | — | The font itself |
| `img/heart_bg.png` | — | The login and register artwork |
| `img/shap_summary.png` | — | Written by the training pipeline, displayed on Model Performance |

**Everything is served from this folder.** No CDN, no Google Fonts. Once the pip
packages are installed the application needs no internet connection at all, so it
runs on a locked-down examination machine without blank charts or fallback fonts.

The browser-side validation in `diagnosis.js` is a convenience, never a control.
`ml_engine.validate_form()` re-checks all of it on the server, because anything
in the browser can be bypassed.

---

## 6. The trained model files

`models/` is written by the training pipeline and read by `ml_engine.py`.

| File | Size | Contents |
|---|---:|---|
| `scaler.pkl` | 1 KB | The fitted `StandardScaler` — one mean and one standard deviation per feature |
| `logistic_regression.pkl` | 798 B | 15 coefficients and an intercept |
| `svm.pkl` | 2 KB | A calibrated linear SVM |
| `decision_tree.pkl` | 19 KB | A depth-limited tree |
| `random_forest.pkl` | 30 MB | 200 fully grown trees — big because every split of every tree is stored |
| `xgboost.pkl` | 1.2 MB | The gradient-boosted ensemble |
| `features.json` | 161 B | The 15 feature names, in order. The authoritative copy |
| `config.json` | 130 B | Which models are switched on |
| `results.json` | 68 KB | Every metric, ROC and PR curve, confusion matrix and CV fold |
| `preprocess_report.txt` | 53 lines | The console log of the last training run |

`features.json` is the authoritative feature order; `Config.FEATURE_ORDER` is
only the fallback used if that file cannot be read.

---

## 7. The research folder

`research/` holds the modelling work. **The web application never imports it.**
It is kept because it is the evidence of how the models in `models/` were
produced, and it is the only code that writes to that folder.

| File | Lines | Purpose |
|---|---:|---|
| `heart.csv` | — | The dataset, 70,000 records, 3.1 MB |
| `train_models.py` | 749 | The 13-step preprocessing and training pipeline |
| `__init__.py` | 13 | Makes it a package and explains what the folder is |

`train_models.py` takes all its paths from `Config`, so it writes to the same
`models/` directory the application reads from, wherever the project is copied.

**Run it with:** `python research/train_models.py` (roughly one to two minutes).
Because every model uses `random_state=42`, rerunning it reproduces the same
`.pkl` files byte for byte.

---

## 8. Tests and verification

### `test_project.py` — 639 lines, 150 checks
Plain `assert`-style checks with no testing framework, so it runs anywhere Python
runs. Eight sections: feature engineering, password security, model loading,
end-to-end prediction, input validation, the database layer, Flask routing with
access control, and the single-administrator role model.

**Run it with:** `python test_project.py`

### `verify_baseline.py` — 204 lines
The regression guard. Five fixed patients spanning the whole risk range are run
through the real pipeline, and every number is compared against
`baseline_predictions.json`:

- the engineered features (BMI, pulse pressure, age group, high-risk flag),
- the complete 15-value feature vector,
- every individual model's probability and label,
- the ensemble result and the risk band,
- the validation outcome.

```bash
python verify_baseline.py --save   # record the current behaviour
python verify_baseline.py          # compare against the recording
```

Probabilities are compared to ten decimal places. Run it after touching anything
in `ml_engine.py`, `config.py` or `models/`.

---

## 9. Data and generated files

| File | Created by | Safe to delete? |
|---|---|---|
| `heart_disease.db` | `auth_db.init_db()` on first run | Yes — it is rebuilt empty, but you lose the demo data |
| `baseline_predictions.json` | `verify_baseline.py --save` | Yes, but regenerate it before refactoring |
| `models/*.pkl` | `research/train_models.py` | **No** — the app cannot predict without them |
| `static/img/shap_summary.png` | `research/train_models.py` | Yes — the page hides the panel if it is missing |
| `__pycache__/` | Python | Yes, always |

---

## 10. Dependency map

Who imports whom. Arrows point from importer to imported.

```
run.py
  └── routes/__init__.py            (create_app)
        ├── config.py
        ├── auth_db.py  ────────────► sqlite3, hashlib, secrets
        ├── ml_engine.py ───────────► pickle, json, pandas, config.py
        ├── routes/security.py ─────► auth_db.py
        ├── routes/auth_routes.py ──► auth_db.py, security.py
        ├── routes/patient_routes.py► auth_db.py, ml_engine.py, security.py
        ├── routes/doctor_routes.py ► auth_db.py, ml_engine.py, security.py
        └── routes/admin/ ──────────► auth_db.py, ml_engine.py, security.py

research/train_models.py ───────────► config.py, pandas, sklearn, xgboost, shap
   (standalone - nothing in the app imports it)

test_project.py ────────────────────► everything
verify_baseline.py ─────────────────► ml_engine.py, config.py
seed_demo_data.py ──────────────────► auth_db.py, ml_engine.py
```

Two properties are worth noticing:

1. **There are no import cycles.** `ml_engine.py` and `auth_db.py` know nothing
   about Flask or about the routes, which is why both can be tested with no
   application context.
2. **`config.py` is imported by everything and imports nothing** except the
   standard library. That is what lets it be the single source of truth.
