# Architecture

How Heart Disease Prediction is put together, and why.

[file_guide.md](file_guide.md) answers *where is it*. This document answers
*why is it there*.

---

## Contents

1. [The shape of the system](#1-the-shape-of-the-system)
2. [Startup: what happens before the first request](#2-startup-what-happens-before-the-first-request)
3. [The life of a request](#3-the-life-of-a-request)
4. [Walkthrough: one risk assessment](#4-walkthrough-one-risk-assessment)
5. [The role model](#5-the-role-model)
6. [Security layers](#6-security-layers)
7. [The database](#7-the-database)
8. [Design decisions and their trade-offs](#8-design-decisions-and-their-trade-offs)
9. [Extending the system](#9-extending-the-system)

---

## 1. The shape of the system

Four layers, each depending only on the one below it.

```
┌────────────────────────────────────────────────────────────┐
│  PRESENTATION      templates/  static/                      │
│                    Jinja2 pages, hand-written CSS,          │
│                    vanilla JS, Chart.js                     │
├────────────────────────────────────────────────────────────┤
│  ROUTING           routes/                                  │
│                    Four blueprints + the security layer.    │
│                    Handles HTTP, sessions, access control,  │
│                    flash messages. Holds no business rules. │
├────────────────────────────────────────────────────────────┤
│  DOMAIN            ml_engine.py        auth_db.py           │
│                    Prediction, feature   Persistence,       │
│                    engineering,          hashing, audit     │
│                    validation            logging            │
├────────────────────────────────────────────────────────────┤
│  DATA              models/*.pkl    heart_disease.db            │
│                    Trained artefacts     SQLite file        │
└────────────────────────────────────────────────────────────┘

                        config.py
       imported by every layer, imports nothing but stdlib
```

The important property is that **the domain layer knows nothing about Flask**.
`ml_engine.py` and `auth_db.py` contain no `request`, no `session`, no
`render_template`. That is why:

- `verify_baseline.py` can run the full prediction pipeline with no web server,
- `test_project.py` can test feature engineering without an application context,
- `seed_demo_data.py` can create real predictions from the command line.

If the domain layer imported Flask, none of those would be possible without
faking a request.

---

## 2. Startup: what happens before the first request

`run.py` calls `create_app()` in `routes/__init__.py`, which does seven things in
order:

```python
app = Flask(__name__)
app.config.from_object(get_config())   # 1. DevConfig or ProdConfig

auth_db.init_db()                      # 2. create tables, run migrations, seed
ml_engine.init_engine(MODELS_DIR)      # 3. unpickle the scaler and 5 models

app.register_blueprint(auth_bp)        # 4. four blueprints
app.register_blueprint(patient_bp)
app.register_blueprint(doctor_bp)
app.register_blueprint(admin_bp)

app.before_request(_check_csrf)        # 5. CSRF on every POST
@app.context_processor ...             # 6. csrf token available in templates
@app.errorhandler(400/403/404/500)     # 7. friendly error pages
```

Steps 2 and 3 happen **once**, at startup, not per request.

For the models that is a deliberate performance decision: `random_forest.pkl`
alone is 30 MB, and unpickling it takes noticeably longer than a page should. By
loading into module-level variables (`_SCALER`, `_MODELS`) once, a prediction
costs no disk access at all — it is a matrix multiply and five `predict_proba`
calls, measured at under 75 ms for all five models combined.

`init_db()` is safe to call repeatedly: every statement is `CREATE TABLE IF NOT
EXISTS`, and the seed accounts are only inserted when the users table is empty.

### Resilient model loading

Each model is loaded inside its own `try`:

```python
for name, filename in MODEL_FILES.items():
    try:
        _MODELS[name] = pickle.load(open(path, "rb"))
    except ModuleNotFoundError as e:
        _LOAD_WARNINGS.append(f"{name}: needs the '{e.name}' library. Skipped.")
    except Exception as e:
        _LOAD_WARNINGS.append(f"{name}: could not be loaded. Skipped.")
```

This replaced an all-or-nothing version where a single unloadable pickle returned
`None, {}` and disabled prediction entirely. A machine without the `xgboost`
package now runs on the other four models and says so on the page, instead of
showing "no models available".

---

## 3. The life of a request

```
Browser
  │  GET /history
  ▼
Flask routing ──► url_map picks patient.history
  │
  ▼
@login_required        no session? → flash + redirect to /login?next=/history
  │
  ▼
@role_required(...)    wrong role? → abort(403) → templates/errors/403.html
  │
  ▼
before_request         POST only: compare the form's csrf_token with the
  │                    session's, using secrets.compare_digest
  ▼
View function          calls auth_db / ml_engine, never writes SQL itself
  │
  ▼
render_template        base.html + the page's own blocks
  │
  ▼
Browser                HTML, plus Chart.js drawing any charts client-side
```

Every unexpected exception is caught by the 500 handler, which logs the real
traceback server-side and shows the user a plain page. The user never sees a
stack trace, a file path or an SQL fragment.

> One trap worth recording: `abort(404)` raises Flask's `NotFound`, which is an
> `Exception`. An over-broad `except Exception` around a view will swallow it and
> turn it into a 500. In this codebase `abort()` is always called *outside* the
> `try`.

---

## 4. Walkthrough: one risk assessment

This is the path an examiner is most likely to ask you to trace. It is
`patient.predict()` in [routes/patient_routes.py](../routes/patient_routes.py).

### Step 1 — the form arrives

`POST /predict` with eleven measurements plus `patient_name`, `patient_id`,
`notes` and the CSRF token.

### Step 2 — CSRF, before the view even runs

`_check_csrf` compares the submitted token with the session's. A form posted from
another site cannot read the session cookie, so it cannot supply the right token.

### Step 3 — server-side validation

```python
is_valid, errors = ml_engine.validate_form(request.form)
```

`validate_form()` checks:

| Check | Source of the rule |
|---|---|
| Every required field present and numeric | — |
| Age, height, weight, both blood pressures within range | `Config.FIELD_RANGES` |
| Dropdown codes are among the allowed values | `Config.CHOICE_FIELDS` |
| Systolic strictly greater than diastolic | Physiology |
| Pulse pressure between 20 and 100 mmHg | `Config.PULSE_PRESSURE_MIN/MAX` |
| BMI between 12 and 60 | `Config.BMI_PLAUSIBLE_MIN/MAX` |
| Patient name length, patient id matches `PT-\d{3,12}`, notes length | `Config` |

`static/js/diagnosis.js` applies the same rules in the browser, but that is a
convenience for the user, not a control. Anyone can disable JavaScript or post
directly with `curl`, so the server repeats every check. **The browser check is
for helpfulness; the server check is for correctness.**

### Step 4 — feature engineering

```python
vector, readable = ml_engine.engineer_features(request.form)
```

Eleven raw values in, fifteen out. The four derived features are:

| Feature | Formula |
|---|---|
| `bmi` | `weight / (height/100)²`, then clipped to 10–70 |
| `pulse_pressure` | `ap_hi - ap_lo` |
| `age_group` | 0–4, from bins `[0, 30, 45, 60, 75, 120]` |
| `high_risk_flag` | `1` when `cholesterol >= 2 and ap_hi >= 140` |

These formulas are duplicated in `research/train_models.py` — deliberately, since
the app must not import the training code — and a test compares the two
implementations so they cannot drift apart.

### Step 5 — scaling

```python
frame  = pd.DataFrame([vector], columns=_FEATURE_NAMES)
scaled = _SCALER.transform(frame)
```

`_SCALER` is the exact `StandardScaler` object that was fitted during training
and pickled. It is **reused, never refitted**.

That matters for two reasons. Refitting on a single patient is meaningless — one
row has no spread, so every value would become zero. And even refitting on many
rows would measure the patient against the wrong population; the models learned
what "high" means from the training distribution, so a patient must be measured
against that same distribution.

The vector is wrapped in a named DataFrame because the scaler was fitted on one.
Passing a bare list still works, but scikit-learn emits a feature-names warning
on every single prediction, which floods the log.

### Step 6 — the models vote

```python
for name in enabled:
    probability = float(model.predict_proba(scaled)[0][1])
    votes[name] = {
        "probability": probability,
        "predicted":   1 if probability >= Config.POSITIVE_THRESHOLD else 0,
    }
```

`predict_proba` returns `[P(no disease), P(disease)]`; index `[1]` is kept.

The ensemble probability is the plain average of the five. A patient is called
"at risk" at or above `Config.POSITIVE_THRESHOLD` (**0.40** — a screening
cut-off, deliberately below the textbook 0.50 because a missed case costs more
than a false alarm; the full rationale sits with the number in `config.py`).
Note that each model's vote applies the same threshold to its probability,
rather than calling the model's own `predict()` — scikit-learn hard-wires that
to 0.50, which would leave the votes and the ensemble using different rules.

Individual votes are kept, not discarded, because agreement is information the
user should see. "5 of 5 models predict At Risk" is a materially different
message from "3 of 5", even when the averaged probability is the same.

### Step 7 — banding for display

```python
band = ml_engine.get_risk_level(probability)
```

| Probability | Band | Colour | Emoji |
|---|---|---|---|
| below 0.40 | LOW RISK | green | 😊 |
| 0.40 – 0.70 | MODERATE RISK | amber | 😐 |
| 0.70 and above | HIGH RISK | red | 😟 |

**The Low/Moderate boundary equals the decision threshold, by design.** The
verdict switches to "at risk" at 0.40 and the badge leaves LOW at the same
0.40, so the two can never contradict each other on screen — every "at risk"
patient sees Moderate or High, every "not at risk" patient sees Low. (An
earlier version decided at 0.50 with bands at 0.40/0.70, so a patient at 0.45
was "not at risk" wearing a Moderate badge — a genuine inconsistency, since
fixed at the source.) 0.70 remains purely presentational: it only separates
amber from red.

### Step 8 — save, then render

`auth_db.add_prediction()` writes the row, attributed to the signed-in user.
If that write fails the failure is flashed, but the result is still rendered —
the computation succeeded, and throwing it away would help nobody.

---

## 5. The role model

Three roles, one rising ladder of privilege. **Administrator is the single
highest tier** — there is no SuperAdmin (an earlier version had one; it was
merged, and a migration folds any surviving rows into `Admin`).

| | Patient | Doctor | Admin |
|---|:---:|:---:|:---:|
| Run an assessment | ✅ | ✅ | ✅ |
| See own history and reports | ✅ | ✅ | ✅ |
| Edit own profile | ✅ | ✅ | ✅ |
| See **all** patients and reports | — | ✅ | ✅ |
| Manage users, roles, suspension | — | — | ✅ |
| Create doctor accounts | — | — | ✅ |
| Dataset, analytics, model performance | — | — | ✅ |
| Run retraining | — | — | ✅ |
| Read and purge the activity log | — | — | ✅ |

### How scoping is actually implemented

One helper, used everywhere:

```python
def _scope_user_id(user):
    return user["id"] if user["role"] == ROLE_PATIENT else None
```

`None` means "no filter". So `auth_db.get_predictions(user_id=...)` serves a
patient looking at their own history and a doctor looking at everyone's, with no
branching in the view and no second query to keep in sync. One rule, one place.

### Last-admin protection

`_can_manage()` in `routes/admin/__init__.py` enforces two invariants on every
state-changing admin action:

1. **Nobody may act on their own account** — you cannot demote, suspend or delete
   yourself.
2. **The last active administrator is protected** — they cannot be demoted,
   suspended or deleted.

Promoting *to* administrator is always allowed; only removal is guarded. Without
rule 2 a single careless click could leave the system with nobody able to
administer it, and no way back in short of editing the database by hand.

---

## 6. Security layers

| Threat | Defence | Where |
|---|---|---|
| Password database theft | PBKDF2-HMAC-SHA256, 16-byte random salt per user, 260,000 iterations | `auth_db.hash_password` |
| Timing attacks on login | `secrets.compare_digest` | `auth_db.verify_password` |
| Online brute force | Rate limiting: 5 failures per username or IP in 15 min locks sign-in for 10 min; a success clears the counters; lockouts are audit-logged | `routes/security.py`, `auth_routes.login` |
| Legacy weak hashes | Old SHA-256 hashes verify and silently upgrade on next sign-in | `auth_db.verify_password` |
| Session hijacking | `HttpOnly`, `SameSite=Lax`, `Secure` when `HDP_HTTPS=1` | `config.py` |
| Session fixation | `session.clear()` before storing the user, so a new id is issued | `auth_routes.login` |
| Cross-site request forgery | Per-session token in a hidden field, checked before every POST | `routes/security.py` |
| SQL injection | Every query parameterised — no string interpolation anywhere | `auth_db.py` |
| Broken access control | `@login_required` and `@role_required` on every protected route | `routes/security.py` |
| Privilege escalation | Public registration is hard-coded to Patient — the posted role is never read, so neither Doctor (which can read every patient's records) nor Administrator can be self-assigned | `auth_routes.register` |
| Open redirect via `?next=` | Only local paths are honoured: must start with `/`, must not start with `//`, and may not contain a backslash (browsers normalise `\` to `/`, so `/\evil.com` would become protocol-relative) | `auth_routes.login` |
| Insecure direct object reference | Ownership checked before a report is sent | `patient_routes.report` |
| Information disclosure | Real errors logged server-side; users see plain error pages | `routes/__init__.py` |
| Denial of service by upload | `MAX_CONTENT_LENGTH` caps any upload at 32 MB | `config.py` |

### Why the CSRF protection is hand-rolled

The obvious choice is Flask-WTF. It was not used because the project's brief was
to avoid dependencies that are not strictly needed, and the mechanism is
thirty lines:

```python
def generate_csrf_token():
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(32)
    return session["_csrf_token"]
```

The token is exposed to every template through a context processor, embedded in a
hidden field in every form, and compared with `compare_digest` in a
`before_request` hook that fires on every POST. The security property is the
same as Flask-WTF's: an attacker's page can make your browser send a request, but
it cannot read your session cookie, so it cannot know the token.

---

## 7. The database

SQLite, one file, four tables.

```
users                       predictions
──────────────────          ──────────────────────────────
id            PK  ◄────┐    id               PK
username      UNIQUE   └────user_id          FK → users(id) ON DELETE CASCADE
password_hash               age, gender, height, weight
role                        ap_hi, ap_lo, cholesterol, gluc
fullname                    smoke, alco, active
email                       predicted_class, probability
is_banned                   model_used
created_at                  patient_id, patient_name, notes
                            timestamp

system_logs                 training_runs
──────────────────          ──────────────────────────
id            PK            id            PK
user_id                     triggered_by
username                    status
action                      duration_s
details                     results_json
timestamp                   timestamp
```

Notes on the design:

- **`ON DELETE CASCADE`** on `predictions.user_id`, with `PRAGMA foreign_keys =
  ON` set in `_connect()`. SQLite does *not* enforce foreign keys by default, so
  without that pragma deleting a user would silently orphan their assessments.
- **`system_logs.username` is stored, not joined.** The audit trail must survive
  the deletion of the account it refers to. A log entry saying "user 47 was
  deleted" is useless once user 47 is gone.
- **The raw inputs are stored alongside the result.** That is what lets a report
  be regenerated later, and what lets anyone verify a stored probability by
  re-entering the same numbers.
- **SQLite over PostgreSQL** because the whole system must run from an extracted
  folder with no server to install. The trade-off is that SQLite serialises
  writes — fine for this workload, wrong for a real hospital deployment.

---

## 8. Design decisions and their trade-offs

Each of these is a place where something else could reasonably have been chosen.

### An application factory instead of a module-level `app`

`create_app()` means the test suite can build a fresh, independently configured
application per test run. A module-level `app = Flask(__name__)` would be shorter
but would tie configuration to import time, which makes testing awkward.

### Blueprints split by *audience*, not by *resource*

`auth`, `patient`, `doctor`, `admin`. The alternative — `users`, `predictions`,
`models` — would scatter one role's pages across several files. Since access
control in this application is role-based, grouping by role means the guard on a
file is uniform and easy to audit.

The admin blueprint grew to 714 lines in a single file, so it was split into a
package by subject while keeping one blueprint object. Endpoint names are
unchanged, so no template and no `url_for` call needed touching.

### Five models instead of one

The five span the useful range of approaches — a linear model, a margin-based
model, a single tree, bagged trees, boosted trees. That supports the comparative
analysis a Final Year Project is expected to show, and it lets the interface
report agreement, which is genuinely useful information for the user.

The cost is honest: `random_forest.pkl` is 30 MB, and running five models takes
about 75 ms rather than about 1 ms. For a per-user web form that is invisible.

### An averaged ensemble rather than the single best model

XGBoost (73.5%) and Random Forest (73.4%) are effectively tied — the gap is far
smaller than the ±0.003 cross-validation spread, so calling either "the winner"
would be overclaiming. Averaging avoids that claim and is more stable than any
single model. The user can still select one model explicitly.

### SHAP computed once, not per request

A SHAP explanation for a single prediction takes seconds on the tree ensembles.
Instead, the training pipeline writes `static/img/shap_summary.png` once, and the
Model Performance page displays it.

The honest trade-off: this is a **global** explanation — which features matter
across the whole population — not a per-patient one. A production system would
want per-prediction attributions; that was out of scope here.

### Charts drawn in the browser

Chart.js renders client-side from JSON embedded in the page. The server never
generates an image, which keeps it stateless and fast. Chart.js is vendored into
`static/js/` rather than loaded from a CDN, so the whole application works with
no internet connection.

### The database file is committed to git

Unusual, and deliberate. It carries the demo accounts and fifteen sample
assessments, so an examiner who clones the repository sees a working system
immediately instead of an empty one. For any real deployment it should be in
`.gitignore`.

---

## 9. Extending the system

### Adding a route

1. Pick the blueprint that matches the **audience**, not the data.
2. Add the view with its guard:
   ```python
   @admin_bp.route("/thing")
   @role_required("admin")
   def thing():
       return render_template("admin/thing.html", ...)
   ```
3. Put the template in the matching `templates/` folder, extending `base.html`.
4. Any form must include the CSRF field:
   ```html
   <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
   ```
5. State-changing actions should call `auth_db.log_activity(...)`.

### Adding a validation rule

Add the bound to `config.py`, enforce it in `ml_engine.validate_form()`, then
mirror it in `static/js/diagnosis.js` for instant feedback. Server first — the
browser copy is optional, the server one is not.

### Changing a feature

This is the one change that can silently break predictions.

`Config.FEATURE_ORDER` and `models/features.json` are a **positional contract**
with `scaler.pkl`. The scaler stores one mean and one standard deviation per
slot and has no idea what the columns are called. Reorder the list and every
value after the change is scaled with the wrong statistics — the app keeps
working and keeps producing confident numbers that are simply wrong.

So: change the feature list *and* retrain, together, never separately. Then run

```bash
python verify_baseline.py
```

and expect it to fail — that is the point. Inspect the differences, satisfy
yourself they are the ones you intended, and re-record with `--save`.

### Before committing any change

```bash
python verify_baseline.py   # 5 predictions unchanged?
python test_project.py      # 150 checks still passing?
```
