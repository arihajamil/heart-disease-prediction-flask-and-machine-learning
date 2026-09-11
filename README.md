# Heart Disease Prediction — Cardiovascular Risk Prediction Portal

A Flask web application that estimates cardiovascular disease risk from eleven
routine clinical measurements, using five machine-learning models trained on
64,825 patient records.

Final Year Project — 2026.

> **Educational tool, not a medical diagnosis.**
> Heart Disease Prediction estimates risk from statistical patterns in population data.
> It has not examined the patient and it can be wrong in both directions.
> Always consult a qualified doctor before acting on any result.

---

## Contents

1. [Features](#features)
2. [Technology](#technology)
3. [Setup](#setup)
4. [Default accounts](#default-accounts)
5. [Route map](#route-map)
6. [Project structure](#project-structure)
7. [How it works](#how-it-works)
8. [Model performance](#model-performance)
9. [Security](#security)
10. [Testing](#testing)
11. [Screenshots](#screenshots)
12. [Dataset citation](#dataset-citation)

### Further documentation

| Document | Read it when you want to know |
|---|---|
| [docs/file_guide.md](docs/file_guide.md) | What every single file in the project is for |
| [docs/architecture.md](docs/architecture.md) | How a request travels through the app, and why it is laid out this way |
| [docs/ml_pipeline.md](docs/ml_pipeline.md) | The full data and modelling pipeline, step by step |
| [docs/viva_qa.md](docs/viva_qa.md) | 30 examiner questions with answers grounded in this code |

---

## Features

### Patients
- **Risk assessment form** with live BMI and WHO category as height and weight are typed
- **Client and server validation** — impossible readings (systolic below diastolic,
  implausible BMI) are blocked with a clear explanation
- **Risk gauge** — Low / Moderate / High banding with a marker showing where the
  patient sits on the scale
- **Model agreement** — "4 of 5 models predict At Risk", so a split verdict is visible
- **Lifestyle guidance** matched to the risk band
- **Downloadable report** (`.txt`) with inputs, result, model breakdown and disclaimer
- **Personal history** with a risk-over-time chart, filters and per-record reports

### Doctors
- **Patient list** ordered by highest recorded risk, with assessment counts
- **Per-patient view** — full assessment history, trend chart and clinical notes
- **Clinical reports** — risk distribution, assessments over time, model usage,
  with a date-range filter

### Administrators
- **User management** — search, filter, change role, suspend, reinstate, delete
- **Doctor accounts** — create clinician logins (doctors cannot self-register)
- **Prediction management** — filter by verdict, model or name; delete individually
  or clear everything behind a typed confirmation
- **Dataset overview** — row count, class balance, per-column statistics, preview
- **Analytics** — assessments per day, risk split, risk bands, users by role, model usage
- **Model performance** — full metric table for the five classifiers **and the
  ensemble the app uses** (both 0.50 and the 0.40 screening cut-off), calibration curves with Brier scores, cross-validation stability,
  confusion matrices, feature importance and the pre-generated SHAP summary
- **Model training** — retrain all five models on a background thread with live status
- **Activity log** — paginated audit trail with filtering

---

## Technology

| Layer | Choice | Why |
|---|---|---|
| Web framework | Flask 3.1 (app factory + blueprints) | Small, explicit, easy to explain |
| Templating | Jinja2 | Ships with Flask |
| Database | SQLite | Serverless, one file, easy to back up |
| ML | scikit-learn 1.9, XGBoost 3.4 | Five classifiers, saved as `.pkl` |
| Charts | Chart.js 4.4 (self-hosted) | Interactive, no server-side image rendering |
| Styling | Hand-written CSS | No Bootstrap — full control of the medical theme |
| JavaScript | Vanilla | No build step, nothing to install |

There is **no Streamlit anywhere** in this project.

### Works fully offline

Every asset is served from this project &mdash; Chart.js and the Inter font are
bundled in `static/`, not loaded from a CDN. Once the Python packages are
installed the application needs **no internet connection at all**, so it runs
on an air-gapped or locked-down machine without blank charts or fallback fonts.

---

## Setup

**Requirements:** Python 3.10 or newer (developed on 3.14).

```bash
# 1. Go to the project folder
cd "Heart Disease Prediction FYP"

# 2. Create a virtual environment (recommended)
python -m venv .venv

#    Windows
.venv\Scripts\activate
#    macOS / Linux
source .venv/bin/activate

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Start the application
python run.py
```

Open <http://127.0.0.1:5000>. The database and the demo accounts are created
automatically on first run &mdash; there is nothing else to configure.

Step 3 is the only requirement beyond extracting the folder. After that the
project is entirely self-contained.

### Optional environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HDP_SECRET_KEY` | insecure dev fallback | **Set this in production** — signs the session cookie |
| `HDP_ENV` | `development` | Set to `production` to disable debug mode |
| `HDP_PORT` | `5000` | Port to listen on |
| `HDP_HTTPS` | `0` | Set to `1` behind HTTPS to mark the cookie Secure |

```bash
# Windows
set HDP_SECRET_KEY=some-long-random-string
set HDP_ENV=production
python run.py
```

### Retraining the models

Either use the interface (**Admin → Model Training**), or run the pipeline directly:

```bash
python research/train_models.py
```

This rewrites everything in `models/` and regenerates `static/img/shap_summary.png`.
It takes roughly one to two minutes on 65,000 rows.

Every model is created with `random_state=42`, so retraining reproduces the same
`.pkl` files byte for byte. Nothing about the application changes when you run it.

---

## Default accounts

| Username | Password | Role | Created by |
|---|---|---|---|
| `doctor` | `doctor123` | Doctor | automatically, on first run |
| `admin` | `admin123` | Admin | automatically, on first run |
| `patient` | `patient123` | Patient | `python seed_demo_data.py` |

The doctor and administrator accounts are seeded automatically when the database
is empty. Running `python seed_demo_data.py` (recommended) creates the demo
patient account and fifteen sample assessments, so every chart has something to
show and all three roles can be demonstrated.

> ⚠ **These credentials are demo-only and are printed in this public README.**
> Anyone who can reach the application can sign in with them. Before exposing
> the system beyond your own machine: sign in as `admin`, change both
> passwords, and set a real `HDP_SECRET_KEY`. Do not skip this.

The public Register page creates **Patient accounts only** — there is no role
choice on the form, and the server hard-codes the role, so even a hand-crafted
POST carrying `role=Doctor` or `role=Admin` still produces a Patient account.
This is deliberate: a Doctor account can read every patient's assessment
history, so that role must never be self-service. **Doctor accounts are created
only by an administrator** (Admin → Doctors), and **Administrator cannot be
self-assigned**; only an existing administrator can promote someone.

### Roles

There are **three roles**, and **Administrator is the single highest tier** —
it holds every privilege in the system.

| Role | Can do |
|---|---|
| **Patient** | Run assessments, view own history and reports, edit own profile |
| **Doctor** | Everything above, plus view all patients and clinical reports |
| **Admin** | Everything above, plus manage every user and role, create doctors, manage datasets, view analytics, run model training and purge the activity log |

**Last-admin protection:** the final active administrator cannot be suspended,
demoted or deleted, and no administrator can act on their own account. Without
this the system could be left with nobody able to administer it.

---

## Route map

### Authentication
| Method | Route | Purpose |
|---|---|---|
| GET, POST | `/login` | Sign in |
| GET, POST | `/register` | Patient self-registration |
| GET | `/logout` | Sign out |

### Patient — sign-in required
| Method | Route | Purpose |
|---|---|---|
| GET | `/dashboard` | Role-aware dashboard with KPIs and trend |
| GET | `/diagnosis` | Risk assessment form |
| POST | `/predict` | Validate, engineer features, predict, save, show result |
| GET | `/history` | Own assessment history with filters |
| GET | `/report/<id>` | Download a report (ownership checked) |
| GET, POST | `/profile` | View and update account details |
| GET | `/help` | Clinical reference and FAQ |

### Doctor — doctor or admin required
| Method | Route | Purpose |
|---|---|---|
| GET | `/doctor/patients` | All patients, ranked by highest risk |
| GET | `/doctor/patients/<id>` | One patient's full history |
| GET | `/doctor/reports` | Aggregate clinical reporting |

### Admin — administrator role required
| Method | Route | Purpose |
|---|---|---|
| GET | `/admin/users` | User list with filters |
| POST | `/admin/users/<id>/role` | Change a user's role |
| POST | `/admin/users/<id>/ban` | Suspend or reinstate |
| POST | `/admin/users/<id>/delete` | Delete an account |
| GET, POST | `/admin/doctors` | List and create doctor accounts |
| GET | `/admin/predictions` | All predictions with filters |
| POST | `/admin/predictions/delete/<id>` | Delete one record |
| POST | `/admin/predictions/clear` | Clear all (typed confirmation) |
| GET | `/admin/dataset` | Dataset overview and preview |
| GET | `/admin/analytics` | System-wide charts |
| GET | `/admin/model-performance` | Metrics, CV, importances, SHAP |
| GET | `/admin/training` | Training page and history |
| POST | `/admin/training/run` | Start training on a background thread |
| GET | `/admin/training/status` | JSON status, polled by the page |
| GET | `/admin/audits` | Paginated activity log |
| POST | `/admin/audits/clear` | Purge the activity log |

---

## Project structure

```
Heart Disease Prediction FYP/
├── run.py                    Entry point - creates the app and runs the dev server
├── config.py                 Every tunable value: paths, risk bands, feature order,
│                             validation ranges. Single source of truth.
├── ml_engine.py              Model loading, feature engineering, validation, predict()
├── auth_db.py                SQLite data layer, PBKDF2 auth, audit logging
├── test_project.py           150-check test suite (run: python test_project.py)
├── verify_baseline.py        Regression guard - proves refactoring changed no prediction
├── seed_demo_data.py         Creates the demo patient and 15 sample assessments
│
├── routes/
│   ├── __init__.py           App factory, blueprint registration, error pages
│   ├── security.py           Role constants, @login_required, @role_required, CSRF
│   ├── auth_routes.py        Login, register, logout
│   ├── patient_routes.py     Dashboard, diagnosis, predict, history, report, profile, help
│   ├── doctor_routes.py      Patient list, patient detail, clinical reports
│   └── admin/                One blueprint, split by subject
│       ├── __init__.py       admin_bp, shared training state, last-admin guard
│       ├── users.py          Accounts, roles, suspension, doctor creation
│       ├── predictions.py    Oversight of every saved assessment
│       ├── data.py           Dataset inspection and system analytics
│       ├── models.py         Model performance and background retraining
│       └── audits.py         Activity log
│
├── templates/
│   ├── base.html             Layout: sidebar, flash messages, footer disclaimer
│   ├── auth/                 login.html, register.html
│   ├── patient/              diagnosis.html, result.html, history.html,
│   │                         profile.html, help.html
│   ├── doctor/               patients.html, patient_detail.html, reports.html
│   ├── admin/                users.html, doctors.html, predictions.html,
│   │                         dataset.html, analytics.html,
│   │                         model_performance.html, training.html, audits.html
│   ├── shared/               dashboard.html
│   └── errors/               400.html, 403.html, 404.html, 500.html
│
├── static/
│   ├── css/style.css         Full custom theme, responsive
│   ├── js/main.js            Sidebar toggle, flash auto-dismiss, confirmations
│   ├── js/diagnosis.js       Live BMI, range checks, blood-pressure validation
│   ├── js/charts.js          Chart wrappers for every chart
│   ├── js/chart.min.js       Chart.js 4.4.1, bundled so no internet is needed
│   ├── css/fonts.css         Self-hosted Inter font faces
│   ├── fonts/                inter-0.woff2, inter-1.woff2
│   └── img/                  heart_bg.png, shap_summary.png (written at training time)
│
├── models/                   scaler.pkl, five model .pkl files, features.json,
│                             config.json, results.json, preprocess_report.txt
│
├── research/                 The modelling work behind the application.
│   ├── heart.csv             Dataset, 70,000 records
│   └── train_models.py       13-step preprocessing and training pipeline
│
├── docs/                     file_guide.md, architecture.md,
│                             ml_pipeline.md, viva_qa.md
│
├── heart_disease.db             SQLite database (created on first run)
├── requirements.txt
└── README.md
```

`research/` is kept separate on purpose. The web application never imports it —
it is the evidence of how the models were produced, and it is the only place that
writes to `models/`.

---

## How it works

```
  Patient measurements (11 inputs)
              │
              ▼
  Server-side validation   ml_engine.validate_form()
              │            ranges, systolic > diastolic, plausible BMI
              ▼
  Feature engineering      ml_engine.engineer_features()
              │            adds BMI, pulse pressure, age group, high-risk flag
              ▼
  StandardScaler           the exact scaler fitted during training
              │
              ▼
  Five models              Logistic Regression, SVM, Decision Tree,
              │            Random Forest, XGBoost
              ▼
  Result                   probability, risk band, model agreement, report
              │
              ▼
  SQLite                   saved and attributed to the signed-in user
```

### The 15 features

Eleven come from the form; four are calculated in `ml_engine.py`:

| Engineered feature | Formula |
|---|---|
| `bmi` | weight ÷ (height in metres)², clipped to 10–70 |
| `pulse_pressure` | systolic − diastolic |
| `age_group` | five bands: 0–30, 31–45, 46–60, 61–75, 76+ |
| `high_risk_flag` | 1 when cholesterol ≥ 2 **and** systolic ≥ 140 |

These formulas exist in exactly one place and are covered by a test that compares
them against the training pipeline's vectorised version, so the two cannot drift apart.

### The decision threshold — 0.40, on purpose

Each model outputs a probability; a probability is not yet a decision. Heart Disease Prediction
calls a patient "at risk" at or above **0.40** (`Config.POSITIVE_THRESHOLD`) rather
than the textbook 0.50, because it is a **screening** tool: a false positive costs
one unnecessary check-up, while a false negative sends a sick person home
reassured. At 0.50 the models miss roughly 31% of true cases (recall ≈ 0.69);
lowering the cut-off trades some precision for substantially higher recall, which
is the correct trade in this context. The same threshold is applied uniformly to
every individual model's vote and to the ensemble, and it equals the Low/Moderate
display-band boundary — so the yes/no verdict and the coloured badge can never
contradict each other. `results.json` reports metrics at **both** cut-offs: 0.50
for comparability with the literature, and 0.40 for how the app actually behaves.

### Preprocessing

`research/train_models.py` runs thirteen documented steps: load → housekeeping →
deduplicate → physiological filtering → IQR winsorization → median imputation →
feature engineering → correlation pruning → 80/20 stratified split →
standardisation → train five classifiers → 5-fold cross-validation → SHAP image.

The physiological filters remove genuine data-entry errors, such as diastolic
readings of 1000 mmHg and systolic readings of 12 mmHg. They discard about 2% of
the data, leaving **64,825 usable records** with a class balance of 1.04 : 1.

---

## Model performance

Measured on a held-out test set of **12,965 patients** never seen during training.

| Model | Accuracy | ROC AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 72.8% | 0.790 | 0.720 | 0.756 | 0.687 |
| Support Vector Machine | 72.9% | 0.790 | 0.724 | 0.750 | 0.700 |
| Decision Tree | 72.9% | 0.792 | 0.719 | 0.762 | 0.680 |
| Random Forest | 73.4% | **0.798** | 0.726 | **0.763** | 0.692 |
| XGBoost | **73.5%** | **0.798** | **0.728** | 0.755 | 0.701 |
| *Ensemble (mean of the five)* | 73.3% | 0.799 | 0.727 | 0.760 | 0.696 |

The ensemble row is the verdict the application actually shows: `ml_engine.predict()`
averages the five probabilities and applies the screening threshold to the mean.
It is scored by `research/train_models.py` on the same held-out set and stored in
`results.json` under the key `Ensemble Voting`. At the app's **0.40** cut-off the
ensemble reaches **recall 0.791 / precision 0.702** (individual models: 0.76–0.81).
Its job is not to beat the best single model - the five share the same 15 features
and mostly agree - but to avoid depending on any one of them and to make disagreement
visible ("4 of 5 models predict At Risk").

Random Forest and XGBoost are effectively tied — the gap is far smaller than the
±0.003 cross-validation spread, so neither can be called the winner on this evidence.

Five-fold cross-validation, with the scaler refitted inside every fold so no
information leaks between them:

| Model | CV ROC AUC |
|---|---|
| Logistic Regression | 0.7886 ± 0.0028 |
| Support Vector Machine | 0.7883 ± 0.0028 |
| Decision Tree | 0.7886 ± 0.0047 |
| Random Forest | 0.7959 ± 0.0034 |
| XGBoost | 0.7950 ± 0.0035 |

The very small standard deviations show the models are stable and are not simply
memorising one particular split of the data.

The table above uses the standard 0.50 cut-off so the figures stay comparable
with published results on this dataset. The application itself decides "at risk"
at the **0.40 screening threshold** (see *How it works*), and `results.json`
additionally records each model's recall, precision and confusion matrix at that
cut-off, plus a **Brier score** and a 10-bin **calibration (reliability) curve**
per model — shown on the Model Performance page — so the probabilities the app
displays to users are demonstrably trustworthy, not just well-ranked.

---

## Security

| Measure | Implementation |
|---|---|
| Password hashing | PBKDF2-HMAC-SHA256, 16-byte random salt per user, 260,000 iterations |
| Password comparison | `secrets.compare_digest` — constant time |
| Brute-force defence | Login rate limiting: 5 failures per username **or** IP within 15 min → 10 min lockout; lockouts audit-logged |
| Legacy migration | Old SHA-256 hashes still verify and upgrade silently on next sign-in |
| Session cookie | Signed with `SECRET_KEY`; HttpOnly, SameSite=Lax, Secure when `HDP_HTTPS=1` |
| Session fixation | Session cleared and reissued on every sign-in |
| CSRF | Random per-session token in a hidden field, checked on every POST |
| SQL injection | Every query parameterised — no string interpolation anywhere |
| Access control | `@login_required` and `@role_required` on every protected route; wrong role gets a 403 page |
| Privilege escalation | Public registration is hard-coded to Patient (the posted role is never read); Doctor accounts are admin-created only; the last administrator is protected from removal |
| Error disclosure | Real errors logged server-side; users see friendly 400/403/404/500 pages |
| Report access | Ownership verified before a report is sent |

---

## Testing

```bash
python test_project.py
```

Runs **around 150 checks** with plain assert-style statements — no testing
framework needed — covering feature engineering, password security, model
loading, end-to-end prediction, input validation, the screening threshold and
its alignment with the display bands, login rate limiting, the database, Flask
routing with access control, and the single-administrator role model. The exact
pass/fail count is printed at the end of every run.

### Regression guard

```bash
python verify_baseline.py
```

Runs five fixed patients through the real pipeline and compares every number
against `baseline_predictions.json`: the engineered features, the full 15-value
vector, each individual model's vote, the ensemble probability and the risk band.

This is what makes refactoring safe. Any change that alters a prediction — even
in the tenth decimal place — fails loudly instead of passing unnoticed. Use it
after touching anything in `ml_engine.py`, `config.py` or `models/`.

---

## Screenshots

Replace each placeholder with a real screenshot before submitting.

| Screen | Image |
|---|---|
| Login page | `![Login](screenshots/01-login.png)` |
| Dashboard | `![Dashboard](screenshots/02-dashboard.png)` |
| Assessment form with live BMI | `![Diagnosis](screenshots/03-diagnosis.png)` |
| Result with risk gauge | `![Result](screenshots/04-result.png)` |
| History with trend chart | `![History](screenshots/05-history.png)` |
| Doctor patient list | `![Patients](screenshots/06-patients.png)` |
| Admin analytics | `![Analytics](screenshots/07-analytics.png)` |
| Model performance | `![Models](screenshots/08-models.png)` |

---

## Dataset citation

**Cardiovascular Disease dataset**
Svetlana Ulianova, published on Kaggle.
<https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>

70,000 patient records collected during medical examinations, holding objective
measurements (age, height, weight, blood pressure), examination results
(cholesterol, glucose) and self-reported lifestyle information (smoking, alcohol,
physical activity), together with a binary label for the presence of
cardiovascular disease.

Note that in this copy, cholesterol and glucose are stored zero-indexed (0, 1, 2).
`research/train_models.py` shifts them to the standard clinical 1–3 scale during
preprocessing. This detail matters more than it looks: filtering on the 1–3 values
without shifting first threw away about 91% of the dataset. See
[docs/ml_pipeline.md](docs/ml_pipeline.md).

---

## Acknowledgements

Built with [Flask](https://flask.palletsprojects.com),
[scikit-learn](https://scikit-learn.org), [XGBoost](https://xgboost.ai),
[SHAP](https://shap.readthedocs.io) and [Chart.js](https://www.chartjs.org).
