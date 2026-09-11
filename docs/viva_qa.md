# Viva Questions and Answers

Thirty questions an examiner is likely to ask about Heart Disease Prediction, with answers
grounded in the actual code — file names and line references included so you can
open the relevant file mid-answer.

A note on how to use this: do not memorise the wording. Understand each answer
well enough to say it in your own words, and be ready for the follow-up. Several
answers below deliberately admit a weakness, because an examiner respects a
candidate who knows the limits of their own work far more than one who claims
there are none.

---

## Contents

- [A. The project overall](#a-the-project-overall) — Q1–Q4
- [B. Data and preprocessing](#b-data-and-preprocessing) — Q5–Q11
- [C. Features and scaling](#c-features-and-scaling) — Q12–Q16
- [D. Models and evaluation](#d-models-and-evaluation) — Q17–Q22
- [E. The application](#e-the-application) — Q23–Q26
- [F. Security](#f-security) — Q27–Q29
- [G. Reflection](#g-reflection) — Q30

---

## A. The project overall

### Q1. What is Heart Disease Prediction, in one minute?

A Flask web application that estimates cardiovascular disease risk from eleven
routine clinical measurements — age, sex, height, weight, both blood pressure
readings, cholesterol, glucose, and three lifestyle factors.

Those eleven become fifteen features, four of them derived. The features are
scaled with the same `StandardScaler` that was fitted during training, then run
through five classifiers — Logistic Regression, SVM, Decision Tree, Random Forest
and XGBoost. Their probabilities are averaged, banded into Low, Moderate or High,
and saved.

Around that sits a complete portal with three roles. Patients run assessments and
see their own history. Doctors see all patients and aggregate clinical reports.
Administrators manage users, inspect the dataset, view model performance and can
retrain the models from the interface.

The models were trained on 64,825 records and reach about 73.5% accuracy with a
ROC AUC of 0.80 on a held-out test set of 12,965 patients.

### Q2. Why Flask rather than Django, FastAPI or Streamlit?

Flask is explicit. Every route, decorator and template render is code I wrote and
can explain, which matters when the work has to be defended rather than just
demonstrated.

- **Django** brings an ORM, an admin site and an auth system. Useful, but it
  would answer the interesting questions for me — the role model and access
  control here are mine, and being able to walk through them is part of the
  project.
- **FastAPI** is excellent for JSON APIs, but this is a server-rendered
  application. Its main advantages — async and automatic OpenAPI docs — do not
  apply.
- **Streamlit** was the *original* implementation. It was replaced deliberately.
  Streamlit reruns the whole script on every interaction, which makes real
  session-based authentication and per-role routing awkward. There is no
  Streamlit anywhere in the final code.

### Q3. Walk me through what happens when a user submits an assessment.

`POST /predict`, handled by `predict()` in `routes/patient_routes.py`. Eight
steps, in this order:

1. **CSRF check** — `before_request` compares the form token with the session
   token using `secrets.compare_digest`.
2. **Server-side validation** — `ml_engine.validate_form()`. Ranges, dropdown
   codes, systolic greater than diastolic, plausible pulse pressure and BMI. The
   browser checks the same things, but the server never trusts that.
3. **Feature engineering** — `engineer_features()` turns 11 values into 15.
4. **Scaling** — the pickled `StandardScaler` transforms the vector. Reused,
   never refitted.
5. **Prediction** — each enabled model returns `predict_proba(...)[0][1]`.
6. **Combination** — average the probabilities; classify "at risk" at the
   0.40 screening threshold (`Config.POSITIVE_THRESHOLD` — see Q20 for why
   it sits deliberately below the textbook 0.50).
7. **Banding** — `get_risk_level()` maps the probability to Low, Moderate or
   High.
8. **Persist and render** — save via `auth_db.add_prediction()`, then show the
   result.

If the database write fails, the failure is flashed but the result is still
rendered. The computation succeeded; discarding it would help nobody.

### Q4. What is the hardest problem you solved in this project?

A silent data-loss bug that had invalidated every model.

The preprocessing pipeline filtered cholesterol and glucose with
`isin([1, 2, 3])`. But the published dataset stores those columns 0-indexed — 0,
1, 2. So every patient with normal cholesterol, the largest single group, failed
the filter and was dropped.

**70,000 rows became 6,583. About 91% of the data, gone, in silence.** Nothing
crashed. The pipeline printed a report and produced models that scored
plausibly — around 0.72 AUC.

The fix was one line in `_basic_clean()`: shift both columns by +1 *before* any
filtering. Rows went to 64,825, class balance to 1.04 : 1, and AUC to 0.80.

What I take from it is that a silent failure is far more dangerous than a crash.
A crash tells you something is wrong. This produced a confident, plausible, wrong
model. The pipeline now prints the row count after every stage, so the next such
loss is visible immediately.

---

## B. Data and preprocessing

### Q5. Describe your dataset.

The Cardiovascular Disease dataset by Svetlana Ulianova on Kaggle: 70,000 patient
records from medical examinations, with a binary target for whether
cardiovascular disease is present.

The variables fall into three kinds, and the distinction matters:

- **Objective** — measured directly: age, height, weight, sex.
- **Examination** — from a clinical test: both blood pressures, cholesterol,
  glucose.
- **Subjective** — self-reported: smoking, alcohol, physical activity.

The third group is the least reliable, because people under-report smoking and
drinking and over-report exercise. That is a limitation of the data, not of the
model, but it caps how well any model can do.

Two quirks of the raw file: age is stored in days, and cholesterol and glucose
are 0-indexed rather than on the clinical 1–3 scale.

### Q6. How much data did you lose in preprocessing, and to what?

70,000 → 64,825. About 7.4%, and I can account for all of it:

- **Exact duplicate rows.** Removed because an identical patient appearing in
  both the training and test splits leaks the answer and inflates the score.
- **Physiologically impossible rows.** The raw file contains diastolic readings
  of 0 and 10,000 mmHg, and systolic readings of 12 and 16,020. These are not
  unusual patients — they are data-entry errors, and no model should learn from
  them.

Losing 7.4% to genuine errors is acceptable. Losing 91% — which is where this
pipeline started, before I found the encoding bug — was not.

### Q7. Why clip outliers instead of deleting those rows?

Because the two cases are different, and the pipeline treats them differently.

Step 4 **deletes** rows that are physiologically impossible: a diastolic pressure
of 10,000 describes no living person, so the record is an error.

Step 5 **clips** values that are extreme but possible, using the 1.5×IQR fences.
A genuinely obese or genuinely hypertensive patient is real. Deleting the row
would throw away nine good measurements to fix one questionable reading — and,
worse, it would systematically remove the sickest patients, who are exactly the
group the model most needs to learn from.

### Q8. Why median imputation rather than mean?

The median is robust to outliers. One erroneous blood-pressure reading of 10,000
mmHg drags a mean noticeably but leaves the median untouched. Since the data
demonstrably contains such errors, using the mean would let them contaminate
every value being filled in.

### Q9. Why did you use a stratified split?

`stratify=y` keeps the class ratio identical in the training and test halves.
Without it, a random split can produce a test set whose balance differs from the
population, and the reported score then reflects the luck of that split rather
than the model.

The same reasoning applies to `StratifiedKFold` in cross-validation.

### Q10. What is data leakage, and where could it have occurred here?

Leakage is when information that would not be available at prediction time
influences training. It makes reported scores optimistic in a way that does not
survive deployment.

Three places it could have crept into this project, and what prevents each:

1. **Scaling before splitting.** If `StandardScaler.fit()` sees the test rows,
   their statistics are baked into the transform applied to the training data.
   Prevented by splitting first, then fitting on `X_train` only.
2. **Scaling before cross-validation.** The same problem at fold scale. Prevented
   by wrapping the scaler and the model in a `Pipeline`, so scikit-learn refits
   the scaler inside every fold. The 20% test set is excluded from CV entirely.
3. **Duplicate rows.** An identical patient in both splits is a direct leak.
   Prevented by step 3.

### Q11. Why did you drop the `id` column?

It is a database key with no clinical meaning. Leaving it in gives the model an
opportunity to key off individual records rather than learn a pattern, and any
apparent signal from it is pure artefact — for example if the file happens to be
sorted by outcome.

---

## C. Features and scaling

### Q12. You engineered four features. Justify each one.

| Feature | Justification |
|---|---|
| `bmi` | A ratio involving a square. A linear model cannot represent that relationship at all from height and weight separately, and it is the standard clinical measure of body composition |
| `pulse_pressure` | Systolic minus diastolic is an established cardiovascular marker in its own right. Arterial stiffness shows up in the *gap*, not in either reading alone |
| `age_group` | Lets a model express "risk jumps after 60" as a step rather than approximating it with a smooth function |
| `high_risk_flag` | Encodes a clinical **interaction**: raised cholesterol *together with* hypertension is worse than the sum of the two separately. An additive model cannot express that unless it is told |

The honest counter-argument is that each derived feature is correlated with its
parents, which risks multicollinearity. That is precisely why step 8 checks every
pair and would drop one of any pair above |r| = 0.90. On this dataset nothing
crosses the threshold.

### Q13. Why does the order of the features matter so much?

Because `scaler.pkl` matches values **by position, not by name**. It stores one
mean and one standard deviation per slot, and has no idea the third column is
called `height`.

If I reorder `Config.FEATURE_ORDER` without retraining, every value after the
change is standardised with the wrong statistics. Nothing crashes. The
application keeps producing confident-looking numbers that are wrong.

That is a genuinely dangerous failure mode, so it is defended in three ways: the
order is declared once in `config.py` with a comment saying it is a contract; the
authoritative copy is written by training to `models/features.json`; and
`verify_baseline.py` stores the complete 15-value vector for five fixed patients,
so any reordering fails the check immediately.

### Q14. Why do you scale the features, and why does scaling not matter for trees?

The raw columns are on wildly different scales — age around 55, height around
165, smoking either 0 or 1. Logistic Regression and SVM optimise a weighted sum
or a distance, so a column with larger numbers dominates simply because its
numbers are larger, regardless of how informative it is. Standardisation rewrites
each column as "how many standard deviations from the mean".

Trees do not care: a tree splits on a threshold, and any monotonic rescaling
produces exactly the same tree. I scale them anyway so that every model in the
project consumes identical input, which removes a whole class of bug at no cost.

### Q15. At prediction time, do you fit a new scaler?

No — and this is the single most important line in the inference path. The
scaler is unpickled at startup and only ever `transform`ed with.

Fitting on one patient would be meaningless: a single row has no spread, so every
standardised value would come out as zero. And fitting on live data would measure
patients against a different population from the one the models learned on. The
models learned what "high blood pressure" means from the training distribution,
so a patient must be compared against that same distribution.

### Q16. How do you know the app's feature engineering matches the training pipeline's?

The formulas exist in two places by necessity — the application must not import
the training script — so they are kept in step two ways.

First, `test_project.py` contains a test that computes the four derived features
both ways and asserts they agree.

Second, `verify_baseline.py` stores the engineered values for five fixed patients
and re-checks them on every run. If the two implementations ever drift, the BMI,
pulse pressure, age group or high-risk flag would move and the comparison would
fail.

---

## D. Models and evaluation

### Q17. Why five models rather than one?

Three reasons.

The five span the useful range of approaches — a linear model, a margin-based
model, a single tree, bagged trees, boosted trees — which supports the
comparative analysis a project like this is expected to show.

The comparison produced a genuine finding, covered in Q18.

And it lets the interface report **agreement**, which is real information for the
user: five models out of five saying "at risk" is a stronger signal than three
out of five, even when the averaged probability is identical.

The cost is honest: `random_forest.pkl` is 30 MB, and running five models takes
about 75 ms instead of about 1 ms. For a per-user web form that is invisible.

### Q18. Which model is best?

That is the interesting question, and the honest answer is that **the difference
between them is not meaningful**.

| Model | Accuracy | ROC AUC |
|---|---:|---:|
| Logistic Regression | 72.80% | 0.7900 |
| Support Vector Machine | 72.90% | 0.7899 |
| Decision Tree | 72.90% | 0.7920 |
| Random Forest | 73.39% | 0.7985 |
| XGBoost | 73.49% | 0.7982 |

The spread from the simplest model to the most sophisticated is **0.7 percentage
points**. XGBoost and Random Forest are separated by 0.10 points of accuracy and
0.0003 of AUC — both far smaller than the ±0.003 cross-validation standard
deviation. Declaring a winner from that would be reading noise.

The likely explanation is that the relationship really is close to linear in
these variables. Blood pressure, age and cholesterol contribute roughly
additively, and once `high_risk_flag` supplies the one clear interaction, there
is little structure left for a tree ensemble to find.

This is why the application defaults to averaging all five rather than shipping a
"best" model.

### Q19. Why report ROC AUC as well as accuracy?

Accuracy depends on the threshold; AUC does not.

Accuracy answers "at a 0.50 cut-off, what fraction did we get right?" AUC answers
"how well does the model *rank* patients by risk, across every possible
threshold?" — equivalently, the probability that a randomly chosen positive
patient is scored higher than a randomly chosen negative one.

That distinction matters here because the deployed threshold is **not** 0.50 —
the app screens at 0.40 (see Q20). AUC tells me the ranking is good at 0.80
regardless of where the cut-off is placed, which is precisely what made moving
the cut-off a safe, measured change rather than a gamble.

Accuracy is also famously misleading on imbalanced data — 95% accuracy is
worthless if 95% of patients are negative. This dataset is nearly balanced at
1.04 : 1, so accuracy is meaningful here, but reporting both is the right habit.

### Q20. Your recall is about 70%. Is that acceptable?

At the standard 0.50 cut-off, no — and that is exactly why the application does
not decide at 0.50.

At 0.50, XGBoost misses 1,986 genuinely at-risk patients out of 6,596. The two
error types are not symmetric: a false positive sends a healthy person for an
unnecessary check-up, while a false negative tells an at-risk person they are
fine. In a screening tool the second is much worse.

So the deployed decision threshold is `Config.POSITIVE_THRESHOLD = 0.40`. That
deliberately trades some precision for substantially higher recall — more false
alarms, fewer missed cases — which is the correct direction for screening.
`results.json` records every model's metrics at **both** cut-offs: 0.50 under
the headline keys so the numbers stay comparable with published results on this
dataset, and 0.40 under the `"screening"` key showing how the app actually
behaves. The same threshold is applied to every individual model's vote in
`ml_engine.predict()` (not scikit-learn's internal 0.50), and it equals the
Low/Moderate display boundary, so the yes/no verdict and the coloured badge can
never contradict each other.

What I would still not claim is that 0.40 is the *medically correct* number.
The right operating point depends on the cost of a follow-up test versus the
cost of a missed case, and setting it properly needs a clinician. 0.40 is a
defensible screening default with the trade-off measured and documented — and
it remains one constant in one file, so a clinician could move it.

I also measured the sensitivity of that choice. Moving the ensemble cut-off from
0.40 to 0.39 changes recall by only +1.2 points (79.1% → 80.4%) while adding 142
false alarms against 80 extra true cases caught; AUC, Brier score and the
calibration curve do not move at all because they are threshold-independent.
That is why 0.40 is kept as a clean, explainable default rather than tuned to a
second decimal place.

### Q20b. What is the recall of the ensemble — the verdict the app actually shows?

**79.1% at the 0.40 screening threshold** (precision 70.2%), measured on the
same 12,965-patient held-out set. At the comparability cut-off of 0.50 it is
69.6%, with ROC AUC 0.799 and Brier 0.182.

The ensemble is not a sixth trained model. `ml_engine.predict()` averages the
five probabilities and applies the threshold to the mean, so
`research/train_models.py` computes that same average on the test set and
scores it with the identical `_evaluate_proba()` function used for every
individual classifier. The result is stored in `results.json` under
`"Ensemble Voting"` and appears as the last row of the Model Performance page
with an "App verdict" badge. It has no cross-validation entry because there is
nothing to re-fit, and it is excluded from the "Best" badge because "best model"
refers to the five saved classifiers it is built from.

Its recall (79.1%) sits inside the range of the individual models (76.2–80.9%),
which is what averaging should produce: the five models share the same 15
features and mostly agree, so the ensemble cannot be dramatically better than
its members. Its value is stability — no dependence on any single model — and
the visible agreement signal ("4 of 5 models predict At Risk"), which turns a
split verdict into a "treat with caution" message rather than a confident
number.

### Q21. How do you know the models are not overfitting?

Four pieces of evidence:

1. **Held-out test set.** 12,965 patients never seen during training. All the
   headline numbers are measured there.
2. **Cross-validation agrees with it.** 5-fold CV on the training split gives
   accuracies within about 0.5 percentage points of the test-set numbers. If CV
   were much higher, I would suspect overfitting or leakage.
3. **Very small fold variance.** Standard deviations around ±0.003 mean
   performance barely moves between folds — the models are not memorising one
   particular split.
4. **Deliberate capacity limits.** The decision tree is capped at `max_depth=7`
   and `min_samples_split=20`; the random forest at depth 12; XGBoost uses
   `learning_rate=0.05` with `subsample=0.8` and `colsample_bytree=0.8`. An
   unrestricted tree on 52,000 rows would fit the training set almost perfectly
   and generalise poorly.

### Q22. How do you explain a prediction?

At two levels, and I should be clear which is which.

**Globally**, the training pipeline computes SHAP values and writes
`static/img/shap_summary.png`, shown on the Model Performance page. SHAP assigns
each feature a signed contribution using Shapley values from cooperative game
theory — the feature's average marginal contribution across all orderings. The
interface also shows tree-based feature importances.

It is computed once during training rather than per request because SHAP on a
tree ensemble takes seconds, which would make the result page unusable.

**Per patient**, the honest answer is that I do not. The user sees the
probability, the risk band, how many models agreed, and lifestyle guidance — but
not "your blood pressure contributed +18 points". That is the clearest piece of
future work, and it is a known limitation rather than an oversight.

I would also flag that impurity-based importance is biased towards
high-cardinality continuous features: `age` has thousands of possible split
points, `smoke` has two. Permutation importance would be fairer, and SHAP fairer
still.

---

## E. The application

### Q23. Explain your role model and how access control is enforced.

Three roles in a rising ladder: **Patient → Doctor → Admin**. Administrator is
the single highest tier and holds every privilege — an earlier version had a
separate SuperAdmin, which was merged, with a migration that folds any surviving
rows into Admin.

Enforcement is two decorators in `routes/security.py`. `@login_required`
redirects anonymous visitors to the login page, remembering where they were
going. `@role_required("doctor", "admin")` returns a proper 403 page to a
signed-in user with the wrong role.

Data scoping is one helper used everywhere:

```python
def _scope_user_id(user):
    return user["id"] if user["role"] == ROLE_PATIENT else None
```

`None` means "no filter", so the same query serves a patient viewing their own
history and a doctor viewing everyone's — one rule, one place, no second query to
keep in sync.

That scoping rule is also why the public register page creates **Patient**
accounts only. A Doctor sees every patient's records by design, so the Doctor
role must never be self-service: the server hard-codes the role and never
reads a posted one, and doctor logins are created only by an administrator on
the Admin → Doctors page (which reuses the same `validate_registration()`
rules with `allowed_roles=(ROLE_DOCTOR,)`).

Two invariants guard administration, in `_can_manage()`: nobody may act on their
own account, and the last active administrator cannot be demoted, suspended or
deleted. Promoting *to* admin is always allowed; only removal is guarded. Without
that, one careless click could leave the system with nobody able to administer
it.

### Q24. Why an application factory and blueprints?

`create_app()` builds and configures the application on demand, so the test suite
can construct a fresh, independently configured instance per run. A module-level
`app = Flask(__name__)` would tie configuration to import time.

Blueprints are split by **audience** — auth, patient, doctor, admin — rather than
by resource. Since access control here is role-based, grouping by role means the
guard on each file is uniform and easy to audit at a glance.

The admin blueprint reached 714 lines in one file, so I split it into a package
by subject (users, predictions, data, models, audits) while keeping a single
blueprint object. Every endpoint name is unchanged, so no template and no
`url_for` call needed touching.

### Q25. You validate in JavaScript and again in Python. Isn't that duplication?

It is duplication, and it is necessary.

The browser check is for **helpfulness** — the user sees the problem as they type
rather than after a round trip. The server check is for **correctness**. Anything
in the browser can be disabled, edited or bypassed entirely by posting with
`curl`, so client-side validation is a user-experience feature, never a security
control.

The duplication is contained: both sides read the same bounds, which are declared
once in `config.py` (`FIELD_RANGES`, `CHOICE_FIELDS`, the pulse-pressure and BMI
limits). Changing a rule means changing one constant and mirroring it in the JS
for feedback.

### Q26. How does the application handle failures?

Layer by layer.

- **Model loading.** Each model is unpickled in its own `try`. A missing file or
  a missing library disables that one model and records a warning shown on the
  page — it does not disable prediction. An earlier version returned `None, {}`
  if any single pickle failed, which took the whole system down.
- **Database writes.** Wrapped, logged, and flashed to the user. A failed save
  does not discard a computed result.
- **Uncaught exceptions.** A 500 handler logs the real traceback server-side and
  shows a plain page. The user never sees a stack trace, a file path or an SQL
  fragment.
- **Bad URLs and forbidden pages.** Dedicated 400, 403 and 404 templates.

One trap I hit and fixed: `abort(404)` raises Flask's `NotFound`, which is an
`Exception`, so an over-broad `except Exception` in a view swallowed it and
turned every 404 into a 500. `abort()` is now always called outside the `try`.

---

## F. Security

### Q27. How are passwords stored?

PBKDF2-HMAC-SHA256, with a 16-byte cryptographically random salt per user and
260,000 iterations, stored as `pbkdf2_sha256$iterations$salt$hash`.

Three properties matter:

- **Salt** — per user, so identical passwords produce different hashes and one
  precomputed rainbow table cannot attack the whole database.
- **Iterations** — 260,000 makes each guess deliberately expensive. A plain
  SHA-256 hash can be tested billions of times per second on a GPU.
- **Constant-time comparison** — `secrets.compare_digest`, so an attacker cannot
  learn the hash byte by byte from response timing.

An earlier version used unsalted SHA-256. Rather than lock those users out, the
verifier still accepts a legacy hash and silently upgrades it to PBKDF2 on the
next successful sign-in.

### Q28. How do you prevent SQL injection and CSRF?

**SQL injection**: every query in the project is parameterised, and every query
lives in `auth_db.py`. There is no string interpolation into SQL anywhere —
values are passed as `?` placeholders, so SQLite treats them as data and never as
syntax. Confining all SQL to one module makes that claim auditable rather than
just asserted.

**CSRF**: a `secrets.token_hex(32)` token is generated per session, exposed to
every template through a context processor, embedded in a hidden field in every
form, and compared with `compare_digest` in a `before_request` hook that fires on
every POST.

The security property is that an attacker's page can make your browser send a
request — the cookie goes along automatically — but it cannot *read* your session
cookie, so it cannot know the token. I implemented it rather than using Flask-WTF
because it is about thirty lines and the brief was to avoid unnecessary
dependencies.

### Q29. What other security measures did you implement?

| Threat | Defence |
|---|---|
| Session hijacking | `HttpOnly`, `SameSite=Lax`, `Secure` when `HDP_HTTPS=1` |
| Session fixation | `session.clear()` before storing the user, so a new session id is issued on every privilege change |
| Online brute force | Login rate limiting: 5 failures per username or IP within 15 min lock sign-in for 10 min; a correct login clears the counters; lockouts are audit-logged |
| Privilege escalation | Public registration is hard-coded to Patient — the posted role is never read. Doctor (which can read every patient's records) and Administrator are admin-created only |
| Open redirect | `?next=` only follows local paths — must start with `/`, not `//`, and no backslash, because browsers turn `/\evil.com` into `//evil.com` |
| Insecure direct object reference | `/report/<id>` verifies ownership before sending — your own report, or any report if you are clinical staff; otherwise 403 |
| Information disclosure | Real errors logged server-side, plain pages shown to users |
| Upload denial of service | `MAX_CONTENT_LENGTH` caps any upload at 32 MB |
| Accountability | Every state-changing action writes to `system_logs` |
| Secret management | `SECRET_KEY` comes from the environment; the fallback is clearly marked as development-only |

One design detail worth mentioning: `system_logs` stores the username as text
rather than joining to `users`. The audit trail has to survive the deletion of
the account it refers to — an entry saying "user 47 was deleted" is useless once
user 47 is gone.

### Q30. What would you do differently, and what would you do next?

**Differently:**

- **Write the verification harness first.** `verify_baseline.py` was built during
  the cleanup phase. Had it existed from the start, the 91% data-loss bug would
  have been caught the day it was introduced instead of much later.
- **Print row counts from the first version of the pipeline.** The bug was
  invisible precisely because nothing reported how much data survived each stage.
- **Design the role model once.** Building SuperAdmin and then merging it into
  Admin cost real time and required a data migration for a distinction that never
  earned its place.

**Next:**

1. **Per-prediction explanations.** The clearest gap. SHAP here is global; a user
   should be able to see which of *their* inputs drove *their* score.
2. **A live threshold tuner.** The cut-off has already been moved to a 0.40
   screening threshold with both operating points measured in `results.json`;
   the next step is exposing it to administrators with the precision/recall
   trade-off shown live, so the operating point becomes a clinical decision
   rather than a config constant.
3. **Hyperparameter search.** Nothing was tuned by grid search. Given that all
   five models land within 0.7 points, I expect little movement — but that is an
   expectation, not a measured result, and it should be measured.
4. **External validation.** The dataset's population, country and period are
   undocumented, so I cannot claim these models generalise. Validating on a
   second, independent dataset is the only way to find out.
5. **Production hardening.** PostgreSQL instead of SQLite, a real WSGI server,
   and HTTPS enforced. (Login rate limiting is already in place — five failures
   per username or IP lock sign-in for ten minutes.)

And the thing I would not change: the disclaimer on every page. At 73% accuracy
this is a screening aid and a teaching tool, not a diagnostic device, and the
interface should never let a user forget that.
