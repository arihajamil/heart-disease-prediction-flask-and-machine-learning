# The Machine Learning Pipeline

From a 70,000-row CSV to five pickled classifiers, and from a web form back to a
probability.

Everything here is implemented in
[research/train_models.py](../research/train_models.py) (training) and
[ml_engine.py](../ml_engine.py) (inference).

---

## Contents

1. [The dataset](#1-the-dataset)
2. [The bug that nearly invalidated the whole project](#2-the-bug-that-nearly-invalidated-the-whole-project)
3. [Preprocessing, step by step](#3-preprocessing-step-by-step)
4. [The 15 features](#4-the-15-features)
5. [Split, then scale — in that order](#5-split-then-scale--in-that-order)
6. [The five models](#6-the-five-models)
7. [Results](#7-results)
8. [Cross-validation](#8-cross-validation)
9. [Explainability](#9-explainability)
10. [Inference: what happens at prediction time](#10-inference-what-happens-at-prediction-time)
11. [Reproducibility](#11-reproducibility)
12. [Honest limitations](#12-honest-limitations)

---

## 1. The dataset

**Cardiovascular Disease dataset** — Svetlana Ulianova, Kaggle.
<https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset>

70,000 patient records from medical examinations. Three kinds of variable:

| Kind | Columns | How reliable |
|---|---|---|
| Objective — measured | `age`, `height`, `weight`, `gender` | High |
| Examination — tested | `ap_hi`, `ap_lo`, `cholesterol`, `gluc` | High |
| Subjective — self-reported | `smoke`, `alco`, `active` | Lower; people under-report |

Target: `cardio`, 1 when cardiovascular disease is present.

Two quirks of the raw file matter:

- **`age` is stored in days**, not years — around 20,000, not around 55.
- **`cholesterol` and `gluc` are stored 0-indexed** (0, 1, 2), while the standard
  clinical scale and everything else in this project uses 1, 2, 3.

---

## 2. The bug that nearly invalidated the whole project

This is worth reading before anything else, because it is the single most
important thing that happened to this pipeline.

The domain filter contains this rule:

```python
masks.append(df[col].isin([1, 2, 3]))     # cholesterol, gluc
```

Perfectly reasonable — except that at the point it ran, those columns still held
0, 1 and 2. Every row whose cholesterol was `0` (the *most common* value, meaning
normal) failed the test and was discarded.

**Result: 70,000 rows reduced to 6,583. About 91% of the data thrown away, in
silence.** Nothing crashed. The pipeline finished, printed a report and produced
models that scored plausibly.

Worse, the loss was not random. Dropping every "normal cholesterol" patient
biased the surviving sample towards sicker people, so the models were trained on
a population that does not exist.

### The fix

One line in `_basic_clean()`, applied *before* any filtering:

```python
# The published file stores these 0-indexed; every downstream rule
# (and the web form) uses 1 = normal, 2 = above, 3 = well above.
df[col] = df[col] + 1
```

### The effect

| | Before | After |
|---|---:|---:|
| Rows surviving preprocessing | 6,583 | **64,825** |
| Proportion of the data retained | 9.4% | **92.6%** |
| Class balance | skewed | **1.04 : 1** |
| Test-set ROC AUC | ~0.72 | **~0.80** |

### What to take from it

A silent 91% data loss is far more dangerous than a crash. A crash tells you
something is wrong; this produced a confident, plausible, wrong model. The
pipeline now prints the row count after every stage, so the next such loss is
visible on the first run.

---

## 3. Preprocessing, step by step

Thirteen steps. The row count is printed after each one.

### Step 1 — Load
Read the CSV exactly as supplied. 70,000 rows.

### Step 2 — Basic housekeeping
- Drop the `id` column: a database key carries no clinical signal, and leaving it
  in invites the model to memorise individuals.
- Convert `age` from days to whole years.
- **Shift `cholesterol` and `gluc` from 0–2 to 1–3** — see section 2.

### Step 3 — Remove duplicates
Drop exact duplicate rows. Not just tidiness: an identical patient appearing in
both the training and the test split leaks the answer and inflates the reported
score.

### Step 4 — Physiological domain filter
Remove rows that describe a body that cannot exist. These are **not** statistical
outlier rules — each bound is a hard physical limit, so a value outside it is a
recording error.

| Column | Kept range | Why |
|---|---|---|
| `age` | 1 – 120 years | — |
| `ap_hi` (systolic) | 60 – 250 mmHg | Below 60 is not survivable; the raw file contains values like 12 and 16,020 |
| `ap_lo` (diastolic) | 40 – 200 mmHg | The raw file contains values like 0 and 10,000 |
| `ap_hi` vs `ap_lo` | systolic **>** diastolic | Always true in a living person |
| `height` | 100 – 250 cm | — |
| `weight` | 20 – 300 kg | — |
| `cholesterol`, `gluc` | 1, 2, 3 | The valid clinical codes |

Together with step 3 this removes about 7.4% of the file.

### Step 5 — IQR winsorization
For each numeric column, clip values outside `Q1 − 1.5·IQR` to `Q3 + 1.5·IQR`
back to those fences.

**Clip, not delete.** These values are extreme but possible — a genuinely obese
patient, a genuinely hypertensive one. The patient is real; only the reading is
implausible. Deleting the row would throw away nine good measurements to fix one
questionable one, and would systematically remove the sickest patients, which is
precisely the group the model most needs to learn.

### Step 6 — Median imputation
Fill any remaining gaps with the column median. Median rather than mean because
one erroneous reading of 10,000 mmHg drags a mean but leaves a median untouched.

### Step 7 — Feature engineering
Derive the four extra features. See section 4.

### Step 8 — Correlation pruning
Drop one column from any pair with `|r| ≥ 0.90`. Two columns carrying the same
information add no signal but make linear models unstable, because the
coefficients can trade off against each other without changing the fit.

On this dataset nothing exceeds the threshold, so all 15 features survive. The
step is kept because it would matter on a different dataset — and because
demonstrating that the check was made is itself the result.

### Step 9 — Stratified 80/20 split
```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
51,860 training rows, 12,965 test rows. `stratify=y` keeps the class ratio
identical in both halves, so the test set is representative rather than lucky.

### Step 10 — Standardisation
Fit a `StandardScaler` **on the training split only**. See section 5.

### Step 11 — Train the five classifiers
See section 6.

### Step 12 — 5-fold stratified cross-validation
On the training split only, with the scaler refitted inside each fold. See
section 8.

### Step 13 — SHAP summary image
Write `static/img/shap_summary.png`. See section 9.

The console output of the last run is saved to `models/preprocess_report.txt`.

---

## 4. The 15 features

Eleven measured, four derived.

| # | Feature | Meaning | Source |
|---:|---|---|---|
| 0 | `age` | Whole years | Form |
| 1 | `gender` | 0 female, 1 male | Form |
| 2 | `height` | Centimetres | Form |
| 3 | `weight` | Kilograms | Form |
| 4 | `ap_hi` | Systolic BP, mmHg | Form |
| 5 | `ap_lo` | Diastolic BP, mmHg | Form |
| 6 | `cholesterol` | 1 normal, 2 above, 3 well above | Form |
| 7 | `gluc` | 1 normal, 2 above, 3 well above | Form |
| 8 | `smoke` | 0 no, 1 yes | Form |
| 9 | `alco` | 0 no, 1 yes | Form |
| 10 | `active` | 0 no, 1 yes | Form |
| 11 | `bmi` | `weight / (height/100)²`, clipped to 10–70 | Derived |
| 12 | `pulse_pressure` | `ap_hi − ap_lo` | Derived |
| 13 | `age_group` | 0–4, from bins `[0, 30, 45, 60, 75, 120]` | Derived |
| 14 | `high_risk_flag` | 1 when `cholesterol ≥ 2` **and** `ap_hi ≥ 140` | Derived |

### Why derive features at all, if the models could learn them?

In principle a tree ensemble could discover BMI from height and weight. In
practice, giving it directly:

- **`bmi`** encodes a genuine non-linear relationship (a ratio involving a
  square) that a linear model cannot represent at all;
- **`pulse_pressure`** is an established cardiovascular marker in its own right —
  arterial stiffness shows up in the *gap* between the two readings, not in
  either one alone;
- **`age_group`** lets the models express "risk jumps after 60" without having to
  approximate a step with a smooth function;
- **`high_risk_flag`** encodes a clinical interaction. Raised cholesterol *and*
  hypertension together are worse than the sum of their separate effects, and an
  additive model cannot express that without being told.

The honest counter-argument: each derived feature is correlated with its parents,
which is exactly why step 8 exists. In this case none crosses the 0.90 threshold.

### The order is a contract

`Config.FEATURE_ORDER` and `models/features.json` list these names in this order,
and `scaler.pkl` matches values **by position, not by name**. It stores one mean
and one standard deviation per slot; it has no idea the third column is called
`height`.

Reorder the list without retraining and every value after the change is scaled
with the wrong statistics. Nothing crashes. The application keeps producing
confident numbers that are simply wrong. This is why `verify_baseline.py` stores
the entire feature vector, not just the final probability.

---

## 5. Split, then scale — in that order

This is the classic examiner question, so it is worth being precise.

```python
X_train, X_test, y_train, y_test = train_test_split(..., stratify=y)   # FIRST
scaler = StandardScaler().fit(X_train)                                  # THEN
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)      # transform only, never fit
```

### Why scale at all?

The raw columns are on wildly different scales: `age` around 55, `height` around
165, `smoke` either 0 or 1. Logistic Regression and SVM optimise a distance or a
weighted sum, so a column with bigger numbers dominates purely because its
numbers are bigger — nothing to do with how informative it is. Standardisation
rewrites every column as "how many standard deviations from the mean", putting
them on equal footing.

Tree-based models do not need it (a tree splits on thresholds, and any monotonic
rescaling gives the same tree). Scaling them anyway costs nothing and means every
model in this project consumes exactly the same input, which removes a whole
class of bug.

### Why fit only on the training data?

`StandardScaler.fit()` computes the mean and standard deviation of each column.
If it sees the test rows, those rows' statistics are baked into the transform
that is applied to the training data — the model has been given a summary of data
it is supposed to be evaluated on. That is **data leakage**, and it makes the
reported score optimistic in a way that does not survive contact with real data.

The same discipline applies inside cross-validation: the scaler is rebuilt inside
each fold, from that fold's training portion only (section 8).

### And at prediction time?

The pickled scaler is loaded and reused. Never refitted. Fitting on one patient
would be meaningless — a single row has no spread, so every standardised value
would be zero — and fitting on live data would measure patients against a
different population from the one the models learned on.

---

## 6. The five models

| Model | Configuration | Why it is included |
|---|---|---|
| **Logistic Regression** | `max_iter=2000, C=1.0, class_weight="balanced"` | The interpretable linear baseline. Every coefficient is directly readable as a log-odds contribution. If something complicated cannot beat this, the complexity is not earning its place |
| **Support Vector Machine** | `LinearSVC` wrapped in `CalibratedClassifierCV(cv=3)` | A different decision principle: maximise the margin between classes. `LinearSVC` for speed on 52,000 rows; the calibration wrapper is what gives it a usable `predict_proba` |
| **Decision Tree** | `max_depth=7, min_samples_split=20, class_weight="balanced"` | The only model a non-technical person can follow end to end. Depth-limited, because an unrestricted tree memorises the training set |
| **Random Forest** | `n_estimators=200, max_depth=12, min_samples_split=10` | Bagging: 200 trees on bootstrap samples, averaged. Individually weak trees, collectively stable |
| **XGBoost** | `n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8` | Boosting: each tree corrects its predecessors' errors. Usually the strongest on structured tabular data |

Every model sets `random_state=42`.

### On `class_weight="balanced"` and `scale_pos_weight`

The dataset is nearly balanced (1.04 : 1) so this changes very little here. It is
set anyway because it makes the pipeline correct on an imbalanced dataset without
a code change — the models learn to weight the rarer class in proportion to its
scarcity rather than optimising accuracy by ignoring it.

### Why an SVM needs calibration

`LinearSVC` outputs a signed distance from the separating hyperplane, not a
probability. `CalibratedClassifierCV` fits a small model mapping that distance
onto [0, 1] using cross-validation. Without it there is no `predict_proba`, and
the SVM could not join an ensemble that averages probabilities.

---

## 7. Results

Measured on the held-out test set of **12,965 patients** never seen during
training.

| Model | Accuracy | ROC AUC | F1 | Precision | Recall | Train (s) | Predict (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 72.80% | 0.7900 | 0.7197 | 0.7562 | 0.6866 | 0.05 | 0.3 |
| Support Vector Machine | 72.90% | 0.7899 | 0.7243 | 0.7505 | 0.7000 | 0.45 | 4.7 |
| Decision Tree | 72.90% | 0.7920 | 0.7187 | **0.7617** | 0.6803 | 0.20 | 0.6 |
| Random Forest | 73.39% | **0.7985** | 0.7256 | 0.7631 | 0.6916 | 2.92 | 62.3 |
| XGBoost | **73.49%** | 0.7982 | **0.7285** | 0.7606 | **0.6989** | 1.66 | 6.3 |

Prediction time is for 1,000 rows.

### Reading these numbers honestly

**The spread is 0.7 percentage points.** From the simplest model to the most
sophisticated. That is the most interesting result in the table, and it deserves
to be stated plainly rather than buried: on this dataset, with these features,
gradient boosting buys almost nothing over logistic regression.

The likely reason is that the relationship really is close to linear in these
variables. Blood pressure, age and cholesterol contribute in a roughly additive
way, and there is not much interaction structure left for a tree ensemble to find
once `high_risk_flag` has been provided explicitly.

**Random Forest and XGBoost are tied.** The 0.10-point accuracy gap and the
0.0003 AUC gap are both far smaller than the ±0.003 cross-validation standard
deviation. Declaring either "the best model" would be reading noise. That is a
substantive reason the application defaults to averaging all five rather than
picking a winner.

### Confusion matrices

Rows are actual, columns predicted.

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 4,909 | 1,460 | 2,067 | 4,529 |
| Support Vector Machine | 4,834 | 1,535 | 1,979 | 4,617 |
| Decision Tree | 4,965 | 1,404 | 2,109 | 4,487 |
| Random Forest | 4,953 | 1,416 | 2,034 | 4,562 |
| XGBoost | 4,918 | 1,451 | 1,986 | 4,610 |

**Recall matters more than precision here, and every model is weakest exactly
where it matters most.** At the standard 0.50 cut-off XGBoost misses 1,986
genuinely at-risk patients out of 6,596 — a recall of 69.9%. A false positive
sends a healthy person for an unnecessary check-up. A false negative tells an
at-risk person they are fine.

**This is why the application does not decide at 0.50.** The deployed decision
threshold is `Config.POSITIVE_THRESHOLD = 0.40`: the cut-off is deliberately
lowered, accepting more false alarms to catch more real cases — the correct
trade for a screening tool. The table above is still reported at 0.50 so the
figures stay comparable with published results on this dataset;
`results.json` additionally records every model's recall, precision, F1 and
confusion matrix at the 0.40 screening threshold (under the `"screening"`
key), so both behaviours are measured. The precise value of the threshold
remains a clinical judgement rather than a purely statistical one — 0.40 is a
defensible screening default, not a medically validated constant, and a real
deployment would set it with a clinician.

---

## 8. Cross-validation

5-fold stratified CV, run on the training split only, with the scaler refitted
inside every fold.

| Model | CV Accuracy | CV ROC AUC |
|---|---|---|
| Logistic Regression | 0.7241 ± 0.0013 | 0.7886 ± 0.0028 |
| Support Vector Machine | 0.7239 ± 0.0016 | 0.7883 ± 0.0028 |
| Decision Tree | 0.7239 ± 0.0026 | 0.7886 ± 0.0047 |
| Random Forest | 0.7275 ± 0.0031 | 0.7959 ± 0.0034 |
| XGBoost | 0.7284 ± 0.0046 | 0.7950 ± 0.0035 |

Three things this tells us:

1. **The models are stable.** Standard deviations around 0.003 mean performance
   barely moves between folds, so the single test-set number is not a fluke of
   one lucky split.
2. **CV and test-set scores agree** to within about 0.5 percentage points. If CV
   were much higher than the test score, we would suspect leakage or overfitting.
3. **The model ranking is inside the noise.** XGBoost leads on CV accuracy by
   0.0009 with a standard deviation of 0.0046. That is not a difference.

### Why the scaler is refitted inside each fold

The pipeline is built with `Pipeline([("scaler", StandardScaler()), ("model",
clone(model))])` and handed to `cross_validate`. scikit-learn then refits the
whole pipeline inside each fold.

Scaling once before CV would let every fold's validation portion contribute to
the mean and standard deviation used to transform its own training portion —
leakage, just at a smaller scale. The 20% test set is excluded from CV entirely,
so it stays genuinely untouched.

---

## 9. Explainability

### SHAP

Step 13 computes SHAP values on a sample of the training data and writes
`static/img/shap_summary.png`. The Model Performance page displays it.

SHAP assigns each feature a contribution to each prediction, based on Shapley
values from cooperative game theory: the feature's average marginal contribution
across all possible orderings. It is model-agnostic and, unlike a raw importance
score, it is signed — it says whether a feature pushed the prediction up or down.

**Computed once, not per request.** SHAP on a tree ensemble takes seconds. Doing
it per prediction would make the result page unusable.

The trade-off, stated plainly: this is a **global** explanation — which features
matter across the population — not a per-patient one. "Why did *this* patient get
83%?" is not answered by this image. Per-prediction attribution would be the
obvious next piece of work.

### Feature importance

`ml_engine.get_feature_importances()` reads `feature_importances_` from the tree
models, normalises them to sum to 1, and the interface displays the top twelve.

A caveat worth knowing: impurity-based importance is biased towards
high-cardinality continuous features. `age` and `ap_hi` have thousands of
possible split points; `smoke` has two. That alone gives the continuous features
more opportunities to look important. Permutation importance would be fairer;
SHAP is fairer still, which is why both are shown.

---

## 10. Inference: what happens at prediction time

```
Form (11 values)
   │
   ▼  ml_engine.validate_form()      ranges, systolic > diastolic, plausible BMI
   │                                  — repeated on the server, never trusted from JS
   ▼  ml_engine.engineer_features()  11 → 15, in FEATURE_ORDER
   │
   ▼  scaler.transform()             the pickled scaler, reused not refitted
   │
   ▼  five × predict_proba()[0][1]   P(disease), one per model
   │
   ▼  average                        the ensemble probability
   │
   ▼  ≥ 0.40 → "At Risk"             Config.POSITIVE_THRESHOLD (screening cut-off)
   │
   ▼  get_risk_level()               < 0.40 Low · < 0.70 Moderate · else High
   │
   ▼  save + render
```

### The threshold and the bands — aligned by design

| | Value | What it decides |
|---|---|---|
| `POSITIVE_THRESHOLD` | 0.40 | The yes/no classification. "At Risk" or not |
| `RISK_LOW_MAX` / `RISK_MODERATE_MAX` | 0.40 / 0.70 | The colour and wording shown to the user |

The decision threshold deliberately **equals** the Low/Moderate band boundary,
so the verdict and the badge can never disagree: every "at risk" patient is
shown Moderate or High, and every "not at risk" patient is shown Low.

That was not always true. An earlier version decided at 0.50 while the bands
switched at 0.40 — so a patient at 0.45 was classified *not at risk* yet wore a
*Moderate Risk* badge. Both numbers were individually defensible, but together
they contradicted each other on screen. Aligning them (and lowering the
decision cut-off for screening reasons — see section 7) removed the
contradiction at the source rather than papering over it in the template.

The threshold is applied in exactly one place — `ml_engine.predict()` — to
every model's probability, deliberately **not** via each model's own
`predict()`, which is hard-wired to 0.50 inside scikit-learn. One shared rule
means the ensemble verdict, every individual vote, and the "N of 5 models"
agreement count all move together when the config value changes.

### Why the individual votes are kept

The ensemble could return one number. It returns all five, because agreement is
real information: five models out of five saying "at risk" is a much stronger
signal than three out of five, even when the average is identical. The interface
shows "4 of 5 models predict At Risk" for exactly this reason.

---

## 11. Reproducibility

Every source of randomness is pinned to `random_state=42`: the train/test split,
the cross-validation folds, and all five classifiers.

This is not a formality. During this project the training script was started by
accident and rewrote all five `.pkl` files. Git reported them **byte-identical**
to the committed versions, and `verify_baseline.py` passed unchanged — which is
about as direct a demonstration of reproducibility as one could ask for.

To reproduce from scratch:

```bash
python research/train_models.py   # ~1-2 minutes
python verify_baseline.py         # must still say PASS
python test_project.py            # 150 checks
```

`requirements.txt` pins exact versions. That matters most for `scikit-learn`: a
`.pkl` written by one version may refuse to load, or load with subtly different
behaviour, under another.

---

## 12. Honest limitations

Worth raising yourself rather than waiting to be asked.

1. **73% accuracy is not clinical-grade.** Roughly one prediction in four is
   wrong. This is a screening aid and an educational tool, and the disclaimer on
   every page says so.

2. **Recall is the weak point, and it is the wrong place to be weak.** About 30%
   of at-risk patients are missed. Reducing the threshold would trade false
   alarms for catches; choosing that trade-off needs a clinician.

3. **Three inputs are self-reported.** Smoking, alcohol and physical activity are
   systematically under-reported, so the models likely underestimate their true
   effect.

4. **The dataset is not described.** The country, the years, and the recruitment
   method are not documented, so we cannot know which population these models
   generalise to. A model trained on one population can be badly wrong on
   another.

5. **Correlation, not causation.** The models find patterns associated with
   disease. Nothing here establishes that any feature *causes* it.

6. **No temporal validation.** Rows are shuffled randomly. A deployed clinical
   model should be validated on data from a later period than it was trained on,
   because populations and measurement practice drift.

7. **Hyperparameters were chosen by judgement, not by search.** No grid search or
   Bayesian optimisation was run. Given that all five models land within 0.7
   points of one another, tuning would probably move the numbers very little —
   but that is an expectation, not a measured result.

8. **The explanation is global, not individual.** See section 9.
