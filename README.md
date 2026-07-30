# Predicting individual survey answers with a gated ML + LLM ensemble

Our entry to the **AI Respondents Challenge** (Oxford LLMs 2026): predicting how individual
survey respondents answered questions that were hidden from us, using only their other answers.

Main notebook: [`pipeline_notebook_v4_documented.ipynb`](pipeline_notebook_v4_documented.ipynb)

---

## The challenge

Real people answered the World Values Survey (WVS). The organizers published 5,000 respondents
in full (the training set, 50 countries × 100 people) and 1,050 more (the test set, 35 countries
× 30 people) with their answers to 13 questions deleted. Those hidden answers exist — the
organizers keep them in a private table — and the task is to predict every one of them:

**1,050 respondents × 13 questions = 13,650 predictions**, each of which must exactly match one
of the question's official answer labels.

Two things make it interesting:

- **20 of the 35 test countries appear in training data, 15 do not.** Results are reported
  separately for both, so a method that merely memorises country-level patterns is exposed.
- **The ranking metric is *skill*, not accuracy:**

  ```
  skill = (accuracy − majority share) / (1 − majority share)
  ```

  Always guessing each question's most common answer scores 0. A perfect prediction scores 1.
  Doing *worse* than the majority answer scores negative — and the floor is far below −1, since
  on a question where 67% give the same answer there is much more room below the majority guess
  than above it. The metric therefore asks a sharp question: **does your method know anything
  about individual people, beyond the population average?**

The provided starter pipeline — an LLM shown six generic demographics per respondent — scored
**−0.30**, i.e. clearly worse than knowing nothing about the individual. That result shaped our
whole design.

---

## The pipeline

Two predictors work together. A classic machine-learning model makes most of the predictions;
an LLM handles the cases the model is unsure about; and a routing rule, learned from data,
decides which predictor answers what.

```
TRAINING DATA (5,000 labelled respondents)
  │
  ├─ [A] Gradient-boosting model, one per question, on all 278 allowed features
  │        ├─ cross-validation holding out whole countries → honest predictions + confidences
  │        └─ fit on all data                              → test predictions + confidences
  │
  ├─ [B] Statistics: Cramér's V between every feature and every target
  │        ├─ top 12 features per question → what the LLM prompt shows
  │        └─ the same values as weights   → similarity search below
  │
  ├─ [C] LLM evaluation on training respondents (answers known)
  │        ├─ 3 models × 2 prompt styles on 250 people → pick the best configuration
  │        └─ best configuration on 1,000 people       → reliable per-question scores
  │
  └─ [D] Routing rule per question: whichever of {model / LLM / model-if-confident-else-LLM}
          scored best in [C], measured with the official skill metric

TEST DATA (1,050 respondents, 13 hidden answers each)
  model predicts all 13,650 pairs → rule sends the ~11% least confident to the LLM
  → calibration check → predictions.csv + features.csv + method/ → submission
```

### [A] The machine-learning model

One `HistGradientBoostingClassifier` per question, trained on **all 278 allowed features** plus
a country code. Gradient boosting builds hundreds of small decision trees in sequence, each one
trained on the errors the previous trees left behind.

Two design choices worth naming:

- **All features, not a hand-picked subset.** Gradient boosting selects features internally, and
  it does so better than any filter we applied by hand: on one question, restricting the model to
  8 "obviously relevant" features scored +0.01, while giving it all 278 scored +0.26.
- **No imputation of missing answers.** The trees learn per split which branch missing values
  belong to. In surveys, non-response is informative — who refuses the income question is not
  random — so imputing would both fabricate answers and erase that signal.

Validation holds out **whole countries** rather than random respondents, mimicking the 15 test
countries with no training data. Every training respondent therefore also receives a prediction
from a model that never saw their country, which is what the routing rule is later tuned on.

### [B] Choosing what the LLM sees

A prompt cannot hold 278 answers, so for each question we measure which features actually
predict it, using **Cramér's V** (an association measure between two categorical variables,
0 = unrelated, 1 = one determines the other), and keep the top 12 with good coverage.

This recovers the block structure of the survey: confidence in parliament is best predicted by
confidence in political parties (V = 0.62) and in the government (0.57), not by age or education.
Demographics — the starter pipeline's entire evidence base — are weak predictors throughout.

### [C] The LLM

Each prompt describes the respondent through those 12 answers, adds **three similar training
respondents together with their true answers**, states the question, and lists the official
labels as a numbered menu. The similar respondents are found by nearest-neighbour search in the
space of the 12 features, weighted so that agreement on strongly predictive features counts more.
They matter because they anchor the model in how people in this dataset actually answer, rather
than in how it imagines people answer.

The answer is read from the **probabilities of the first generated token**, not from the reply
text: we request the 20 most likely first tokens and collect the probability mass on each option
digit. One call therefore yields a full probability distribution — the top option is the
prediction, its probability is the confidence — and no reply ever has to be parsed as prose.

We compared three models (Qwen3-32B, Qwen3-235B, Llama-3.3-70B) and two prompt styles
(third-person prediction vs. first-person roleplay) on the same training respondents. Results, all
within noise of each other, are in the notebook; Qwen3-235B with the third-person prompt scored
best and was used for the submission.

### [D] The routing rule

For each question we compare three options on 1,000 training respondents whose answers we know:
always the ML model, always the LLM, or the model when its confidence clears a threshold and the
LLM otherwise (thresholds 0.4–0.8 tested). The best-scoring option becomes that question's rule;
ties go to the ML model.

The outcome: pure ML on 5 questions, confidence-based routing on 8, pure LLM on none. On the test
set this sent **1,488 of 13,650 predictions (≈11%)** to the LLM — precisely the respondents the ML
model was least sure about.

### Safeguards

- **Format validation** — row count, respondent coverage, no missing predictions, and every
  predicted string checked against the official label space.
- **Calibration check** — predicted answer distributions per country compared against the
  training distributions. This caught a real bug: one target question is stored in the training
  data on a raw 1–10 scale while its official labels define 5 bins, so a naive mapping silently
  dropped and mislabelled ~80% of that question's training answers. Re-binning restored it and
  the question's predictions changed completely.

---

## Results

Final standings on the live leaderboard (16 Jul, 15:30 UTC). Our team: **the 5 of 4**.

**Prediction · In-domain** — seen countries, held-out respondents

| # | Team | Skill | Alignment | F1-macro | Coverage |
|---|------|-------|-----------|----------|----------|
| 🥇 | pre-dish salad | 0.360 | 0.949 | 0.568 | 100% |
| 🥈 | **the 5 of 4** | **0.280** | 0.933 | 0.522 | 100% |
| 🥉 | Group 6 | 0.275 | 0.935 | 0.519 | 100% |
| 4 | Group 2 | 0.162 | 0.956 | 0.471 | 100% |
| 5 | Group 5 | 0.147 | 0.873 | 0.444 | 100% |
| 6 | Group 3 | 0.100 | 0.855 | 0.461 | 100% |
| 7 | *Majority class (baseline)* | *−0.013* | *0.652* | *0.153* | *100%* |
| 8 | *Random draw (baseline)* | *−0.192* | *0.980* | *0.252* | *100%* |

**Prediction · Out-of-domain** — held-out countries, unseen populations

| # | Team | Skill | Alignment | F1-macro | Coverage |
|---|------|-------|-----------|----------|----------|
| 🥇 | pre-dish salad | 0.348 | 0.903 | 0.538 | 100% |
| 🥈 | Group 6 | 0.264 | 0.919 | 0.503 | 100% |
| 🥉 | **the 5 of 4** | **0.255** | 0.917 | 0.498 | 100% |
| 4 | Group 5 | 0.191 | 0.881 | 0.458 | 100% |
| 5 | Group 2 | 0.151 | 0.924 | 0.473 | 100% |
| 6 | Group 3 | 0.145 | 0.875 | 0.471 | 100% |
| 7 | *Majority class (baseline)* | *−0.002* | *0.650* | *0.157* | *100%* |
| 8 | *Random draw (baseline)* | *−0.209* | *0.958* | *0.259* | *100%* |

**Prediction · Hidden questions** — 4 questions never announced as targets

| # | Team | Skill | Alignment | F1-macro | Coverage |
|---|------|-------|-----------|----------|----------|
| 🥇 | **the 5 of 4** | **0.368** | 0.934 | 0.507 | 100% |
| 🥈 | pre-dish salad | 0.333 | 0.950 | 0.513 | 100% |
| 🥉 | Group 6 | 0.318 | 0.925 | 0.472 | 100% |
| 4 | Group 2 | 0.139 | 0.943 | 0.436 | 100% |
| 5 | *Majority class (baseline)* | *0.000* | *0.681* | *0.163* | *100%* |
| 6 | *Random draw (baseline)* | *−0.196* | *0.967* | *0.245* | *100%* |

**Prediction · Hidden questions, held-out countries** — compositional shift

| # | Team | Skill | Alignment | F1-macro | Coverage |
|---|------|-------|-----------|----------|----------|
| 🥇 | **the 5 of 4** | **0.240** | 0.924 | 0.430 | 100% |
| 🥈 | pre-dish salad | 0.218 | 0.926 | 0.420 | 100% |
| 🥉 | Group 6 | 0.203 | 0.914 | 0.406 | 100% |
| 4 | Group 2 | 0.047 | 0.938 | 0.392 | 100% |
| 5 | *Majority class (baseline)* | *−0.035* | *0.677* | *0.161* | *100%* |
| 6 | *Random draw (baseline)* | *−0.195* | *0.982* | *0.256* | *100%* |

**How to read the columns.** *Skill* is the ranking metric described above. *Alignment* measures
how close the predicted distribution of answers is to the true one (1 = identical); note that the
random baseline scores highest here, because always predicting the single most likely answer
narrows the distribution — a trade-off we accepted, since skill is what teams are ranked on.
*F1-macro* weights every answer option equally, so rare answers count as much as common ones.
*Coverage* is the share of respondent–question pairs answered; unanswered pairs count as wrong.

### What the numbers say

- **First place on both hidden-question boards** (0.368 and 0.240) — the four questions that were
  never announced as targets. Nothing in the pipeline could have been tuned towards them, so
  these boards test the method rather than the tuning, and are the results we are happiest with.
- **Second in-domain (0.280) and third out-of-domain (0.255)**, in both cases several times the
  majority-class baseline and far above the provided starter pipeline (−0.30).
- **The drop from seen to unseen countries is small on the original questions** (0.280 → 0.255).
  Predictions rest on what a person answered rather than on where they live, so performance
  largely survives populations the pipeline never saw. The drop is steeper on the hidden
  questions (0.368 → 0.240), where unfamiliar questions and unfamiliar populations compound.
- **Offline estimates held up.** Cross-validation on training data predicted ≈ +0.33 mean skill
  across the 13 questions; the boards returned an equivalent of ≈ +0.31 in-domain. The small
  shrinkage is expected — model, prompt and routing choices were all selected on those same
  validation samples — and holding out whole countries during validation is what kept the
  estimate close.
- **Alignment is consistently ~0.92–0.95** while the random baseline reaches 0.97. Always
  predicting the single most likely answer narrows the spread of predicted answers; that costs
  alignment and gains skill, which is the trade we deliberately made.

---

## Repository contents

| File | Description |
|------|-------------|
| `pipeline_notebook_v4_documented.ipynb` | The full pipeline, with outputs from the submission run |
| `eda_notebook.ipynb` | Exploratory analysis: target distributions, feature–target associations, offline scoring harness |
| `tier4_zero_shot_notebook.ipynb` | Held-out surveys (ESS Wave 11, Latinobarómetro 2023) — no training data, so LLM-only |
| `method_description.md` | Method write-up as submitted |

## Running it

The notebook runs in Google Colab as-is. It needs a
[Nebius AI Studio](https://studio.nebius.com/) API key for the LLM calls; the data downloads
automatically from the
[challenge dataset](https://huggingface.co/datasets/oxford-llms/ai-respondents-challenge).

The gradient-boosting stage alone produces a complete, valid submission and costs nothing —
set `RUN_LLM_EVAL = False` and `RUN_LLM_TEST = False` to skip all API calls. A full run with LLM
evaluation and test-time routing cost roughly $25 in total.

To run the pipeline on a different survey with the same file layout, point `LOCAL_DATA_DIR` at
the folder and run top to bottom; nothing is specific to the World Values Survey.
