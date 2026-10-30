# XAI report: global and local SHAP explanations and rejection reasons

> Educational use only. Not for real financial decisions. The alternative data is simulated, and the simulator uses the target for every row, so the effect of the alternative variables is larger than it would be in reality.

- Scope: the shipped model (`models/xgboost_v1.0.joblib`, XGBoost, 21 features, 13 monotone constraints). Global explanations use the **test set**, 22,500 applicants.
- Code: `src/explainer/shap_explainer.py` (TreeExplainer), `src/explainer/global_shap.py` (global figures), `src/explainer/reasons.py` (rejection reason sentences). Settings: `explainer` in `configs/config.yaml`.
- Run: `python -m src.explainer.global_shap` (see the README for passing a Korean font). API: `POST /explain` (local explanation, rejection reasons), `GET /explain/global` (global top 10). Dashboard: "XAI 분석" tab (waterfall).
- Figures: `reports/figures/13_shap_summary_{ko,en}.png`, `reports/figures/14_shap_dependence_{ko,en}.png`.

## 1. What SHAP is

SHAP splits one model prediction into **a sum of per-feature contributions**.

```
model output (default log-odds) = base value + SHAP of feature 1 + SHAP of feature 2 + ... + SHAP of feature 21
```

- **Base value** = -2.675: the log-odds of the average training applicant, about 6.4% default probability.
- A **positive SHAP value** means the feature raised default risk (toward rejection); a **negative** one means it lowered it (toward approval).
- The sum equals the model output exactly, so the explanation cannot disagree with the decision. Log-odds convert to default probability, then to a score, then to approval (`configs/scoring.yaml`).
- For tree models such as XGBoost, **TreeExplainer** computes exact SHAP values quickly (a few seconds for 22,500 applicants).

## 2. Global explanation: what the model relies on

### 2.1 Summary plot (`13_shap_summary_en.png`)

Each dot is one applicant. **Horizontal position** is the feature's SHAP value (right = higher risk); **color** is the feature value (orange = high, blue = low, gray = missing). The 15 features with the largest mean |SHAP| are shown, most important at the top.

| Rank | Feature | Mean \|SHAP\| | Actionability |
|---|---|---|---|
| 1 | App login frequency | 0.671 | Not recommended |
| 2 | Number of recurring payments | 0.553 | Not recommended |
| 3 | Spending consistency score | 0.528 | Actionable |
| 4 | Months with current phone carrier | 0.469 | Time-dependent |
| 5 | Health insurance / national pension payment months | 0.452 | Time-dependent |
| 6 | Revolving credit utilization rate | 0.415 | Actionable |
| 7 | Times 90+ days late | 0.216 | Non-decreasing |
| 8 | Age | 0.214 | Immutable (protected) |
| 9 | Times 30 to 59 days past due | 0.204 | Non-decreasing |
| 10 | Phone bill on-time payment rate | 0.152 | Actionable |

(Ranks 11 to 15: utility on-time rate, times 60 to 89 days past due, open loans and credit lines, debt ratio, real estate loans.)

**Reading and findings:**
1. **The top 5 are all alternative variables.** Following the mission rule, the simulator links them strongly to default (each mission variable alone reaches AUC 0.80 to 0.84). Real alternative data may be weaker.
2. **The color directions make sense.** Higher app logins, recurring payments, spending consistency and payment months (orange) sit on the left, lowering risk. Higher utilization, late payments and debt ratio sit on the right.
3. **Ranks 1 and 2 are "not recommended" variables.** They matter to the model, but advising an applicant to "log in more" would encourage gaming the score. Rejection reasons therefore show them only as "reference (not changeable)", and approval paths (DiCE) do not change them.
4. **For late payments, any versus none is what counts.** With no late payments (gray) the dots cluster near 0; with even one (orange) they sit at +0.5 or more.

### 2.2 Dependence plots (`14_shap_dependence_en.png`)

Each panel shows how the SHAP value (vertical) changes with the feature value (horizontal). The 4 features come from `explainer.dependence_features`: the top actionable variable, a key traditional credit variable, the protected attribute, and a monotone late-payment variable.

| Feature | Shape | Meaning |
|---|---|---|
| Spending consistency score | Falls in steps; drops to -1.5 to -2.0 above 90 | The strongest actionable variable. The audit found it in the most approval paths for declined applicants (1,329 single-variable paths) |
| Revolving utilization | About -0.5 at 0, +0.5 at 1 (100%); rises steadily | The monotone constraint (+1) holds. The steps near 0.3 and 0.5 are tree split points |
| Age | About +0.25 in the 20s and 30s, falls with age to about -0.5 in the 70s, rises slightly after 80 | No monotone constraint (protected). **It lowers young applicants' scores directly, which is the source of the age-band EO gap in the fairness report** (`reports/fairness_report_en.md`) |
| Times 90+ days late | About -0.2 at 0, +0.8 to +1.2 at 1, about +1.5 at 4 or more | The monotone constraint (+1) holds. The first late payment has the largest effect; later ones add less |

## 3. Local explanation: why was this applicant declined?

`POST /explain` computes SHAP values for one applicant. The dashboard draws them as a **waterfall plot**: bars that start at the base value and add each feature's SHAP value until they reach the final log-odds. Bars to the right (red) raised risk; bars to the left (green) lowered it.

Below are the API responses for 2 real declined applicants from the test set (applicants without cleaning flags; the model score and the API score were checked to match).

### Applicant A: thin-filer, no late payments, score 466 (grade D, default probability 11.3%)

Key values: age 34, revolving utilization 100%, 2 open loans and credit lines, 0 real estate loans (thin-filer), 14 months of health insurance payments, 17 months with the phone carrier, spending consistency 71.4, utility on-time rate 58%.

| Feature | SHAP | Direction |
|---|---|---|
| Health insurance / pension payment months (14) | +0.613 | Higher risk |
| Revolving utilization (100%) | +0.608 | Higher risk |
| Spending consistency score (71.4) | -0.269 | Lower risk |
| Months with phone carrier (17) | +0.252 | Higher risk |
| Age (34) | +0.240 | Higher risk |
| App login frequency (22) | -0.195 | Lower risk |
| Open loans and credit lines (2) | -0.191 | Lower risk |
| Times 90+ days late (0) | -0.171 | Lower risk |

Base value -2.675 + SHAP total +0.616 = -2.059 (default probability 11.3%, score 466). 9 points short of the 475 cutoff.

### Applicant B: not a thin-filer, one 90+ days late, score 454 (grade D, default probability 13.1%)

| Feature | SHAP | Direction |
|---|---|---|
| Times 90+ days late (1) | +0.972 | Higher risk |
| Health insurance / pension payment months (21) | -0.878 | Lower risk |
| Months with phone carrier (80) | -0.811 | Lower risk |
| App login frequency (19) | +0.738 | Higher risk |
| Spending consistency score (48.8) | +0.466 | Higher risk |
| Revolving utilization (73%) | +0.430 | Higher risk |

Base value -2.675 + SHAP total +0.786 = -1.889 (default probability 13.1%, score 454).

## 4. Automatic rejection reasons

`src/explainer/reasons.py` builds rejection reasons from the SHAP values.

1. Keep only features that raised default risk (SHAP at least 0.01, `rejection_reason_min_contribution`).
2. Take up to 5 (`rejection_reasons_top_k`), largest SHAP first.
3. Turn each into a sentence: feature name, current value, **comparison with the training-data median**, and contribution. Value formats (%, times, months, won) and Korean particles are handled automatically.
4. As in the mission example, contributions are shown **in the approval direction** (= -SHAP; negative = toward rejection).
5. Features the applicant should not or cannot change ("not recommended", "immutable" and "non-decreasing" classes, `explainer.reference_only_classes`) get "[참고(변경 불가)]" (reference, not changeable): information, not advice. Actionable and time-dependent features do not.

**Applicant A's reasons (API output, Korean as served, translated here):**

1. Health insurance / pension payment months is 14 months, shorter than the training median (18 months) (contribution: -0.61)
2. Revolving utilization is 100%, higher than the training median (15%) (contribution: -0.61)
3. Months with the phone carrier is 17, shorter than the training median (40) (contribution: -0.25)
4. Age is 34, lower than the training median (52) (contribution: -0.24) [reference (not changeable)]
5. Utility on-time rate is 58%, lower than the training median (83%) (contribution: -0.13)

**Applicant B's reasons:**

1. Times 90+ days late is 1, more than the training median (0) (contribution: -0.97) [reference (not changeable)]
2. App login frequency is 19, fewer than the training median (23) (contribution: -0.74) [reference (not changeable)]
3. Spending consistency score is 48.8, lower than the training median (73.6) (contribution: -0.47)
4. Revolving utilization is 73%, higher than the training median (15%) (contribution: -0.43)
5. Age is 33, lower than the training median (52) (contribution: -0.17) [reference (not changeable)]

**Reading:**
- Applicant A has no late payments but was declined for a short history (14 months of payments, 17 months with the carrier) and a maxed-out card limit: a typical thin-filer rejection. Lowering utilization helps right away; payment months and carrier tenure improve with time.
- Applicant B's main reason is one past 90+ day late payment, which cannot be reduced. A long payment history (21 and 80 months) lowered the risk a lot, but not enough to offset the late payment and low spending consistency.
- **Age** appears in both applicants' reasons: the age effect of section 2.2 shows up at the individual level too.

## 5. Limitations and open items

- **The effect of the alternative variables is inflated.** The simulator built them from the real default outcome, so the top 5 being alternative variables is a result of the simulation design.
- **SHAP describes the model, not causes.** "Raising spending consistency raises the score" is true for the model, but that behavior is not guaranteed to lower real default risk.
- **The comparison point is the training-data median.** "Higher/lower than the median" only names the comparison; whether the value hurts is decided by the SHAP contribution.
- **Reasons that cannot be changed are still shown.** Age or past late payments are real reasons for the decision, so they are not hidden but marked "reference (not changeable)". Applicants learn the true reasons, but fewer of them may be actionable (2 of 5 for applicant B).
- Time-dependent features (payment months, tenure) should be framed as waiting time, not as actions. The current rejection sentences only state facts, so they comply, but approval-path (DiCE) wording must follow this rule.
