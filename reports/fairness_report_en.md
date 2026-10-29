# Fairness check and bias mitigation report

> Educational use only. Not for real financial decisions. Gender is a simulated value, generated independent of default.

- Code: `src/fairness/` (metrics `metrics.py`, mitigation `mitigation.py`, inputs `data.py`). Tests: `tests/test_fairness_metrics.py`, `tests/test_fairness_mitigation.py`.
- Run: `python -m src.fairness` (a few seconds), `python -m src.fairness.mitigation` (about 1 to 2 minutes, mostly retraining).
- Figures: `reports/fairness/fairness_metrics.json`, `reports/fairness/mitigation.json`. API: `GET /fairness`, `GET /fairness/mitigation`.
- Scope: decisions of the shipped model (`models/xgboost_v1.0.joblib`) on the **test set** (22,500 applicants). Approval = score 475 or higher (`configs/scoring.yaml`).
- Thresholds (mission checklist, `fairness` in `configs/config.yaml`): **DI ratio at least 0.8, Equalized Odds gap at most 0.10.**

## Summary

| Item | Result | Verdict |
|---|---|---|
| Gender fairness (before) | DI 0.995, EO gap 0.038 | Pass |
| Age-band fairness (before) | DI 0.832, **EO gap 0.121** | DI pass, **EO fail** |
| Mitigation A. Cutoff per age band | Age DI 0.840, EO gap 0.097 | Pass (margin 0.003) |
| **Mitigation B. Reweighting (adopted)** | Age DI 0.912, **EO gap 0.025** | **Pass (margin 0.075)** |
| Cost of mitigation (B) | AUC 0.9492 to 0.9458, default rate among approved 1.46% to 1.49% | Small |

The API and dashboard show B as "after" (`fairness.mitigation.api_method: reweighting`). A's results are in the same JSON file.

## 1. Metric definitions

The model scores each applicant and approves at 475 or higher. The fairness check compares these decisions across the groups of a protected attribute (gender, age band). The "positive" outcome is **approval**.

| Metric | Definition | Meaning |
|---|---|---|
| Approval rate | Share of the group approved | |
| TPR | Approval rate among applicants who repaid | Higher is better (approve those who repay) |
| FPR | Approval rate among applicants who defaulted | Lower is better (decline those who default) |
| **DI ratio** | Lowest approval rate / highest approval rate | Pass at 0.8 or higher (US "four-fifths rule") |
| **EO gap** | The larger of the largest TPR gap and the largest FPR gap between groups | Pass at 0.10 or lower |

DI looks only at outcomes. When groups have different real default rates (11.8% at 20-34, 3.5% at 55+), an approval gap is not discrimination by itself. EO compares **people with the same outcome (repaid or defaulted)**, which addresses this.

Implementation notes:
- Each group's approval rate carries a 95% interval (normal approximation).
- A group with no defaulters has no FPR; it is left empty (`null`) and skipped in the gap.
- Thresholds are read from `configs/config.yaml`, not written in code. A WARNING is logged when one is broken.
- Check: on the same data the module reproduces the earlier audit (`reports/audit/2_fairness.json`, `notebooks/audit.py`) to the fourth decimal (`test_reproduces_audit_on_test_set`).

## 2. Before mitigation

**Gender (simulated):**

| Group | Rows | Default rate | Approval rate | TPR | FPR |
|---|---|---|---|---|---|
| Female | 11,287 | 6.6% | 86.7% | 0.916 | 0.171 |
| Male | 11,213 | 6.7% | 87.2% | 0.920 | 0.209 |

DI 0.995, TPR gap 0.003, FPR gap 0.038. **Pass.**

**Age band:**

| Group | Rows | Default rate | Approval rate | TPR | FPR |
|---|---|---|---|---|---|
| 20-34 | 2,970 | 11.8% | 77.4% | 0.857 | **0.149** |
| 35-54 | 9,753 | 8.4% | 83.8% | 0.898 | 0.175 |
| 55+ | 9,777 | 3.5% | 93.0% | 0.954 | **0.269** |

DI 0.832 (pass), TPR gap 0.096, **FPR gap 0.121 (fail).**

**Reading:** applicants aged 55+ default rarely, so the whole group scores high. As a result 26.9% of 55+ applicants who later default are approved (14.9% at 20-34). In the other direction, young applicants who repay are approved less often (85.7% vs 95.4%). The model is more lenient with older applicants.

## 3. Mitigation methods

Both methods target the age band (`fairness.mitigation.attribute`). Gender already passes, so it is only checked for side effects.

### A. Cutoff per age band (threshold adjustment, post-processing)

The model stays as it is; each age band gets its own approval cutoff instead of 475.

- **Data use:** cutoffs are chosen on the **validation set** and applied to the test set once. Choosing them on the test set would overstate the result.
- **Search:** 31 candidates per age band (400 to 550, step 5). All 31 x 31 x 31 = 29,791 combinations are tried.
- **Rules:** (1) EO gap at most 0.10, (2) DI within 0.8 to 1.25, (3) default rate among approved applicants not above the single-cutoff (475) rate, so risk does not go up.
- **Choice:** among the combinations that meet the rules (2,065), the one that **approves the most people**. Ties go to the smaller EO gap, then to cutoffs closer to 475.

Chosen cutoffs: **20-34: 475 (unchanged), 35-54: 465, 55+: 490.**

| Validation set | DI | TPR gap | FPR gap | Approval rate | Default rate among approved |
|---|---|---|---|---|---|
| Before | 0.840 | 0.101 | 0.098 | 86.59% | 1.29% |
| After | 0.849 | 0.092 | 0.060 | 86.69% | 1.28% |

### B. Reweighting (pre-processing, Kamiran and Calders 2012)

The cutoff stays at 475; the model is retrained with sample weights.

- **Weights:** w(group, outcome) = P(group) x P(outcome) / P(group, outcome). In the weighted data every age band has the same default rate, so the model learns less of the link between age and default.
- **Training:** same setup as the shipped model (XGBoost, 21 features, 13 monotone constraints, `src.models.train.make_estimator`), fit on train + validation. Nothing is tuned, so validation is not held out.
- The retrained model is used only for this comparison and is not saved. The shipped model file does not change.

| Age band | Weight, repaid | Weight, defaulted |
|---|---|---|
| 20-34 | 1.049 | 0.604 |
| 35-54 | 1.018 | 0.804 |
| 55+ | 0.970 | **1.769** |

Defaulters aged 55+ are rare in the data, so they get a large weight and the model pays more attention to them.

## 4. Before and after (test set)

**Age band (target):**

| | DI | TPR gap | FPR gap | EO gap | Verdict |
|---|---|---|---|---|---|
| Before | 0.832 | 0.096 | 0.121 | 0.121 | **EO fail** |
| A. Cutoff per age band | 0.840 | 0.088 | 0.097 | 0.097 | Pass |
| **B. Reweighting** | **0.912** | **0.021** | **0.025** | **0.025** | **Pass** |

| FPR per group (approved among defaulters) | 20-34 | 35-54 | 55+ |
|---|---|---|---|
| Before | 0.149 | 0.175 | 0.269 |
| A | 0.149 | 0.188 | 0.246 |
| B | 0.191 | 0.202 | 0.178 |

| Approval rate per group | 20-34 | 35-54 | 55+ |
|---|---|---|---|
| Before | 77.4% | 83.8% | 93.0% |
| A | 77.4% | 84.8% | 92.1% |
| B | 82.0% | 85.3% | 89.9% |

**Gender (side effects):**

| | DI | EO gap | Verdict |
|---|---|---|---|
| Before | 0.995 | 0.038 | Pass |
| A | 0.993 | 0.041 | Pass |
| B | 0.992 | 0.049 | Pass |

**Performance (cost):**

| | AUC | Approval rate | Default rate among approved | Decisions changed |
|---|---|---|---|---|
| Before | 0.9492 | 86.93% | 1.46% | |
| A | 0.9492 | 86.98% | 1.47% | 192 (0.9%) |
| B | 0.9458 | 86.88% | 1.49% | 598 (2.7%) |

A does not change scores, so its AUC is the same.

## 5. Reading and choice

- **A only just passes.** Its test EO gap is 0.097, 0.003 under the limit. On validation the gap before mitigation was already 0.101, close to the limit, so the cutoffs moved only a little. The test set has about 350 defaulters aged 20-34, so their FPR moves by about +/- 0.02 from sample to sample. Another sample could fail. The default rate among approved held on validation but rose by 0.01 points on test.
- **B passes with a wide margin** (0.025). It lowers the 55+ FPR from 0.269 to 0.178 and raises approvals at 20-34, closing the gap from both sides. In exchange, approvals of young defaulters also rise (FPR 0.149 to 0.191), so the overall default rate among approved rises by 0.03 points.
- **B is adopted:** (1) it meets the limit with a wide margin, (2) it does not use a different cutoff per age at decision time, and (3) the cost is small (AUC -0.003, default rate among approved +0.03 points).
- A is kept as a comparison. "Age-specific cutoffs have a small and fragile effect" is a finding in itself.

## 6. Limitations

- **Gender is simulated** and independent of default, so passing on gender is expected. For a real-gender check see the German Credit secondary model (`reports/german_report_en.md`).
- **Age is a model feature** (no monotone constraint, protected attribute). B does not remove age; it only weakens the age-default link through the training weights.
- **A uses age directly in the decision.** Real lenders may face legal and ethical limits on this.
- **No confidence intervals.** Each EO gap is one test-set value. With about 350 defaulters aged 20-34, A's 0.097 in particular is statistically indistinguishable from 0.10.
- When the training sample changes, 26.8% of decisions near the cutoff (+/- 30 points) flip (`reports/audit_report_en.md`, section 3). B is also a retrained model, so individual decisions near the cutoff are as unstable as the shipped model's. Read these as group-level figures.
- The simulator uses the target for every row (as in the mission example), so AUC and other performance figures are optimistic.
