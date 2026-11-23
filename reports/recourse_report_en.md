# DiCE approval paths (Actionable Recourse) report: comparison by time horizon

> Educational use only. Not for real financial decisions. The alternative variables are simulated.

- Code: `src/explainer/recourse.py`. Tests: `tests/test_recourse.py`. Settings: `explainer.recourse` in `configs/config.yaml`.
- Run: `python -m src.explainer.recourse` (dev container, about 7 minutes on 16 cores). Pilot: `--limit 100`.
- Figures: `reports/recourse/summary.json`, paths per applicant in `reports/recourse/paths_{6,9,12,18,24}m.csv`. MLflow: experiment `nextstep-recourse`, run `dice_horizons`.
- Scope: the 2,941 applicants the shipped model (`models/xgboost_v1.0.joblib`) rejects on the **test set** (score below 475), 446 of them thin-filers.
- Path rule: score **495 or higher** (`recourse_target_score` in `configs/scoring.yaml`), **at most 3** changed variables, horizons of 6 / 9 / 12 / 18 / 24 months. Charter target: coverage of at least 90%.

## Summary

| Horizon | Corner bound | **DiCE coverage** | Thin-filers | Others | 90% target |
|---|---|---|---|---|---|
| 6 months | 69.6% | **69.5%** | 66.1% | 70.1% | missed |
| 9 months | 81.0% | **81.0%** | 80.5% | 81.0% | missed |
| 12 months | 87.9% | **87.9%** | 87.0% | 88.0% | missed |
| 18 months | 91.2% | **91.2%** | 91.0% | 91.2% | met |
| 24 months | 91.8% | **91.8%** | 91.7% | 91.8% | met |

- DiCE coverage = share of rejected applicants for whom DiCE found a path. Thin-filers / Others = the same share among rejected thin-filers (446) and the rest (2,495).
- Corner bound = the largest share for which a path can exist (section 2). DiCE cannot exceed it.
- **DiCE found almost every path that exists** (99.85% to 100% of the corner bound; 0 to 3 applicants missed per horizon).
- **The 90% target is met from 18 months.** 12 months (87.9%) misses it, and going from 18 to 24 months adds only 0.6 points.

## 1. Path rules

The variables that may change are the 8 model features with `dice_vary: true` in `configs/actionability.yaml`. Direction, step, bounds, months per step and difficulty come from the same file.

| Variable | Direction | Step | Months per step | Difficulty |
|---|---|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | decrease | 0.05 | 1 | 2 |
| DebtRatio | decrease | 0.05 | 3 | 3 |
| MonthlyIncome | increase | 5% | 6 | 3 |
| telecom_payment_rate | increase | 1/12 | 1 | 1 |
| utility_payment_rate | increase | 1/12 | 1 | 1 |
| telecom_tenure_months (time only) | increase | 1 | 1 | 1 |
| insurance_paid_months (time only) | increase | 1 | 1 | 2 |
| spending_consistency | increase | 5 | 1 | 2 |

- Within a horizon of h months, each variable may move at most floor(h / months per step) steps. Variables change in parallel, so a path takes as long as its slowest variable.
- Not-recommended (recurring payments, app logins and others), immutable (age and others) and cannot-decrease (past late payments) variables never change.
- Missing values, zero income and values already at their limit are not changed for that applicant.
- **Income linked to DebtRatio:** in GMSC, DebtRatio is "monthly debt payments (including living costs) / monthly income". When income rises, debt is assumed to stay the same, so the DebtRatio the model sees is recomputed as `DebtRatio x old income / new income`. Paths assume no new debt, so the effect of higher income is slightly optimistic.

## 2. Method

1. **Corner bound.** For every subset of 1 to 3 variables (92 subsets), move the chosen variables to their horizon limit and score the result. If any subset reaches 495, a path can exist. All 8 variables are monotone in the model (moving further in the allowed direction never lowers the score), so this is an exact upper bound. Applicants that fail it are not sent to DiCE.
2. **DiCE (dice-ml 0.12, genetic).** Per applicant, each variable's range is "current value to horizon limit". DiCE looks for 5 counterfactuals whose model output (score before rounding) lies in 495 to 1000. Proximity weight 0.2, sparsity weight 0.2, diversity weight 5.0, at most 500 generations. Feature distance weights are the difficulties.
3. **Changed starting population.** The stock dice-ml start scores one random point at a time until it has 50 points above the target. For low-scoring applicants this takes tens of thousands of model calls (over 70 seconds for one test applicant), and it never ends when no path exists. Only this start was changed: points are drawn from the same uniform distribution 2,000 at a time and scored together (at most 100,000), the corner points that reach the target are added, and the population is filled with below-target points if needed. The evolution steps and the loss are unchanged dice-ml code. About 0.45 seconds per applicant.
4. **Post-processing.** For each DiCE counterfactual: (a) round each change up to whole steps, (b) if more than 3 variables change, or a change is not needed for the target, drop the change whose removal keeps the highest score first, (c) find the fewest steps per remaining variable that keep 495, (d) recheck 495 with the official integer score.
5. **Choice.** The remaining path with the lowest cost. Cost = sum of difficulty x (months used / horizon).

**Validation:** every saved path was read back and rescored separately. 0 direction violations, 0 changes beyond the horizon limit, at most 3 variables, every path within its horizon, lowest rescored score 495.

## 3. Results by horizon

| Horizon | Paths | 1 / 2 / 3 variables | Blocked by the 3-variable limit | Not reachable with all 8 | Months taken (median, quartiles) | Uses a time-only variable |
|---|---|---|---|---|---|---|
| 6 months | 2,045 | 1,033 / 686 / 326 | 254 | 639 | 5 (3 to 6) | 78.4% |
| 9 months | 2,381 | 1,294 / 834 / 253 | 295 | 263 | 6 (4 to 8) | 70.5% |
| 12 months | 2,584 | 1,477 / 872 / 235 | 240 | 117 | 6 (4 to 9) | 65.1% |
| 18 months | 2,682 | 1,539 / 883 / 260 | 238 | 21 | 6 (4 to 9) | 62.6% |
| 24 months | 2,700 | 1,551 / 876 / 273 | 229 | 11 | 6 (4 to 10) | 62.5% |

Main paths at 12 months: insurance months alone 771, spending consistency alone 653, both 487, utility payment rate + insurance + spending consistency 122.

- **The largest gain is from 6 to 9 months** (+11.5 points). At 6 months, 639 applicants cannot reach 495 even with all 8 variables.
- **Longer horizons do not make paths longer.** The median path takes 5 to 6 months at every horizon; a longer horizon only gives paths to more applicants.
- **From 18 months on, the 3-variable limit is the main barrier.** At 24 months, 229 of the 241 applicants without a path would have one with 4 or more variables.
- **The share of time-only variables falls as the horizon grows** (78.4% to 62.5%). insurance_paid_months is capped at 24 months, so it has little room left after 12 months, while actionable variables (utilization, DebtRatio, income) get larger allowed changes.
- **The thin-filer gap appears only at short horizons.** At 6 months, thin-filers 66.1% vs others 70.1% (about 4 points). From 12 months it is 1 point or less.
- **Linking income and DebtRatio did not change coverage.** 87.86% at 12 months and 91.84% at 24 months equal the unlinked audit values (`reports/audit_report_en.md`, section 1). Income rises only 5% per 6 months, so the link rarely decides whether a path exists (at 12 months: blocked by the 3-variable limit 238 to 240, not reachable with all 8 119 to 117).

## 4. Notes on choosing a horizon

- With 12 months as the main figure, coverage is 87.9%, below the charter target. With 18 months it is 91.2%, above it.
- Choosing 18 months does not change the advice most applicants get: the median path still takes 6 months. What changes is whether about 100 applicants who need longer paths (2,584 at 12 months to 2,682 at 18 months) get one.
- A longer horizon is more exposed to retraining and data drift in the meantime. Whether a followed path is still approved (path robustness) was not checked here (section 5).
- The choice is a team decision. This report changes no settings.

## 5. Limitations

- **Simulator optimism:** the alternative variables use the target for every row, test set included (as in the mission example). Paths that lean on alternative variables look easier than they would be.
- **Path robustness not checked:** whether paths stay approved under retrained models was not tested. In the audit, minimal 12-month paths to 495 stayed approved under all 3 retrained models in 90.8% of cases.
- **Steps, months per step and difficulties are team assumptions** (`configs/actionability.yaml`); spending_consistency in particular needs mentor review. The cost function is a simple use of the difficulties.
- **Other variables held fixed:** new late payments or other changes during the path are not modelled.
- **DiCE results are search results.** The seed makes them reproducible with the same settings, but because the starting population was changed, paths differ from stock dice-ml. The dice-ml 0.12 mutation step always inserts the range's lower end instead of a random value (library behavior, left as is). Every final path is rechecked in post-processing.
