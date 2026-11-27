# ANOVA statistical test report

> Educational use only. Not for real financial decisions. The alternative data is simulated, and the simulator uses the target for every row, so AUCs that use alternative data are optimistic.

- Code: `src/statistics/anova.py` (one-way ANOVA, eta squared, Tukey HSD, bonus two-way ANOVA). Tests: `tests/test_reasons_and_anova.py`, `tests/test_api_endpoints.py`.
- Test definitions: `statistics.anova` in `configs/config.yaml` (group column, filter, labels) and `statistics.two_way_anova` (bonus). Significance level alpha = 0.05; eta squared bands 0.01 / 0.06 / 0.14 (small / medium / large).
- Input: `reports/cv_fold_auc.csv` (described in `reports/anova_handoff_en.md`). API: `GET /anova`. Dashboard: "통계 검증" (statistics) tab.

## Summary

| Test | Question | F | p | eta squared | Conclusion |
|---|---|---|---|---|---|
| 1. By feature group | Does alternative data change AUC? | 17,086 | < 0.001 | **0.981 (large)** | All three groups differ: combined > alternative > traditional |
| 2. By algorithm (combined features) | Which algorithm is better? | 11.8 | < 0.001 | 0.096 (medium) | Both tree models beat LR by 0.002; XGBoost and LightGBM do not differ |
| 3. Bonus: segment x feature set (interaction) | Is alternative data worth more to thin-filers? | 9.08 | 0.003 | partial 0.086 (medium) | Yes: thin-filer lift +0.090, general +0.082; the paired test also gives p < 0.001 |

Tests 1 and 2 have p < 0.05, so Tukey HSD post-hoc tests were run for both. Test 3 has only two levels per factor, so it needs no post-hoc test (a significant effect means the two levels differ).

## 1. Method

**ANOVA (analysis of variance)** tests whether the means of three or more groups differ by more than chance.

| Value | Meaning |
|---|---|
| F statistic | Difference between groups / spread within groups. Larger means a clearer difference |
| p-value | Chance of a difference this large if there were no real difference. Below 0.05 = significant |
| Eta squared | Share of the total variation explained by the groups: how *large* the difference is |
| Tukey HSD | After a significant ANOVA, shows which pairs of groups differ, corrected for making several comparisons |

The p-value answers "is there a difference"; eta squared answers "how big is it". With many samples even a tiny difference reaches p < 0.05, so the two must be read together.

**Data:** per-fold AUCs from 5 repeats of stratified 5-fold cross-validation inside the train split. 27 runs = 3 algorithms (LR, XGBoost, LightGBM) x 3 feature groups (traditional only, alternative only, combined) x 3 imbalance options (none, class_weight, SMOTE). Each run has 25 AUCs (5 repeats x 5 folds), and every run uses the same folds. Both tests use all validation rows (`segment = all`).

## 2. Test 1: AUC by feature group

**Design:** groups = feature groups. Each group has 3 algorithms x 3 imbalance options x 25 = 225 AUCs.

| Feature group | Count | Mean AUC | SD |
|---|---|---|---|
| Traditional only (GMSC variables) | 225 | 0.8572 | 0.0069 |
| Alternative only (alternative variables) | 225 | 0.9124 | 0.0038 |
| Combined (both) | 225 | **0.9408** | 0.0031 |

**ANOVA:** F = 17,086, p < 0.001, eta squared = 0.981 (large).

**Tukey HSD:**

| Comparison | Mean difference | 95% interval | Adjusted p | Significant |
|---|---|---|---|---|
| Combined - traditional | +0.0836 | 0.0825 to 0.0847 | < 0.001 | Yes |
| Combined - alternative | +0.0284 | 0.0273 to 0.0294 | < 0.001 | Yes |
| Alternative - traditional | +0.0552 | 0.0542 to 0.0563 | < 0.001 | Yes |

## 3. Test 2: AUC by algorithm (combined features)

**Design:** groups = algorithms. **Only combined-feature runs** are used. Each group has 3 imbalance options x 25 = 75 AUCs.

**Why combined features only:** mixing feature groups puts AUCs from 0.86 (traditional) to 0.94 (combined) inside each algorithm group. That large spread counts as "spread within groups" and hides the small differences between algorithms. Tested with feature groups mixed, the result is F = 0.16, p = 0.85, eta squared = 0.0005, "no difference". As the table below shows, within one feature group the algorithm differences are consistent, so that conclusion is an artifact of the design. The final model uses combined features, so comparing on combined features answers "which algorithm should the final model use".

| Mean AUC | Traditional only | Alternative only | Combined |
|---|---|---|---|
| LR | 0.853 | 0.914 | 0.939 |
| XGBoost | 0.859 | 0.912 | 0.942 |
| LightGBM | 0.859 | 0.911 | 0.941 |

| Algorithm | Count | Mean AUC (combined) |
|---|---|---|
| LogisticRegression | 75 | 0.9394 |
| XGBoost | 75 | **0.9415** |
| LightGBM | 75 | 0.9414 |

**ANOVA:** F = 11.8, p = 0.00001, eta squared = 0.096 (medium).

**Tukey HSD:**

| Comparison | Mean difference | 95% interval | Adjusted p | Significant |
|---|---|---|---|---|
| XGBoost - LR | +0.0021 | 0.0009 to 0.0032 | < 0.001 | Yes |
| LightGBM - LR | +0.0020 | 0.0008 to 0.0031 | < 0.001 | Yes |
| XGBoost - LightGBM | +0.0001 | -0.0010 to 0.0012 | 0.971 | No |

## 4. Interpretation

**Test 1: the effect of alternative data is large and clear.** Adding alternative variables to traditional ones raises AUC by 0.084 on average (0.857 to 0.941). Eta squared = 0.98 means 98% of the AUC variation is explained by the feature group. Which data you use matters far more than the algorithm or imbalance option. Alternative variables alone beat traditional variables alone by 0.055. Combined beats alternative-only by a further 0.028, so the two kinds of data complement rather than replace each other.

This size is optimistic. The simulator builds the alternative variables from the real default outcome (as in the mission example), so each mission variable alone reaches AUC 0.80 to 0.84. The realistic reference is traditional-only at 0.857. This test confirms that the alternative data adds information as the simulation was designed to; it does not show the effect of real alternative data.

**Test 2: the algorithm difference is significant but small.** The tree models (XGBoost, LightGBM) beat LR by about 0.002 AUC, and the two tree models do not differ. Eta squared = 0.096 means the algorithm explains about 10% of the AUC variation within combined features. But 0.002 AUC is one fortieth of the test 1 effect (0.084) and is hard to tell apart in practice. The final XGBoost model was picked by the training rule (highest AUC among combined-feature runs, `configs/train.yaml`). **This test supports that choice over LR statistically, but XGBoost and LightGBM perform the same, so either would do.** Using LR would have cost little.

**p-values and effect sizes:** both tests have p < 0.001, yet their effects differ by a factor of 40. With many samples even tiny differences become significant, so decisions should rest on the mean difference and eta squared rather than on the p-value.

**A lesson in test design:** the same data can give opposite conclusions depending on what is mixed into the comparison (test 2: p = 0.85 versus p = 0.00001). Fix the factor with the large effect (here, the feature group) to judge a small effect fairly.

## 5. Bonus: segment x feature set (two-way ANOVA)

**Question:** does adding alternative data raise AUC by a different amount for thin-filers than for other customers? This checks whether alternative data helps thin-filers, who have little credit history, more.

**Design:** 2 factors = segment (thin-filer / general) x feature set (traditional only / combined). Each cell has 25 AUCs (5 repeats x 5 folds), 100 in total. Only the final setting (XGBoost, no resampling) is used. Mixing algorithms and imbalance options would put their differences into the residual and hide small effects (the same reason as in test 2). Thin-filers follow the proxy rule (`NumberOfOpenCreditLinesAndLoans <= 2` and `NumberRealEstateLoansOrLines == 0`). Each fold has about 1,610 thin-filer validation rows (210 defaults) and about 19,390 general rows (1,193 defaults).

**The 3 effects a two-way ANOVA tests:**

| Effect | Question |
|---|---|
| Feature set main effect | Whatever the segment, does adding alternative data raise AUC? |
| Segment main effect | Whatever the feature set, do thin-filers and general customers differ in AUC? |
| Interaction | Does the effect of alternative data depend on the segment? |

A one-way ANOVA over the 4 cells could only say "something differs"; it cannot split the difference by factor. Effect sizes use **partial eta squared** (the effect's sum of squares / (the effect's + the residual sum of squares)). With two factors, plain eta squared makes an effect look small when the other factor is large, so partial eta squared is the right measure for comparing effects.

**Mean AUC per cell:**

| Segment | Traditional only | Combined | Lift (combined - traditional) |
|---|---|---|---|
| Thin-filer | 0.8490 (SD 0.0102) | 0.9391 (SD 0.0072) | **+0.0901** |
| General | 0.8596 (SD 0.0054) | 0.9412 (SD 0.0031) | +0.0817 |

**Two-way ANOVA** (residual degrees of freedom 96):

| Effect | F | p | Partial eta squared |
|---|---|---|---|
| Feature set | 3,789.5 | < 0.001 | 0.975 (large) |
| Segment | 20.8 | < 0.001 | 0.178 (large) |
| **Interaction** | **9.08** | **0.003** | **0.086 (medium)** |

**Supporting check (paired interaction test):** a fold's thin-filer AUC and general AUC come from the same model, so they are not independent (section 6). For each fold we therefore computed "thin-filer lift - general lift" and ran a one-sample t-test on whether the mean of these 25 values is 0. Subtracting within a fold removes how well or badly that fold's model did. Result: mean +0.0084 (SD 0.0102), t = 4.14, p < 0.001, positive in 18 of 25 folds. The conclusion matches the ANOVA.

**Interpretation:**

- **Interaction: alternative data helps thin-filers more.** The thin-filer AUC lift (+0.090) is 0.008 larger than the general lift (+0.082), and both tests are significant. Put another way, thin-filers trail general customers by 0.011 AUC with traditional data only, but by only 0.002 with combined data. Alternative data makes up for much of the missing credit history, which is what this system is for.
- **Feature set main effect:** alternative data raises AUC by more than 0.08 in both segments. A significant interaction makes a main effect hard to state on its own. Here both lifts point the same way and differ only in size, so the summary is "alternative data helps both segments, and thin-filers a little more".
- **Segment main effect:** thin-filers have lower AUC than general customers. This fits the expectation that little credit history makes prediction harder, and most of the gap comes from the traditional-only models.
- **The interaction is small.** 0.008 is about one tenth of the feature set effect (about 0.086). And as section 4 notes, the alternative data is simulated, so real alternative data may not show the same difference.

## 6. Limitations

- **Fold AUCs are not independent samples.** The folds of a 5-fold cross-validation share most of their training data, and the 5 repeats re-split the same train data. ANOVA assumes independent samples, so the p-values are approximate and usually too small. Test 1's effect is so large that its conclusion would not change; read test 2's p-value with care.
- **Every run uses the same folds.** A repeated-measures design that pairs results by fold (for example, folds as blocks) would be more accurate; one-way ANOVA is used to match the mission requirement.
- **Test 2 mixes the three imbalance options.** Their AUC differences are 0.002 or less but add to the within-group spread. Using only the final setting (no resampling) gives F = 7.6, p = 0.001, eta squared = 0.17, the same conclusion.
- **In test 3, a fold's thin-filer AUC and general AUC come from the same model,** so the two segment values are not independent. The paired interaction test in section 5 removes this correlation and gives the same conclusion, but the folds still share training data. A mixed model with folds as a random effect would correct the main effects the same way; it was not used because the interaction, which is the effect of interest, is covered by the paired test.
- **The thin-filer cells of test 3 have larger variance.** Each fold has only about 1,610 thin-filer validation rows, so their AUC SD is about twice that of general customers. This does not fully meet ANOVA's equal-variance assumption, but the design is balanced (same sample size in every cell), so the effect on the result is small.
- All AUCs are cross-validation values inside the train split. For official performance see the test-set results (`reports/model_report_en.md`).
