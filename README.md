# Credit Risk Scorecard and Loan Portfolio Analytics

An end-to-end credit risk project: a points-based **scorecard** (WoE + logistic regression), an **XGBoost challenger**, **SHAP** explanations, an **expected-loss and cutoff-optimisation** layer, and a **Streamlit** app that scores a single applicant.

**Live app:** _add your Streamlit link here_

## Business problem

A lender earns interest on good loans and loses money on defaults. Given an applicant's profile, estimate the probability of default (PD), turn it into a score, and choose an approval cutoff that balances approval rate against losses.

## Data

Kaggle *Give Me Some Credit* (2011 competition data): 150,000 borrowers, 10 predictors, target `SeriousDlqin2yrs` (90+ days late within 2 years). The default rate is 6.68%, so accuracy is a misleading metric and the model is judged on AUC, Gini and KS.

## Method

1. **Data quality checks.** Found and handled:
   - 1 row with age 0 (removed)
   - 19.8% of `MonthlyIncome` missing and 3,924 missing `NumberOfDependents`
   - 269 rows holding special codes (96/98) in the three "days late" columns; these rows default at 54.6% against 6.6% for everyone else, so they are flagged (`special_code`) rather than deleted
   - `DebtRatio` is not a real ratio when income is missing or zero (median 1,159 against 0.30 when income is known)
   - extreme outliers, capped at the training 99th percentile
2. **Split first, clean second.** 70/30 stratified train/test split. Every cleaning rule, bin edge and WoE value is learned from the training set only, to avoid data leakage.
3. **Feature selection with Information Value.** Utilisation, 90+ days late, 30-59 days late and 60-89 days late are the strongest predictors; age is medium.
4. **Scorecard.** WoE-transformed features, logistic regression, scaled to a score (600 points at 50:1 good:bad odds, 20 points to double the odds).
5. **Challenger.** XGBoost (300 shallow trees) on the cleaned features.
6. **Explainability.** SHAP values for global importance and for individual decisions.
7. **Finance layer.** Expected loss = PD x LGD x EAD, and a profit-versus-cutoff analysis.

## Results (held-out 30% test set, 45,000 borrowers)

| Model | AUC | Gini | KS |
|---|---|---|---|
| Logistic regression scorecard | 0.858 | 0.717 | 0.562 |
| XGBoost challenger | 0.867 | 0.734 | 0.576 |

Train and test scores are almost identical for both models, so neither is overfitting noticeably. XGBoost gains only about 0.01 AUC, so the explainable scorecard is the natural choice for deployment, with XGBoost kept as a challenger.

**Risk bands (scorecard, test set):**

| Band | Score | Share of borrowers | Actual bad rate |
|---|---|---|---|
| A Very Low | 610+ | 23.3% | 0.6% |
| B Low | 590-610 | 29.6% | 1.6% |
| C Medium | 570-590 | 18.2% | 4.0% |
| D High | 540-570 | 18.8% | 9.1% |
| E Very High | below 540 | 10.1% | 36.1% |

Average predicted PD is close to the actual bad rate in every band (slightly conservative in the safest bands), so the probabilities are reasonably well calibrated.

## Finance layer (assumptions, not facts from the data)

The dataset has no loan amounts, rates or recoveries, so these are illustrative assumptions:

| Assumption | Value |
|---|---|
| EAD | Rs 1,00,000 per loan |
| LGD | 60% (unsecured) |
| Margin on a good loan (interest minus funding and operating cost) | Rs 10,000 |

- Portfolio expected loss on the 45,000 test applicants: **Rs 17.8 crore**; band E alone accounts for Rs 9.5 crore.
- A loan breaks even when the bad rate is 10,000 / (10,000 + 60,000) = 14.3%.
- **Profit-maximising cutoff: 540.** It approves 89.9% of applicants with a 3.4% bad rate, against 6.7% if everyone is approved (about half the default rate). Simulated profit is Rs 30.9 crore, about 29% higher than approving everyone (Rs 23.9 crore) and far above a very strict cutoff of 600 (Rs 16.9 crore).

## SHAP findings

Credit utilisation is the strongest driver, followed by 30-59 days late and 90+ days late, then age. Higher utilisation and any serious delinquency push predicted risk up; older age pushes it down. The ranking agrees with the Information Value ranking.

## Limitations

- Old (2011) public competition data; results would need revalidation on current data.
- Loan amount, interest, LGD and margin are assumptions; profit figures are simulated and depend on them.
- Only approved applicants are in the data (no reject inference).
- Age is a strong predictor here, but using it directly in real lending can raise fairness and regulatory questions.
- Model performance is measured on one random split; no out-of-time validation.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Needs `scorecard_artifacts.joblib` (saved by the notebook) in the same folder as `app.py`.

## Files

- `notebook.ipynb` - full analysis
- `app.py` - Streamlit scoring app
- `scorecard_artifacts.joblib` - trained models, cleaning rules and WoE tables
- `requirements.txt` - dependencies
