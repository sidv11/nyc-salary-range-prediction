# NYC Job Salary Range Prediction

Predicting the minimum and maximum salary for a New York City government job posting, using only the details available in the posting itself — no negotiated figures, no insider information, just what a job listing already says about the role.

## Explained simply

When a company posts a job, they usually already have a salary range in mind before anyone applies. This project builds a model that guesses that range on its own, just by reading the posting: the job title, which agency is hiring, how senior the role is, and a few other clues. Feed it a new posting it's never seen, and it predicts roughly what the pay range should be.

## Problem statement

Given the attributes of a job posting (agency, title, category, career level, and similar fields), predict both `Salary Range From` (the minimum) and `Salary Range To` (the maximum) as two separate regression targets.

## Dataset

"Jobs NYC Postings," a real dataset of New York City government job listings. It isn't included in this repository — see `data/README.md` for why and how to get the same, publicly available dataset yourself in a couple of minutes.

The raw file had two data-quality problems that had to be fixed before any modeling could start:

- **Duplicate postings.** 2,494 of 5,120 rows shared a Job ID with another row — the same listing reposted internally and externally. Confirmed these were exact reposts, not conflicting records, and kept only the first occurrence of each.
- **Mixed salary frequency.** Salaries were listed as Annual, Hourly, or Daily, all mixed into the same two numeric columns. Comparing a $22/hour listing directly against a $95,000/year listing would be meaningless, so every salary was annualized onto one consistent scale before anything else happened.

After cleaning: **2,615 unique postings** (down from 5,120), plus 23 rows with an unusable zero salary that were dropped rather than guessed at.

## Approach

1. **Clean and normalize** — de-duplicate, annualize mixed salary frequencies, drop unusable rows.
2. **Explore** — look at the salary distribution, and how pay varies by agency and career level, before deciding what should feed the model.
3. **Engineer features** — categorical fields (agency, category, career level, civil service title, etc.) plus a couple of cheap text-derived signals (job description length, whether preferred skills were listed) that tend to track seniority.
4. **Encode carefully** — `Civil Service Title` and `Job Category` have hundreds of distinct values each, too many for one-hot encoding to work well. Used k-fold target encoding instead, so each category is represented by the (smoothed, leakage-safe) average salary for postings like it.
5. **Compare candidate models** — five model families (Ridge regression up through gradient boosting) trained and compared on the same held-out test set, predicting both salary targets jointly.
6. **Tune the winner** — XGBoost came out ahead; tuned it separately for each target with `RandomizedSearchCV` (5-fold CV, scored on MAE).
7. **Evaluate honestly** — final numbers below are from the tuned models on data they never saw during training or tuning.

## Results

Final test-set performance (from `predictions/model_scores.csv`):

| Target | R² | MAE | RMSE |
|---|---|---|---|
| Salary Range From (minimum) | 0.653 | $8,156 | $15,208 |
| Salary Range To (maximum) | 0.799 | $11,443 | $18,801 |

In plain terms: using only what's written in a job posting — no negotiated numbers, no insider access — the model explains about 65% of the variation in the minimum salary and about 80% of the variation in the maximum salary. The gap between the two makes sense: the ceiling of a salary range tends to track a role's official grade/title more tightly, while the floor has more room for negotiation and discretion that isn't visible from posting text alone.

Feature importance confirms this lines up with how NYC civil-service pay actually works: title and category dominate, since pay is largely grade-driven rather than free-form.

## What I'd improve next

1. Try a model per agency or job family instead of one global model, since pay structures likely differ meaningfully across, say, uniformed services versus administrative roles.
2. Bring in external data the client brief explicitly allowed adding — cost-of-living by work location, or public NYC salary/headcount data from other years — to see if it improves the minimum-salary prediction specifically, since that's the weaker of the two targets.
3. Try quantile regression instead of a plain point estimate, so the output could be "a range we're confident about" rather than a single number.
4. Revisit the zero-salary rows that were dropped — with more time, it may be possible to recover some of them from related fields instead of discarding them outright.

## Repository layout

```
nyc-salary-range-prediction/
  README.md
  data/
    README.md              Why the data folder is mostly empty, and how to get it yourself
    raw/
      data_dictionary.txt    Column descriptions (not the data itself)
    processed/              (empty — populated when you run the notebook)
  notebook/
    Salary_Range_Prediction.ipynb
  models/
    salary_from_xgb.pkl      Tuned XGBoost model, minimum salary
    salary_to_xgb.pkl         Tuned XGBoost model, maximum salary
  predictions/
    model_scores.csv           Final R² / MAE / RMSE on the held-out test set
  report/
    Salary_Range_Prediction_Report.docx   Full written report with charts
```

## How to run

1. Get the dataset — see `data/README.md`.
2. Install dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`.
3. Open `notebook/Salary_Range_Prediction.ipynb` and run it top to bottom.
