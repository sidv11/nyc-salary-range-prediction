# NYC Job Salary Range Prediction

🔗 **Live Demo:** [NYC Salary Range Predictor](https://nyc-salary-range-prediction.streamlit.app/)

Predictin the minimum and maximum salary for a New York City government job posting, using only the details available in the posting itself — no negotiated figures, no insider information, just what a job listing already says about the role.

## 🚀 Live Demo

Try the deployed Streamlit application:

👉 **[Open NYC Salary Range Predictor](https://nyc-salary-range-prediction.streamlit.app/)**

Enter details from an NYC government job posting and the model will estimate its annual salary range.

---

## Explained simply

When a company posts a job, they usually already have a salary range in mind before anyone applies. This project builds a model that guesses that range on its own, just by reading the posting: the job title, which agency is hiring, how senior the role is, and a few other clues.

Feed it a new posting it has never seen, and it predicts roughly what the pay range should be.

---

## Problem statement

Given the attributes of a job posting — agency, title, category, career level, and similar fields — predict both:

- `Salary Range From` — the minimum salary
- `Salary Range To` — the maximum salary

These are treated as two separate regression targets.

---

## Dataset

**Jobs NYC Postings** is a real dataset of New York City government job listings.

The dataset is not included in this repository. See [`data/README.md`](data/README.md) for why and how to obtain the same publicly available dataset.

The raw file had two data-quality problems that had to be fixed before modeling:

### Duplicate postings

2,494 of 5,120 rows shared a Job ID with another row — the same listing reposted internally and externally.

These were confirmed to be exact reposts rather than conflicting records, so only the first occurrence of each Job ID was retained.

### Mixed salary frequency

Salaries were listed as **Annual, Hourly, or Daily**, all mixed into the same two numeric columns.

Comparing a $22/hour listing directly against a $95,000/year listing would be meaningless, so every salary was annualized onto one consistent scale before modeling.

After cleaning:

- **2,615 unique postings**
- **23 rows with unusable zero salary dropped**
- All remaining salary values normalized to an annual scale

---

## Approach

### 1. Clean and normalize

- De-duplicate job postings
- Annualize mixed salary frequencies
- Remove unusable salary records
- Standardize the dataset for modeling

### 2. Explore

Analyze:

- Salary distributions
- Salary differences across agencies
- Career-level salary patterns
- Job-category differences
- Relationships between job characteristics and compensation

### 3. Engineer features

The model uses categorical job-posting information such as:

- Agency
- Job Category
- Career Level
- Level
- Civil Service Title
- Title Classification
- Posting Type
- Full-Time / Part-Time indicator

It also uses lightweight text-derived features, including:

- Job description word count
- Minimum qualification word count
- Whether preferred skills were provided
- Posting month
- Number of positions

### 4. Encode carefully

`Civil Service Title` and `Job Category` contain hundreds of distinct values, making straightforward one-hot encoding less practical.

The project therefore uses **smoothed target encoding** to represent high-cardinality categorical variables using salary information while reducing the risk of overfitting and target leakage.

### 5. Compare candidate models

Five model families were evaluated using the same held-out test set, ranging from linear regression approaches to gradient boosting models.

Both salary targets were evaluated separately.

### 6. Tune the winner

**XGBoost** performed best among the evaluated models.

The final models were tuned separately for:

- Minimum salary
- Maximum salary

Hyperparameter optimization was performed using `RandomizedSearchCV` with 5-fold cross-validation and MAE as the optimization metric.

### 7. Evaluate honestly

The final reported metrics come from the tuned models evaluated on a held-out test set that was not used during model training or hyperparameter tuning.

---

## Results

Final test-set performance from [`predictions/model_scores.csv`](predictions/model_scores.csv):

| Target | R² | MAE | RMSE |
|---|---:|---:|---:|
| Salary Range From (minimum) | 0.653 | $8,156 | $15,208 |
| Salary Range To (maximum) | 0.799 | $11,443 | $18,801 |

### In plain terms

Using only information available in a job posting:

- The model explains approximately **65% of the variation in minimum salary**.
- The model explains approximately **80% of the variation in maximum salary**.

The maximum salary target performs better than the minimum salary target.

This is consistent with the idea that the upper end of a salary range may be more closely associated with the formal title, grade, and classification of a role, while the lower end can have additional variation that is not directly observable from the posting.

Feature importance also indicates that **title- and category-related information is highly influential**, which is consistent with the structured nature of NYC civil-service compensation.

---

## 🖥️ Streamlit Application

The project includes a Streamlit web application that allows users to interact with the trained models without running the full notebook.

The application accepts information such as:

- Agency
- Posting type
- Job category
- Career level
- Civil service title
- Number of positions
- Job description
- Minimum qualifications
- Preferred skills
- Posting month

It then returns:

**Estimated minimum salary → Estimated maximum salary**

### Example

```text
Estimated annual salary range

$81,247 – $137,766

Estimated midpoint: $109,507
