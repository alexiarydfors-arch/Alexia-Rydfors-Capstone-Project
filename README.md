# Does Money Win Elections? Predicting Congressional Election Outcomes from Campaign Finance Data

**Alexia Rydfors**

---

## 1. Problem Statement

Campaign finance is one of the most debated topics in American democracy, yet most discussions rely on anecdote rather than data. This project addresses a clear, testable question: **Can the amount and source of campaign finance contributions predict whether a U.S. congressional candidate wins or loses their election?**

**Goals:**
- Build a classification model that predicts binary win/loss outcomes for U.S. House candidates using campaign finance variables
- Identify which financial factors (total raised, PAC money, individual donations, spending) are most predictive of electoral success
- Provide a data-driven foundation for evaluating campaign finance reform proposals

**Challenges:**
- FEC and MIT datasets use different candidate name formats, requiring an alternative merge strategy (state + district + party)
- Campaign finance data is heavily right-skewed — a small number of candidates raise orders of magnitude more than the median
- Financial variables are highly correlated with each other, requiring careful feature engineering

**Potential Benefits:**
- Gives voters, journalists, and policymakers a quantitative answer to how much money drives congressional election outcomes
- Identifies which financial factors matter most, helping reformers prioritize what to target
- Demonstrates that predictive modeling can surface patterns in public data that anecdotal analysis misses

---

## 2. Model Outcomes and Predictions

**Type of learning:** Supervised classification

**Output:** Binary prediction — 1 (candidate wins) or 0 (candidate loses)

**Algorithms used:**
- Logistic Regression — interpretable linear baseline
- Decision Tree — fully interpretable single-tree model
- Random Forest — ensemble of decision trees
- Gradient Boosting — sequential boosted tree ensemble

All four models are supervised classification algorithms trained on labeled data (historical election outcomes). No unsupervised learning is used, as we have clear ground-truth win/loss labels for every race.

**Best model:** Gradient Boosting Classifier
- **Accuracy:** 0.88
- **ROC-AUC:** 0.94
- **F1-Score:** 0.89
- **Recall:** 0.95

---

## 3. Data Acquisition

This project draws on two publicly available datasets from authoritative sources:

### Federal Election Commission (FEC) — Candidate Summary File
- **Source:** https://www.fec.gov/data/browse-data/?tab=bulk-data
- **Why this data:** The FEC is the official U.S. government repository for campaign finance data. The candidate summary file provides comprehensive financial totals — receipts, disbursements, PAC contributions, individual contributions, and loans — for every registered federal candidate in a given election cycle.
- **Download:** Scroll to "Candidate summary files" → download `weball22.zip` → unzip → rename to `fec_candidates.csv` → place in `data/` folder

### MIT Election Data and Science Lab (MEDSL) — U.S. House Results
- **Source:** https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2023/2023-11-07/house.csv
- **Why this data:** MEDSL is the leading academic source for historical U.S. election results. This file contains district-level vote totals and winner/loser outcomes for every House general election from 1976 to 2022, enabling us to label each candidate as a winner or loser.
- **Download:** Open the URL above → save as `mit_results.csv` → place in `data/` folder

**Merge strategy:** Because FEC stores names as "LAST, FIRST" and MIT stores names as "FIRST LAST," direct name matching fails. Instead, datasets are merged on **state + congressional district + party abbreviation**, producing one financial record per party per district matched to the corresponding candidate's vote totals.

**Visualization of data potential:** Exploratory analysis confirms strong separation between winners and losers on key financial variables — particularly total funds raised — providing clear evidence that the data has the potential to solve the classification problem. See Notebook 1 for full EDA.

---

## 4. Data Preprocessing and Preparation

### Handling Missing Values and Inconsistencies
- Derived election cycle year from the FEC file's `Coverage_End_Date` column and URL metadata
- Standardized party abbreviations across datasets (e.g., "DEMOCRAT" → "DEM")
- Filled ratio features (`pac_share`, `indiv_share`, `raise_spend_ratio`) with 0 where denominators were zero
- Applied `SimpleImputer(strategy='median')` as the first step in every model pipeline as a safety net for any remaining NaN values
- Removed candidate-party combinations with zero reported fundraising (likely data artifacts)

### Encoding
- Party affiliation encoded as binary flags: `is_democrat` (1 = Democrat) and `is_republican` (1 = Republican)
- Monetary amounts log-transformed (`log_raised`, `log_spent`) to handle extreme right skew
- All ratio features bounded between 0 and 1 (proportion of total raised)

### Train/Test Split
- Data split **75% training / 25% test** using `train_test_split` with `stratify=y` to preserve the win/loss ratio in both sets
- Final dataset: 863 rows (after filtering), ~647 training rows, ~216 test rows

---

## 5. Modeling

Four supervised classification algorithms were selected to address the binary win/loss prediction problem:

**Logistic Regression** — chosen as the interpretable linear baseline. Allows direct examination of coefficient signs and magnitudes to understand the direction of each feature's effect on win probability. Tuned regularization strength (`C`).

**Decision Tree** — chosen for full interpretability. A single tree can be visualized and explained to any audience. Tuned maximum depth and minimum samples per split to control overfitting.

**Random Forest** — chosen as a robust ensemble method. By averaging across hundreds of trees trained on random subsets of data and features, it reduces variance and typically outperforms a single tree. Tuned number of estimators, max depth, and max features.

**Gradient Boosting** — chosen as the most powerful tree-based method for structured tabular data. Builds trees sequentially, each correcting the errors of the previous. Tuned learning rate, number of estimators, max depth, and subsampling rate.

All models were implemented as `sklearn` pipelines with a `SimpleImputer` and (where applicable) `StandardScaler` as preprocessing steps, ensuring no data leakage between training and test sets. Hyperparameters were tuned using `RandomizedSearchCV` with `StratifiedKFold(n_splits=5)` cross-validation.

---

## 6. Model Evaluation

**Models considered:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting — all supervised binary classification models.

**Evaluation metrics:**
- **ROC-AUC (primary):** Measures how well the model separates winners from losers across all classification thresholds. A score of 0.5 is no better than chance; 1.0 is perfect. Chosen as primary metric because it is robust to class balance and captures the model's ability to rank candidates by win probability regardless of the specific decision threshold.
- **Accuracy (secondary):** Proportion of correctly classified candidates. Appropriate here because the dataset is approximately balanced (~50% winners, ~50% losers), making accuracy a meaningful and interpretable metric.
- **F1-Score and Recall** also reported to characterize error patterns.

**Results:**

| Model | Accuracy | ROC-AUC | F1 | Recall |
|---|---|---|---|---|
| Gradient Boosting | **0.88** | **0.94** | **0.89** | **0.95** |
| Random Forest | 2nd | 2nd | 2nd | 2nd |
| Logistic Regression | 3rd | 3rd | 3rd | 3rd |
| Decision Tree | 4th | 4th | 4th | 4th |

**Why Gradient Boosting is most optimal:** It achieves the highest AUC (0.94) and accuracy (0.88) of all four models. The sequential boosting approach allows it to capture non-linear interactions between features — for example, the combined effect of high PAC share and high total raised — that logistic regression's linear structure cannot model. Compared to a single decision tree, it dramatically reduces variance by aggregating across many trees. The result is a model that correctly identifies 88% of winners and losers from financial data alone, with an AUC that indicates near-excellent class separation.

---

## Project Structure

```
├── data/
│   ├── .gitkeep                   <- placeholder (data files not tracked by Git)
│   ├── fec_candidates.csv         <- download separately (see Section 3)
│   ├── mit_results.csv            <- download separately (see Section 3)
│   ├── campaign_finance_clean.csv <- created by Notebook 1 when you run it
│   └── test_predictions.csv       <- created by Notebook 2 when you run it
├── 1__Campaign_Finance_EDA.ipynb
├── 2__Campaign_Finance_Modeling.ipynb
├── 3__Campaign_Finance_Evaluation.ipynb
└── README.md
```

**Run notebooks in order: 1 → 2 → 3.**

---

## Contact

Questions about this project? Reach out via LinkedIn or email.
