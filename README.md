# Credit Scorecard Development

A complete credit scorecard development pipeline built with Python, following industry-standard methodology used in commercial and development banks.

## Overview

This project implements a logistic regression-based credit scorecard from end to end, trained on the **Give Me Some Credit** dataset (Kaggle, 150,000 records, 6.7% default rate):

1. **Exploratory Data Analysis** — portfolio composition, missing values, default rate analysis
2. **Weight of Evidence (WoE) & Information Value (IV)** — feature transformation and selection
3. **Scorecard Model** — logistic regression with PDO-based score scaling
4. **Model Validation** — AUC, Gini, KS statistic, and Population Stability Index (PSI)

## Dataset

**Give Me Some Credit** — Kaggle 2011 competition dataset (consumer credit bureau data)

| Feature | Description |
|---------|-------------|
| `age` | Borrower age in years |
| `annual_income` | Annual gross income (USD) |
| `credit_utilization` | Revolving credit utilization rate |
| `num_delinquencies` | Total 30/60/90-day delinquency counts |
| `num_credit_accounts` | Number of open credit lines and loans |
| `debt_to_income` | Debt ratio (monthly obligations / income) |
| `num_real_estate_loans` | Number of mortgage and real estate loans |
| `num_dependents` | Number of dependents in family |

**Target:** `SeriousDlqin2yrs` — serious delinquency within 2 years (1 = default)

## Methodology

### Weight of Evidence (WoE)
$$WoE_i = \ln\left(\frac{\text{Distribution of Events}_i}{\text{Distribution of Non-Events}_i}\right)$$

### Information Value (IV)
$$IV = \sum_{i} \left(\text{Dist Events}_i - \text{Dist Non-Events}_i\right) \times WoE_i$$

| IV Range | Predictive Power |
|----------|-----------------|
| < 0.02   | Useless         |
| 0.02–0.1 | Weak            |
| 0.1–0.3  | Medium          |
| 0.3–0.5  | Strong          |
| > 0.5    | Suspicious      |

### Score Scaling (PDO Method)
$$\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})$$

Where:
- **Factor** = PDO / ln(2)
- **Offset** = Base Score − Factor × ln(Base Odds)
- **PDO** = 20 (Points to Double the Odds) — industry standard

### Validation Metrics

| Metric | Description | Minimum Threshold |
|--------|-------------|------------------|
| AUC    | Area under ROC curve | > 0.70 |
| Gini   | 2 × AUC − 1 | > 0.40 |
| KS     | Max CDF separation | > 0.30 |
| PSI    | Population Stability Index | < 0.10 (stable) |

## Project Structure

```
credit-scorecard-development/
├── a_score_model.ipynb     # End-to-end scorecard: EDA → WoE/IV → Model → Validation
├── src/
│   ├── data_loader.py      # Give Me Some Credit data loading and cleaning
│   ├── data_generator.py   # Synthetic data (fallback / unit testing)
│   ├── woe_iv.py           # WoE/IV calculation and binning
│   ├── scorecard.py        # Logistic regression scorecard with PDO scaling
│   └── validation.py       # KS, Gini, AUC, PSI metrics
├── plots/                  # Generated visualizations
├── data/                   # Data directory (add dataset here — see below)
└── requirements.txt
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/wavetnsch/credit-scorecard-development.git
cd credit-scorecard-development

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json first
kaggle datasets download -d brycecf/give-me-some-credit-dataset -p data/
unzip data/give-me-some-credit-dataset.zip -d data/

# Open the notebook
jupyter notebook a_score_model.ipynb
```

## Key Results

| Metric | Value | Benchmark | Status |
|--------|-------|-----------|--------|
| Dataset | Give Me Some Credit (n = 150,000) | — | — |
| Default Rate | 6.68% | — | — |
| AUC-ROC | 0.8471 | > 0.70 | PASS |
| Gini Coefficient | 0.6941 | > 0.40 | PASS |
| KS Statistic | 0.5324 | > 0.30 | PASS |
| PSI (stability) | 0.0005 | < 0.10 | PASS |

## Technical Stack

- **Python 3.10+**
- **pandas / numpy** — data manipulation
- **scikit-learn** — logistic regression, preprocessing
- **matplotlib / seaborn** — visualization

## References

- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
- Anderson, R. (2007). *The Credit Scoring Toolkit*. Oxford University Press.
- Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital Standards (Basel II)*.
