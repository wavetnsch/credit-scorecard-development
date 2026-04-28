# Credit Scorecard Development

A complete credit scorecard development pipeline built with Python, following industry-standard methodology used in commercial and development banks.

## Overview

This project implements a logistic regression-based credit scorecard from end to end:

1. **Exploratory Data Analysis** — portfolio composition, missing values, default rate analysis
2. **Weight of Evidence (WoE) & Information Value (IV)** — feature transformation and selection
3. **Scorecard Model** — logistic regression with PDO-based score scaling
4. **Model Validation** — AUC, Gini, KS statistic, and Population Stability Index (PSI)

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
├── src/
│   ├── data_generator.py   # Synthetic credit data generation
│   ├── woe_iv.py           # WoE/IV calculation and binning
│   ├── scorecard.py        # Logistic regression scorecard with PDO scaling
│   └── validation.py       # KS, Gini, AUC, PSI metrics
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_woe_iv_analysis.ipynb
│   ├── 03_scorecard_development.ipynb
│   └── 04_model_validation.ipynb
├── plots/                  # Generated visualizations
├── data/                   # Data directory (populated at runtime)
└── requirements.txt
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/wavetnsch/credit-scorecard-development.git
cd credit-scorecard-development

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/
```

## Key Results

The scorecard achieves strong discrimination on the synthetic dataset:

- **AUC** ≈ 0.78–0.82 (train/test)
- **Gini** ≈ 0.56–0.64
- **KS** ≈ 0.42–0.50
- **PSI** < 0.10 (stable population)

## Technical Stack

- **Python 3.10+**
- **pandas / numpy** — data manipulation
- **scikit-learn** — logistic regression, preprocessing
- **matplotlib / seaborn** — visualization

## References

- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
- Anderson, R. (2007). *The Credit Scoring Toolkit*. Oxford University Press.
- Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital Standards (Basel II)*.
