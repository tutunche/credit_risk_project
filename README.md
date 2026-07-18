# Credit Risk Modeling: Home Credit Default Risk

This project builds a credit default prediction pipeline using the Home Credit Default Risk dataset. The goal is to estimate probability of default for loan applicants using application data, bureau history, previous applications, POS cash behavior, credit card balances, and installment repayment history.

## Why This Project Matters

Credit risk models are used by banks and lenders to estimate repayment risk, support underwriting, and prioritize manual review. This project focuses on three banking ML themes:

- Feature engineering from relational customer credit history
- Model benchmarking across interpretable and high-performing models
- Explainability for model risk review

## Dataset

Source: Kaggle Home Credit Default Risk  
Target: `TARGET`, where `1` means client had payment difficulty and `0` means no payment difficulty.

The training set is highly imbalanced:

| Class | Meaning | Share |
|---|---|---:|
| 0 | No default/payment difficulty | ~91.9% |
| 1 | Default/payment difficulty | ~8.1% |

## Modeling Results

| Model | Feature Set | Validation Method | ROC AUC |
|---|---|---|---:|
| Logistic Regression (elastic net, grid-searched) | Engineered multi-table features | Holdout | 0.7784 |
| XGBoost | Application only | Holdout | 0.7493 |
| LightGBM | Engineered multi-table features | 5-fold CV | 0.7883 +/- 0.0045 |
| Neural Network | Engineered multi-table features | Holdout | 0.7787 |
| Tuned XGBoost | Engineered multi-table features | Holdout | 0.7938 |

## Key Feature Groups

- Application affordability: credit-to-income, annuity-to-income, credit term
- External risk scores: `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`
- Bureau behavior: active credit count, debt-to-credit ratio, overdue history
- Previous applications: approved/refused loan history, application-to-credit ratios
- Credit card behavior: utilization and payment coverage
- Installments: late payment and adequate payment indicators

## Reproduce

```bash

pip install -r requirements.txt
bash scripts/download_data.sh
jupyter lab