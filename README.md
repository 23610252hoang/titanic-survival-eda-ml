# Titanic Survival Prediction: EDA, Feature Engineering, and Business Insight

This project predicts passenger survival on the Titanic using exploratory data analysis, feature engineering, and supervised machine learning.

The goal is not only to maximize validation accuracy, but to demonstrate a practical DA/DS/BA workflow:

1. Understand the raw data.
2. Explore relationships between passenger attributes and survival.
3. Form hypotheses from EDA.
4. Convert hypotheses into model-ready features.
5. Train baseline and improved models.
6. Evaluate on a local validation set.
7. Translate results into business-style insights.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── raw/
├── notebooks/
│   └── titanic_eda_feature_engineering_solution.ipynb
├── outputs/
│   ├── submission_eval.csv
│   └── titanic_experiment_results.csv
├── reports/
│   └── titanic_resume_vi.md
└── src/
    └── titanic_solution.py
```

## Dataset

The project uses the following files:

- `train_local (1).csv`: training data with `Survived`.
- `valid (1).csv`: validation data with `Survived`.
- `eval (1).csv`: evaluation data without `Survived`.
- `sample_submission.csv`: submission format sample.

For privacy and course-material safety, raw CSV files are not committed by default. Place them in `data/raw/` if you want to run the project locally.

Expected local paths:

```text
data/raw/train_local (1).csv
data/raw/valid (1).csv
data/raw/eval (1).csv
data/raw/sample_submission.csv
```

The script also supports the original local `Downloads` paths used during development.

## Key EDA Findings

- `Sex` is strongly related to survival. Female passengers had a much higher survival rate than male passengers.
- `Pclass` is important. First-class passengers had a higher survival rate than second- and third-class passengers.
- `Age` has missing values, but it may capture child/adult survival differences.
- `Cabin` is highly missing, but whether cabin information exists can still be useful.
- `Name` contains titles such as `Mr`, `Mrs`, `Miss`, and `Master`, which encode gender, age, and social status signals.

## Feature Engineering

Created features:

- `FamilySize = SibSp + Parch + 1`
- `IsAlone`
- `Title`, extracted from `Name`
- `HasCabin`
- `Deck`, extracted from the first letter of `Cabin`

These features were designed from EDA hypotheses rather than added randomly.

## Model Results

Validation accuracy on `valid.csv`:

| Experiment | Accuracy |
| --- | ---: |
| baseline_logreg | 0.805970 |
| baseline_rf | 0.820896 |
| feature_gb | 0.850746 |
| feature_logreg | 0.873134 |
| feature_rf | 0.873134 |

Best validation accuracy:

```text
0.873134
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python src/titanic_solution.py
```

Outputs will be written to:

```text
outputs/titanic_experiment_results.csv
outputs/submission_eval.csv
```

## Business Analysis Perspective

This project is a small but useful example of business analysis thinking:

- Identify variables associated with an outcome.
- Explain why those variables matter.
- Create interpretable features.
- Compare a baseline with improved versions.
- Report the result in a way that supports decision-making.

In a real business setting, the same workflow can be applied to churn prediction, customer segmentation, lead scoring, campaign targeting, and risk analysis.

## Next Improvements

- Add cross-validation.
- Tune model hyperparameters.
- Test additional models such as XGBoost or LightGBM.
- Create `AgeGroup` and `FareGroup`.
- Analyze feature importance for better explainability.
