from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DOWNLOADS_DIR = Path.home() / "Downloads"


def resolve_data_path(filename):
    project_path = DATA_DIR / filename
    if project_path.exists():
        return project_path

    downloads_path = DOWNLOADS_DIR / filename
    if downloads_path.exists():
        return downloads_path

    raise FileNotFoundError(
        f"Could not find {filename}. Put it in {DATA_DIR} or {DOWNLOADS_DIR}."
    )


def load_data():
    train = pd.read_csv(resolve_data_path("train_local (1).csv"))
    valid = pd.read_csv(resolve_data_path("valid (1).csv"))
    eval_df = pd.read_csv(resolve_data_path("eval (1).csv"))
    return train, valid, eval_df


def extract_title(name):
    title = name.split(",")[1].split(".")[0].strip()
    rare_titles = {
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    }
    if title in rare_titles:
        return "Rare"
    if title in {"Mlle", "Ms"}:
        return "Miss"
    if title == "Mme":
        return "Mrs"
    return title


def add_features(df):
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Title"] = df["Name"].apply(extract_title)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].fillna("Unknown").astype(str).str[0]
    df.loc[df["Deck"] == "U", "Deck"] = "Unknown"
    return df


def make_model(numeric_features, categorical_features, estimator):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def run_experiments():
    train, valid, eval_df = load_data()
    y_train = train["Survived"]
    y_valid = valid["Survived"]

    train_fe = add_features(train)
    valid_fe = add_features(valid)
    eval_fe = add_features(eval_df)

    feature_numeric = [
        "Pclass",
        "Age",
        "Fare",
        "SibSp",
        "Parch",
        "FamilySize",
        "IsAlone",
        "HasCabin",
    ]
    feature_categorical = ["Sex", "Embarked", "Title", "Deck"]

    experiments = [
        {
            "name": "baseline_logreg",
            "train": train,
            "valid": valid,
            "eval": eval_df,
            "numeric": ["Pclass", "Age", "Fare"],
            "categorical": ["Sex", "Embarked"],
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
        },
        {
            "name": "baseline_rf",
            "train": train,
            "valid": valid,
            "eval": eval_df,
            "numeric": ["Pclass", "Age", "Fare"],
            "categorical": ["Sex", "Embarked"],
            "estimator": RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=4,
                random_state=42,
            ),
        },
        {
            "name": "feature_logreg",
            "train": train_fe,
            "valid": valid_fe,
            "eval": eval_fe,
            "numeric": feature_numeric,
            "categorical": feature_categorical,
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
        },
        {
            "name": "feature_rf",
            "train": train_fe,
            "valid": valid_fe,
            "eval": eval_fe,
            "numeric": feature_numeric,
            "categorical": feature_categorical,
            "estimator": RandomForestClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=3,
                random_state=42,
            ),
        },
        {
            "name": "feature_gb",
            "train": train_fe,
            "valid": valid_fe,
            "eval": eval_fe,
            "numeric": feature_numeric,
            "categorical": feature_categorical,
            "estimator": GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.04,
                max_depth=3,
                random_state=42,
            ),
        },
    ]

    rows = []
    fitted_models = {}
    for exp in experiments:
        model = make_model(exp["numeric"], exp["categorical"], exp["estimator"])
        model.fit(exp["train"], y_train)
        valid_pred = model.predict(exp["valid"])
        acc = accuracy_score(y_valid, valid_pred)
        rows.append(
            {
                "experiment": exp["name"],
                "accuracy": acc,
                "numeric_features": ", ".join(exp["numeric"]),
                "categorical_features": ", ".join(exp["categorical"]),
            }
        )
        fitted_models[exp["name"]] = (model, exp)

    results = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    best_name = results.iloc[0]["experiment"]
    best_model, best_exp = fitted_models[best_name]

    eval_pred = best_model.predict(best_exp["eval"])
    submission = pd.DataFrame(
        {
            "PassengerId": eval_df["PassengerId"],
            "Survived": eval_pred.astype(int),
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "titanic_experiment_results.csv", index=False)
    submission.to_csv(OUTPUT_DIR / "submission_eval.csv", index=False)

    return results, submission


if __name__ == "__main__":
    results_df, submission_df = run_experiments()
    print(results_df.to_string(index=False))
    print()
    print("Created:", OUTPUT_DIR / "titanic_experiment_results.csv")
    print("Created:", OUTPUT_DIR / "submission_eval.csv")
    print("Submission rows:", len(submission_df))
