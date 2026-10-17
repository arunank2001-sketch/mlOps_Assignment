import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor

RANDOM_STATE = 42
SCORING = ["accuracy", "precision", "recall", "roc_auc"]


def load_data():
    df = pd.read_csv("data/raw/heart.csv")
    y = (df["target"] > 0).astype(int)  # collapse severity 0-4 to binary
    X = df.drop(columns=["target"])
    return X, y


def evaluate(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=SCORING)
    return {
        metric: (scores[f"test_{metric}"].mean(), scores[f"test_{metric}"].std())
        for metric in SCORING
    }


def print_results(name, results):
    print(f"\n{name}")
    for metric, (mean, std) in results.items():
        print(f"  {metric:<10} {mean:.3f} ± {std:.3f}")


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

RF_GRID = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [None, 4, 8],
    "clf__min_samples_leaf": [1, 3],
}


def build_random_forest():
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    return GridSearchCV(
        pipe,
        RF_GRID,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
    )

import mlflow
import mlflow.sklearn

mlflow.set_experiment("heart-disease-risk")


def log_run(run_name, model, params, X, y):
    with mlflow.start_run(run_name=run_name):
        results = evaluate(model, X, y)

        mlflow.log_params(params)
        for metric, (mean, std) in results.items():
            mlflow.log_metric(f"cv_{metric}_mean", mean)
            mlflow.log_metric(f"cv_{metric}_std", std)

        model.fit(X, y)
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
        )    
        return model

if __name__ == "__main__":
    X, y = load_data()

    baseline = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    log_run(
        "logistic_regression",
        baseline,
        {"model": "logistic_regression", "max_iter": 1000},
        X, y,
    )

    forest = build_random_forest()
    forest.fit(X, y)  # runs the grid search, sets best_params_
    log_run(
        "random_forest_tuned",
        forest,
        {"model": "random_forest", "grid_search": "3-fold", **forest.best_params_},
        X, y,
    )