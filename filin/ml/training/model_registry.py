from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_PRIORITY = [
    "LogisticRegression",
    "RandomForestClassifier",
    "HistGradientBoostingClassifier",
    "DummyClassifier",
]


def build_baseline_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "DummyClassifier": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "LogisticRegression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "HistGradientBoostingClassifier": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }
