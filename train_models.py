"""
Train and compare classifiers; persist artifacts for the API and UI.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ROOT / "models"
EDA_DIR = ARTIFACTS / "eda"

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {
        "target": "target",
        "heartdisease": "target",
        "heart_disease": "target",
        "num": "target",
        "condition": "target",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    return df


def _find_target_column(df: pd.DataFrame) -> str:
    for c in ("target", "num", "heartdisease", "heart_disease"):
        if c in df.columns:
            return c
    return df.columns[-1]


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/fetch_data.py")
    raw = pd.read_csv(path)
    df = _normalize_columns(raw)
    y_col = _find_target_column(df)
    X = df.drop(columns=[y_col])
    X.columns = [str(c).strip().lower() for c in X.columns]
    for c in FEATURE_NAMES:
        if c not in X.columns:
            X[c] = np.nan
    X = X[FEATURE_NAMES]
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = df[y_col]
    y = pd.to_numeric(y, errors="coerce")
    ymax = float(np.nanmax(y.to_numpy()))
    if ymax > 1:
        y = (y > 0).astype(int)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask]
    y = y.astype(int)
    return X, y


def evaluate_model(name: str, estimator, X_test, y_test) -> dict:
    y_pred = estimator.predict(X_test)
    proba = None
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if proba is not None and len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    else:
        metrics["roc_auc"] = None
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return metrics


def plot_eda(X: pd.DataFrame, y: pd.Series) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    data = X.copy()
    data["target"] = y.values
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="vlag", center=0)
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_heatmap.png", dpi=120)
    plt.close()
    plt.figure(figsize=(8, 5))
    vc = y.value_counts().sort_index()
    plt.bar(vc.index.astype(str), vc.values, color=["#4c78a8", "#f58518"])
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.title("Target distribution (0 = no disease, 1 = disease)")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "target_distribution.png", dpi=120)
    plt.close()


def _json_default(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(repr(obj))


def feature_importance_payload(models: dict, feature_names: list[str]) -> dict:
    out: dict = {}
    if "random_forest" in models:
        rf = models["random_forest"]
        if hasattr(rf, "feature_importances_"):
            out["random_forest"] = dict(
                sorted(zip(feature_names, rf.feature_importances_.tolist()), key=lambda x: -x[1])
            )
    if "decision_tree" in models:
        dt = models["decision_tree"]
        if hasattr(dt, "feature_importances_"):
            out["decision_tree"] = dict(
                sorted(zip(feature_names, dt.feature_importances_.tolist()), key=lambda x: -x[1])
            )
    if "logistic_regression" in models:
        lr = models["logistic_regression"]
        if hasattr(lr, "coef_"):
            coef = lr.coef_.ravel()
            out["logistic_regression"] = dict(
                sorted(zip(feature_names, np.abs(coef).tolist()), key=lambda x: -x[1])
            )
    return out


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset(DATA_PATH)
    plot_eda(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)
    X_test_s = scaler.transform(X_test_i)

    joblib.dump(imputer, ARTIFACTS / "imputer.joblib")
    joblib.dump(scaler, ARTIFACTS / "scaler.joblib")
    (ARTIFACTS / "feature_order.json").write_text(json.dumps(FEATURE_NAMES), encoding="utf-8")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grids = {
        "logistic_regression": GridSearchCV(
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"C": [0.05, 0.1, 0.5, 1.0, 2.0], "solver": ["lbfgs"]},
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        ),
        "decision_tree": GridSearchCV(
            DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            {"max_depth": [3, 5, 7, 10], "min_samples_leaf": [1, 2, 4]},
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        ),
        "random_forest": GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {"n_estimators": [100, 200], "max_depth": [None, 8, 12], "min_samples_leaf": [1, 2]},
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        ),
        "mlp_classifier": GridSearchCV(
            MLPClassifier(random_state=42, max_iter=800),
            {
                "hidden_layer_sizes": [(32,), (64,), (32, 16)],
                "alpha": [1e-4, 1e-3],
                "learning_rate_init": [1e-3, 3e-3],
            },
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        ),
    }

    fitted: dict = {}
    for key, grid in grids.items():
        grid.fit(X_train_s, y_train)
        fitted[key] = grid.best_estimator_

    joblib.dump(fitted["logistic_regression"], MODEL_DIR / "logistic_regression.joblib")
    joblib.dump(fitted["decision_tree"], MODEL_DIR / "decision_tree.joblib")
    joblib.dump(fitted["random_forest"], MODEL_DIR / "random_forest.joblib")
    joblib.dump(fitted["mlp_classifier"], MODEL_DIR / "mlp_classifier.joblib")

    metrics_all = {}
    for name, est in fitted.items():
        metrics_all[name] = evaluate_model(name, est, X_test_s, y_test)

    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(metrics_all, indent=2, default=_json_default),
        encoding="utf-8",
    )
    fi = feature_importance_payload(
        {
            "logistic_regression": fitted["logistic_regression"],
            "decision_tree": fitted["decision_tree"],
            "random_forest": fitted["random_forest"],
        },
        FEATURE_NAMES,
    )
    (ARTIFACTS / "feature_importance.json").write_text(json.dumps(fi, indent=2), encoding="utf-8")

    if hasattr(fitted["logistic_regression"], "predict_proba"):
        proba = fitted["logistic_regression"].predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"LR AUC={auc(fpr, tpr):.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve (Logistic Regression, hold-out)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(EDA_DIR / "roc_logistic_regression.png", dpi=120)
        plt.close()

    print("Training complete.")
    for k, v in metrics_all.items():
        print(k, "accuracy:", round(v["accuracy"], 4), "roc_auc:", v.get("roc_auc"))


if __name__ == "__main__":
    main()
