# rf_meningitis_pipeline.py


import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from scipy.stats import randint
import matplotlib.pyplot as plt

#########################
# 1. Paths and settings #
#########################

DATA_PATH = Path("data/BASE DE DATOS CORREGIDA PARA EDITAR.xlsx")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42


###############################
# 2. Utility / preprocessing  #
###############################

def load_and_clean(filepath: Path) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    # Drop full duplicates
    df = df.drop_duplicates()
    # Keep valid mRs (0–6)
    df = df[df["mRs egreso"].between(0, 6)]
    # Binary target
    df["poor_outcome"] = (df["mRs egreso"] >= 5).astype(int)

    # Remove obvious identifiers / empty columns
    cols_to_drop = [
        "RUT", "mRs egreso", *[c for c in df.columns if isinstance(c, str) and c.startswith("Unnamed")]
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Drop datetime columns (not used in model)
    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
    df = df.drop(columns=date_cols, errors="ignore")

    # Impute numerical and categorical
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(["poor_outcome"])
    cat_cols = df.select_dtypes(include="object").columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna("Desconocido").astype("category")

    # One‑hot encode
    df = pd.get_dummies(df, drop_first=True)
    df.columns = df.columns.astype(str) 
    return df


def save_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

############################
# 3. Training + evaluation #
############################

def train_model(df: pd.DataFrame):
    X = df.drop(columns=["poor_outcome"])
    y = df["poor_outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    rf = RandomForestClassifier(random_state=SEED)

    param_distributions = {
        "n_estimators": randint(100, 1001),
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "criterion": ["gini", "entropy"],
        "bootstrap": [True, False],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_distributions,
        n_iter=100,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        random_state=SEED,
        verbose=0,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Test set predictions
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "AUC-ROC": roc_auc_score(y_test, y_proba),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
    }

    # Save metrics to txt
    metrics_path = RESULTS_DIR / "metrics_report.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.3f}\n")
    print(f"Metrics saved to {metrics_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    save_confusion_matrix(cm, labels=["Good", "Poor"], path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Save feature importances
    importances = (
        pd.Series(best_model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(20)
    )
    importances.to_csv(RESULTS_DIR / "top20_importances.csv", index=True)
    print("Top‑20 importances saved.")

    # Persist full cv results
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(RESULTS_DIR / "cv_results_random_search.csv", index=False)

    return metrics


if __name__ == "__main__":
    t0 = time.time()
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en {DATA_PATH}. Colóquelo antes de ejecutar."
        )
    df_clean = load_and_clean(DATA_PATH)
    metrics = train_model(df_clean)
    elapsed = time.time() - t0
    print("\nEntrenamiento completo.")
    print("Tiempo total: %.1f s" % elapsed)
    print("\nMétricas principales:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
