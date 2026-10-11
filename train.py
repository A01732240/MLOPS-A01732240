"""
Entrenamiento del modelo GANADOR de churn Telco, instrumentado con MLflow.

===============================================================================
 IMPORTANTE: ESTE SCRIPT YA NO BUSCA EL MEJOR MODELO, SOLO ENTRENA EL GANADOR
===============================================================================
La búsqueda del mejor modelo se hizo en el notebook de la semana anterior
(`churn_telco_flujo_ml.ipynb`) de forma iterativa, por rondas:

  - Ronda 0  baseline (DummyClassifier)                         AUC 0.500
  - Ronda 1  logreg, árbol y random forest con variables originales
             (el árbol sin límite sobreajustaba: F1 train 0.996 vs 0.48 test)
  - Ronda 2-3 se agregaron 7 variables nuevas            -> sin mejora real
  - Ronda 4  se quitaron variables redundantes (VIF) y no significativas
             (chi-cuadrada): subio_precio, TotalCharges, gender, PhoneService
  - Ronda 5  se ajustó el desbalance con class_weight="balanced"
             -> el recall subió de ~0.50 a ~0.80

GANADOR: Regresión logística con class_weight="balanced" y las 22 variables de
la ronda 4. Tuvo el mejor AUC (0.846) y el mejor recall (0.80), no sobreajusta
(F1 train ~ F1 test) y es el modelo más fácil de explicar (odds ratios).

Por eso aquí se eliminó todo lo que servía para comparar modelos (árbol, random
forest, rondas de variables, GridSearch, VIF). Lo único que queda es el modelo
ganador, y lo que se experimenta con MLflow son SUS hiperparámetros:

  --C              fuerza de la regularización
  --class-weight   ponderar o no la clase minoritaria (churn)
  --umbral         probabilidad a partir de la cual se predice "se va"
  --seed           semilla aleatoria (split, validación cruzada y modelo)

Con los valores por defecto se reproduce exactamente el modelo final del
notebook: accuracy 0.744 | precision 0.511 | recall 0.799 | F1 0.624 | AUC 0.846
===============================================================================

Uso
---
    python train.py                         # modelo ganador, valores por defecto
    python train.py --C 0.01                # más regularización
    python train.py --class-weight none     # sin ajustar el desbalance
    python train.py --umbral 0.35 --seed 7  # otro umbral y otra semilla
    python train.py --help                  # todas las opciones

    mlflow ui --backend-store-uri sqlite:///mlflow.db   # ver las corridas
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # guardar gráficas sin abrir ventana
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silenciar avisos de MLflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RUTA_DATOS = Path(__file__).parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
HASH_ESPERADO = "0f9de68e012bd3aed5fa7cdc9fc421af"  # MD5 del CSV original
EXPERIMENTO = "churn-telco-logreg-ganador"
OBJETIVO = "Churn"

SERVICIOS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
             "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

# 22 variables del modelo ganador (ronda 4 del notebook).
# Fuera: subio_precio y TotalCharges (VIF alto), gender y PhoneService (chi² no significativa)
VARIABLES = [
    "SeniorCitizen", "Partner", "Dependents", "tenure", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "n_servicios", "gasto_prom", "cliente_nuevo",
    "pago_auto", "tiene_proteccion", "mes_a_mes_fibra",
]


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
def fijar_semilla(seed: int) -> None:
    """Fija la semilla global. También se pasa como random_state al split, CV y modelo."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def md5_archivo(ruta: Path) -> str:
    """Hash del archivo para verificar que sean los mismos datos."""
    with open(ruta, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def cargar_datos(ruta: Path) -> tuple[pd.DataFrame, str]:
    """Lee el CSV y avisa si no es el mismo archivo del notebook."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No encuentro {ruta}. Descarga el dataset de Kaggle "
            "(blastchar/telco-customer-churn) y colócalo en data/."
        )
    hash_datos = md5_archivo(ruta)
    if hash_datos != HASH_ESPERADO:
        print(f"[aviso] El MD5 del archivo ({hash_datos}) no coincide con el del "
              f"notebook ({HASH_ESPERADO}). Los resultados pueden variar.")
    return pd.read_csv(ruta), hash_datos


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza definida en el EDA."""
    df = df.copy()
    # TotalCharges viene como texto; los 11 vacíos son clientes nuevos (tenure 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[OBJETIVO] = (df[OBJETIVO] == "Yes").astype(int)
    return df.drop(columns="customerID")


def crear_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Variables nuevas del EDA. TotalCharges solo se usa aquí para gasto_prom."""
    df = df.copy()
    df["n_servicios"] = (df[SERVICIOS] == "Yes").sum(axis=1)
    df["gasto_prom"] = np.where(df["tenure"] > 0,
                                df["TotalCharges"] / df["tenure"].replace(0, 1),
                                df["MonthlyCharges"])
    df["cliente_nuevo"] = (df["tenure"] <= 6).astype(int)
    df["pago_auto"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    df["tiene_proteccion"] = ((df["OnlineSecurity"] == "Yes") |
                              (df["TechSupport"] == "Yes")).astype(int)
    df["mes_a_mes_fibra"] = ((df["Contract"] == "Month-to-month") &
                             (df["InternetService"] == "Fiber optic")).astype(int)
    return df


def dividir(df: pd.DataFrame, test_size: float, seed: int):
    """Split train/test estratificado por Churn."""
    X, y = df[VARIABLES], df[OBJETIVO]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def construir_modelo(X_train: pd.DataFrame, C: float, class_weight: str,
                     max_iter: int, seed: int) -> Pipeline:
    """Escalado de numéricas + One-Hot de categóricas + regresión logística."""
    num = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    cat = [c for c in X_train.columns if c not in num]
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat),
    ])
    modelo = LogisticRegression(
        C=C,
        class_weight=None if class_weight == "none" else class_weight,
        max_iter=max_iter,
        random_state=seed,
    )
    return Pipeline([("prep", prep), ("modelo", modelo)])


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def evaluar(pipe: Pipeline, X_train, X_test, y_train, y_test, umbral: float,
            cv_folds: int, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Métricas en test, AUC en CV y F1 en train (para revisar sobreajuste)."""
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= umbral).astype(int)
    pred_train = (pipe.predict_proba(X_train)[:, 1] >= umbral).astype(int)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    auc_cv = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    metricas = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
        "auc_cv": auc_cv.mean(),
        "auc_cv_std": auc_cv.std(),
        "f1_train": f1_score(y_train, pred_train),
    }
    return metricas, prob, pred


def grafica_evaluacion(y_test, prob, pred, ruta: Path, titulo: str) -> None:
    """Matriz de confusión y curva ROC en una sola imagen."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=["se queda", "se va"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Matriz de confusión (test)")
    RocCurveDisplay.from_predictions(y_test, prob, ax=axes[1], name="logreg")
    axes[1].plot([0, 1], [0, 1], "k--", label="azar")
    axes[1].set_title("Curva ROC (test)")
    axes[1].legend(loc="lower right")
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110, bbox_inches="tight")
    plt.close(fig)


def odds_ratios(pipe: Pipeline) -> pd.DataFrame:
    """Coeficientes como odds ratio (>1 aumenta el riesgo de churn)."""
    nombres = (pd.Index(pipe["prep"].get_feature_names_out())
               .str.replace("num__", "").str.replace("cat__", ""))
    coef = pipe["modelo"].coef_[0]
    tabla = pd.DataFrame({"variable": nombres, "coeficiente": coef,
                          "odds_ratio": np.exp(coef)})
    return tabla.reindex(tabla["coeficiente"].abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------------------
# Argumentos de línea de comandos
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrena el modelo ganador de churn Telco (regresión logística) "
                    "y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("hiperparámetros del modelo")
    g.add_argument("--C", type=float, default=1.0,
                   help="Inverso de la regularización (menor = más regularización)")
    g.add_argument("--class-weight", choices=["balanced", "none"], default="balanced",
                   help="Ponderación de clases para atacar el desbalance (26.5%% churn)")
    g.add_argument("--umbral", type=float, default=0.5,
                   help="Probabilidad a partir de la cual se predice 'se va'")
    g.add_argument("--max-iter", type=int, default=2000,
                   help="Iteraciones máximas del optimizador")

    g = p.add_argument_group("reproducibilidad y evaluación")
    g.add_argument("--seed", type=int, default=42, help="Semilla aleatoria global")
    g.add_argument("--test-size", type=float, default=0.2, help="Proporción de test")
    g.add_argument("--cv-folds", type=int, default=5, help="Folds de validación cruzada")

    g = p.add_argument_group("MLflow y datos")
    g.add_argument("--data", type=Path, default=RUTA_DATOS, help="Ruta al CSV crudo")
    g.add_argument("--experiment-name", default=EXPERIMENTO, help="Experimento de MLflow")
    g.add_argument("--tracking-uri", default="sqlite:///mlflow.db", help="Backend de MLflow")
    g.add_argument("--run-name", default=None,
                   help="Nombre de la corrida (por defecto se arma con los hiperparámetros)")
    return p.parse_args(argv)


def nombre_run(args: argparse.Namespace) -> str:
    """Nombre de la corrida a partir de sus hiperparámetros, ej. logreg_C1_cw-balanced_u0.5_s42."""
    if args.run_name:
        return args.run_name
    return f"logreg_C{args.C:g}_cw-{args.class_weight}_u{args.umbral:g}_s{args.seed}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> dict:
    args = parse_args(argv)
    fijar_semilla(args.seed)

    # Datos
    df_raw, hash_datos = cargar_datos(args.data)
    df = crear_variables(limpiar(df_raw))
    X_train, X_test, y_train, y_test = dividir(df, args.test_size, args.seed)

    # Experimento de MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=nombre_run(args)) as run:
        # Parámetros
        hiperparametros = {
            "modelo": "LogisticRegression",
            "C": args.C,
            "class_weight": args.class_weight,
            "umbral": args.umbral,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "test_size": args.test_size,
            "cv_folds": args.cv_folds,
            "n_variables": len(VARIABLES),
        }
        for nombre, valor in hiperparametros.items():
            mlflow.log_param(nombre, valor)
        mlflow.set_tags({"hash_datos": hash_datos, "dataset": "telco-customer-churn",
                         "origen": "modelo ganador del notebook (ronda 5)"})

        # Entrenamiento
        pipe = construir_modelo(X_train, args.C, args.class_weight, args.max_iter, args.seed)
        pipe.fit(X_train, y_train)

        # Métricas
        metricas, prob, pred = evaluar(pipe, X_train, X_test, y_train, y_test,
                                       args.umbral, args.cv_folds, args.seed)
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)
        # hash de las predicciones, para verificar reproducibilidad entre corridas
        huella = hashlib.md5(np.round(prob, 10).tobytes()).hexdigest()
        mlflow.set_tag("hash_predicciones", huella)

        # Artefactos
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            grafica_evaluacion(y_test, prob, pred, tmp / "evaluacion.png", nombre_run(args))
            (tmp / "metricas.json").write_text(json.dumps(
                {"run_id": run.info.run_id, "hiperparametros": hiperparametros,
                 "metricas": metricas, "hash_datos": hash_datos,
                 "hash_predicciones": huella}, indent=2))
            (tmp / "reporte_clasificacion.txt").write_text(classification_report(
                y_test, pred, target_names=["se queda", "se va"], digits=3))
            odds_ratios(pipe).to_csv(tmp / "odds_ratios.csv", index=False)
            for archivo in sorted(tmp.iterdir()):
                mlflow.log_artifact(str(archivo))

        # Modelo
        firma = infer_signature(X_test, pipe.predict_proba(X_test)[:, 1])
        mlflow.sklearn.log_model(pipe, name="modelo", signature=firma,
                                 input_example=X_test.head(3))

        # Resumen en terminal
        print(f"\nCorrida: {run.info.run_name} ({run.info.run_id})")
        print(f"Experimento: {args.experiment_name}")
        for nombre, valor in metricas.items():
            print(f"  {nombre:11s} {valor:.4f}")
        print(f"  hash_predicciones {huella}")
    return metricas


if __name__ == "__main__":
    main()
