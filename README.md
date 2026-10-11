# Churn Telco — experimentación reproducible con MLflow

## Overview

Predicción de cancelación de clientes (*churn*) con el dataset *Telco Customer Churn* (IBM/Kaggle). La exploración y la selección de modelo se hicieron en el notebook; el ganador fue una **regresión logística con `class_weight="balanced"`** (AUC 0.846, recall 0.80).

`train.py` entrena ese modelo, recibe los hiperparámetros por línea de comandos y registra cada corrida en MLflow (parámetros, métricas, artefactos y modelo) dentro del experimento `churn-telco-logreg-ganador`.

## Estructura

```
├── train.py                # entrenamiento + registro en MLflow
├── run_experiments.sh      # las 8 corridas del experimento
├── requirements.txt        # dependencias con versiones fijas
├── churn_telco_eda.ipynb   # exploración de datos
└── data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Ejecución

Python 3.13. Desde la raíz del repositorio:

```bash
# 1. Entorno y dependencias
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Modelo ganador (valores por defecto)
python train.py
# equivalente a:
python train.py --C 1.0 --class-weight balanced --umbral 0.5 --seed 42

# 3. Las 8 corridas del experimento (correr una sola vez)
bash run_experiments.sh

# 4. Interfaz de MLflow -> http://127.0.0.1:5001
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

> En macOS se usa el puerto 5001 porque el 5000 lo ocupa AirPlay.

### Argumentos

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | `1.0` | Inverso de la regularización |
| `--class-weight` | `balanced` | `balanced` o `none` |
| `--umbral` | `0.5` | Probabilidad a partir de la cual se predice churn |
| `--seed` | `42` | Semilla (split, validación cruzada y modelo) |

`python train.py --help` muestra todas las opciones.

## Reproducibilidad

La semilla se fija en `fijar_semilla()` y se pasa como `random_state` al split, a la validación cruzada y al modelo; `requirements.txt` fija las versiones. `run_experiments.sh` incluye una corrida de control (`repro_check_defaults`) con los mismos argumentos que la base: ambas producen métricas idénticas y el mismo `hash_predicciones`.

`mlflow.db` y `mlruns/` no se versionan; se generan al correr el paso 3.
