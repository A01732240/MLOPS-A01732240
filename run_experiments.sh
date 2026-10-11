#!/usr/bin/env bash
# Experimento churn-telco-logreg-ganador: 8 corridas del modelo ganador
# variando C, class_weight, umbral y semilla.
# Uso: bash run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")"

# Base: C=1, balanced, umbral 0.5, seed 42
python train.py

# Regularización
python train.py --C 0.01
python train.py --C 100

# Sin balancear clases
python train.py --class-weight none

# Umbral de decisión
python train.py --umbral 0.35
python train.py --umbral 0.65

# Control de reproducibilidad: mismos argumentos que la base
python train.py --run-name repro_check_defaults

# Otra semilla (cambia el split)
python train.py --seed 7
