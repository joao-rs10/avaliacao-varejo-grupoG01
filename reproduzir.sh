#!/usr/bin/env bash
# Reproduz todos os resultados principais (≈2 min). Uso: bash reproduzir.sh
set -euo pipefail
pip install -r requirements.txt
python -m pytest -q
python questao_01/main.py --seed 1021 --modo completo
python questao_02/main.py --seed 1021 --cenarios 5000
python docs/gerar_arquitetura.py
