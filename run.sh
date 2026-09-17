#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/code/predict_final.py" "${1}" "${2}" "${3}" --model_dir "${SCRIPT_DIR}/models"
