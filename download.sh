#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# TODO: replace with the real Google Drive file ID for models.zip before submission
GDRIVE_FILE_ID="REPLACE_WITH_GDRIVE_FILE_ID"

cd "${SCRIPT_DIR}"
gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O models.tar.gz
tar -xzf models.tar.gz
rm models.tar.gz
