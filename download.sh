#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GDRIVE_FILE_ID="1k0aMwypAyXA97cjbiebHtVrZutN7Yy0_"

cd "${SCRIPT_DIR}"
gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O models.tar.gz
tar -xzf models.tar.gz
rm models.tar.gz
