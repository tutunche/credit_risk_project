#!/usr/bin/env bash

set -euo pipefail

COMPETITION="home-credit-default-risk"

# Resolve project root from this script's location so the script
# works no matter where it is invoked from
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Download zip into project folder
ZIP_DIR="${PROJECT_ROOT}/data_zip"

# Unzip files into project folder
DATA_DIR="${PROJECT_ROOT}/data"

ZIP_PATH="${ZIP_DIR}/${COMPETITION}.zip"

echo "Creating directories..."
mkdir -p "$ZIP_DIR"
mkdir -p "$DATA_DIR"

echo "Checking Kaggle CLI..."
if ! command -v kaggle >/dev/null 2>&1; then
    echo "Error: kaggle CLI not found."
    echo "Install it with:"
    echo "  pip install kaggle"
    exit 1
fi

echo "Downloading Home Credit Default Risk data into:"
echo "  $ZIP_DIR"

echo "If this fails with 403, accept the competition rules on Kaggle first."

kaggle competitions download \
    -c "$COMPETITION" \
    -p "$ZIP_DIR"

echo "Unzipping dataset into:"
echo "  $DATA_DIR"

unzip -o "$ZIP_PATH" -d "$DATA_DIR"

echo ""
echo "Done."
echo "Zip file saved to:"
echo "  $ZIP_PATH"
echo ""
echo "Data extracted to:"
echo "  $DATA_DIR"
echo ""
echo "Files:"
ls -lh "$DATA_DIR"