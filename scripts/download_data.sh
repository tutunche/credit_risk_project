#!/usr/bin/env bash

set -euo pipefail

COMPETITION="home-credit-default-risk"

# Download zip one level above project folder
ZIP_DIR="../data_zip"

# Unzip files one level above project folder
DATA_DIR="../data"

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