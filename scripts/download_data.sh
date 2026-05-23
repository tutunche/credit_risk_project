#!/usr/bin/env bash

set -e

COMPETITION="home-credit-default-risk"
RAW_DIR="data/raw"
DATA_DIR="${RAW_DIR}/${COMPETITION}"
ZIP_PATH="${RAW_DIR}/${COMPETITION}.zip"

mkdir -p "$RAW_DIR"

echo "Downloading Kaggle competition data..."
kaggle competitions download -c "$COMPETITION" -p "$RAW_DIR"

echo "Unzipping data..."
mkdir -p "$DATA_DIR"
unzip -o "$ZIP_PATH" -d "$DATA_DIR"

echo "Done."
echo "Data saved to: $DATA_DIR"