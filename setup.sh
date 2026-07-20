#!/usr/bin/env bash
set -euo pipefail

echo ">>> creating venv"
python3 -m venv .venv
source .venv/bin/activate

echo ">>> installing deps"
pip install --upgrade pip
pip install -r requirements.txt

echo ">>> done"
echo ">>> run: source .venv/bin/activate"
