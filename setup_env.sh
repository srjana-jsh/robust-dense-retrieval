#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ipykernel

python -m ipykernel install --user   --name robust-dense-retrieval   --display-name "Python (robust-dense-retrieval)"
