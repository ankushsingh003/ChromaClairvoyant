#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip first
python -m pip install --upgrade pip

# Install dependencies with no-cache to prevent corrupted wheel issues
pip install --no-cache-dir -r requirements.txt

python manage.py collectstatic --no-input
python manage.py migrate
