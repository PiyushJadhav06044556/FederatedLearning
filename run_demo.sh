#!/usr/bin/env bash
set -euo pipefail

# Preprocess CSVs
python -m scripts.preprocess_data --input data/creditcard.csv --clients 3 --out-root data --seed 42 --test-size 0.2

# Start from a clean server state so round numbering and metrics begin at 0
rm -rf server/state
mkdir -p server/state

# Build server image (clients reuse it)
export COMPOSE_PROJECT_NAME=fedavg_demo
docker compose -f docker/docker-compose.yml build server

# Launch server + 3 clients (will keep running until you press Ctrl-C)
exec docker compose -f docker/docker-compose.yml up --no-build