#!/usr/bin/env bash
set -euo pipefail

for config in config/*.json; do
    echo "Running embedding_analyzer -c ${config}"
    embedding_analyzer -c "${config}"
done
