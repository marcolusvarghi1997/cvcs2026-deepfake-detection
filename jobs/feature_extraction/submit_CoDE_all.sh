#!/bin/bash

set -euo pipefail

SBATCH_FILE="/homes/mlusvarghi/cvcs2026/jobs/feature_extraction/feature_extraction_CoDE.sbatch"

for CASE in case1 case2; do
    for SPLIT_NAME in train val test; do

        JOB_NAME="CoDE_${CASE}_${SPLIT_NAME}"

        sbatch \
            --job-name="${JOB_NAME}" \
            "${SBATCH_FILE}" \
            "${CASE}" \
            "${SPLIT_NAME}"

    done
done