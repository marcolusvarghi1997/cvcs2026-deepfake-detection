#!/bin/bash

set -euo pipefail

SBATCH_FILE="/homes/mlusvarghi/cvcs2026/jobs/detection_from_paper/detection_Effort_official.sbatch"

CHECKPOINT_DIR="/work/cvcs2026/resnet_gang/external/Effort-AIGI-Detection/checkpoints"

SDV14_WEIGHTS="${CHECKPOINT_DIR}/effort_clip_L14_trainOn_sdv14.pth"
CHAMELEON_WEIGHTS="${CHECKPOINT_DIR}/effort_clip_L14_trainOn_chameleon.pth"

module load python/3.10.16-gcc-11.4.0

source /work/cvcs2026/resnet_gang/env.sh

export HF_HOME=/work/cvcs2026/resnet_gang/.cache/huggingface

if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "File SBATCH non trovato: ${SBATCH_FILE}"
    exit 1
fi

if [[ ! -f "${SDV14_WEIGHTS}" ]]; then
    echo "Checkpoint SD v1.4 non trovato: ${SDV14_WEIGHTS}"
    exit 1
fi

if [[ ! -f "${CHAMELEON_WEIGHTS}" ]]; then
    echo "Checkpoint Chameleon non trovato: ${CHAMELEON_WEIGHTS}"
    exit 1
fi

declare -A WEIGHTS_BY_VARIANT=(
    ["sdv14"]="${SDV14_WEIGHTS}"
    ["chameleon"]="${CHAMELEON_WEIGHTS}"
)

for EFFORT_VARIANT in sdv14 chameleon; do
    WEIGHTS_PATH="${WEIGHTS_BY_VARIANT[${EFFORT_VARIANT}]}"
    JOB_NAME="Effort_${EFFORT_VARIANT}_openfake_official"

    echo "Invio job:"
    echo "  nome:    ${JOB_NAME}"
    echo "  variant: ${EFFORT_VARIANT}"
    echo "  weights: ${WEIGHTS_PATH}"

    sbatch \
        --job-name="${JOB_NAME}" \
        "${SBATCH_FILE}" \
        "${EFFORT_VARIANT}" \
        "${WEIGHTS_PATH}"
done