#!/bin/bash

set -euo pipefail

ROOT="/work/cvcs2026/resnet_gang/external/deepfakebench"
WEIGHTS_DIR="${ROOT}/weights"
CONFIG_DIR="${ROOT}/training/config/detector"

SBATCH_FILE="/homes/mlusvarghi/cvcs2026/jobs/detection_from_paper_social_like/detection_DeepfakeBench_social_like.sbatch"

DETECTORS=(
    core
    spsl
    ucf
)

declare -A WEIGHTS=(
    [core]="core_best.pth"
    [spsl]="spsl_best.pth"
    [ucf]="ucf_best.pth"
)

declare -A CONFIGS=(
    [core]="core.yaml"
    [spsl]="spsl.yaml"
    [ucf]="ucf.yaml"
)

if [[ ! -d "${ROOT}" ]]; then
    echo "Repository non trovata: ${ROOT}"
    exit 1
fi

if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "File SBATCH non trovato: ${SBATCH_FILE}"
    exit 1
fi

REVISION="$(git -C "${ROOT}" describe --tags --always 2>/dev/null || true)"

if [[ "${REVISION}" != "v1.0.1" ]]; then
    echo "DeepfakeBench deve essere fissato a v1.0.1"
    echo "Revisione corrente: ${REVISION:-sconosciuta}"
    exit 1
fi

for DETECTOR in "${DETECTORS[@]}"; do
    CONFIG_PATH="${CONFIG_DIR}/${CONFIGS[$DETECTOR]}"
    WEIGHTS_PATH="${WEIGHTS_DIR}/${WEIGHTS[$DETECTOR]}"
    JOB_NAME="DFB_${DETECTOR}_OpenFake_social_like"

    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "Config non trovata: ${CONFIG_PATH}"
        exit 1
    fi

    if [[ ! -s "${WEIGHTS_PATH}" ]]; then
        echo "Checkpoint non trovato o vuoto: ${WEIGHTS_PATH}"
        exit 1
    fi

    JOB_ID="$(
        sbatch \
            --parsable \
            --job-name="${JOB_NAME}" \
            "${SBATCH_FILE}" \
            "${DETECTOR}" \
            "${CONFIG_PATH}" \
            "${WEIGHTS_PATH}"
    )"

    echo "${DETECTOR}: job ${JOB_ID}"
done