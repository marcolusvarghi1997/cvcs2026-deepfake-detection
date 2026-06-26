#!/bin/bash

set -euo pipefail

SBATCH_FILE="/homes/mlusvarghi/cvcs2026/jobs/detection_from_paper/detection_DeepfakeBench_official.sbatch"

DEEPFAKEBENCH_ROOT="/work/cvcs2026/resnet_gang/external/DeepfakeBench"
CONFIG_DIR="${DEEPFAKEBENCH_ROOT}/training/config/detector"
WEIGHTS_DIR="${DEEPFAKEBENCH_ROOT}/training/weights"

module load python/3.10.16-gcc-11.4.0
source /work/cvcs2026/resnet_gang/env.sh

DETECTORS=(
    effort
    clip
    cnn_aug
    xception
    efficientnetb4
    f3net
    spsl
    srm
)

declare -A CONFIG_CANDIDATES

CONFIG_CANDIDATES[effort]="effort.yaml"
CONFIG_CANDIDATES[clip]="clip.yaml"
CONFIG_CANDIDATES[cnn_aug]="resnet34.yaml cnn_aug.yaml cnn-aug.yaml"
CONFIG_CANDIDATES[xception]="xception.yaml"
CONFIG_CANDIDATES[efficientnetb4]="efficientnetb4.yaml efficientnet_b4.yaml efficientnet.yaml"
CONFIG_CANDIDATES[f3net]="f3net.yaml"
CONFIG_CANDIDATES[spsl]="spsl.yaml"
CONFIG_CANDIDATES[srm]="srm.yaml"

declare -A WEIGHT_PATTERNS

WEIGHT_PATTERNS[effort]="effort"
WEIGHT_PATTERNS[clip]="clip"
WEIGHT_PATTERNS[cnn_aug]="resnet34 cnn_aug cnn-aug"
WEIGHT_PATTERNS[xception]="xception"
WEIGHT_PATTERNS[efficientnetb4]="efficientnetb4 efficientnet_b4 efficientnet"
WEIGHT_PATTERNS[f3net]="f3net"
WEIGHT_PATTERNS[spsl]="spsl"
WEIGHT_PATTERNS[srm]="srm"

find_config() {
    local detector="$1"
    local candidate

    for candidate in ${CONFIG_CANDIDATES[$detector]}; do
        if [[ -f "${CONFIG_DIR}/${candidate}" ]]; then
            printf '%s\n' "${CONFIG_DIR}/${candidate}"
            return 0
        fi
    done

    return 1
}

find_weights() {
    local detector="$1"
    local pattern
    local result

    for pattern in ${WEIGHT_PATTERNS[$detector]}; do
        result="$(
            find "${WEIGHTS_DIR}" \
                -maxdepth 3 \
                -type f \
                \( \
                    -iname "*${pattern}*best*.pth" \
                    -o -iname "*${pattern}*.pth" \
                    -o -iname "*${pattern}*.pt" \
                    -o -iname "*${pattern}*.ckpt" \
                \) \
                | sort \
                | head -n 1
        )"

        if [[ -n "${result}" ]]; then
            printf '%s\n' "${result}"
            return 0
        fi
    done

    return 1
}

if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "File sbatch non trovato: ${SBATCH_FILE}"
    exit 1
fi

if [[ ! -d "${DEEPFAKEBENCH_ROOT}" ]]; then
    echo "Repository DeepfakeBench non trovato: ${DEEPFAKEBENCH_ROOT}"
    exit 1
fi

if [[ ! -d "${CONFIG_DIR}" ]]; then
    echo "Directory config non trovata: ${CONFIG_DIR}"
    exit 1
fi

if [[ ! -d "${WEIGHTS_DIR}" ]]; then
    echo "Directory pesi non trovata: ${WEIGHTS_DIR}"
    exit 1
fi

for DETECTOR in "${DETECTORS[@]}"; do
    if ! CONFIG_PATH="$(find_config "${DETECTOR}")"; then
        echo "Config non trovato per ${DETECTOR}"
        continue
    fi

    if ! WEIGHTS_PATH="$(find_weights "${DETECTOR}")"; then
        echo "Checkpoint non trovato per ${DETECTOR}"
        continue
    fi

    JOB_NAME="DeepfakeBench_${DETECTOR}_openfake_official"

    echo "============================================================"
    echo "DETECTOR: ${DETECTOR}"
    echo "CONFIG:   ${CONFIG_PATH}"
    echo "WEIGHTS:  ${WEIGHTS_PATH}"
    echo "============================================================"

    sbatch \
        --job-name="${JOB_NAME}" \
        "${SBATCH_FILE}" \
        "${DETECTOR}" \
        "${CONFIG_PATH}" \
        "${WEIGHTS_PATH}"
done