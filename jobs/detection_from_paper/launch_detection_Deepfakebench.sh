#!/bin/bash

set -euo pipefail

DEEPFAKEBENCH_ROOT="${DEEPFAKEBENCH_ROOT:-/work/cvcs2026/resnet_gang/external/DeepfakeBench}"

SBATCH_FILE="/homes/mlusvarghi/cvcs2026/jobs/detection_from_paper/detection_DeepfakeBench.sbatch"

CONFIG_DIR="${DEEPFAKEBENCH_ROOT}/training/config/detector"
WEIGHTS_DIR="${DEEPFAKEBENCH_ROOT}/training/weights"

JSONL_PATH="/work/cvcs2026/resnet_gang/datasets/json_standardized/json/openfake.jsonl"

OUTPUT_ROOT="/work/cvcs2026/resnet_gang/outputs/detectors_from_deepfakebench/detection_from_paper"

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

if [[ ! -d "${DEEPFAKEBENCH_ROOT}" ]]; then
    echo "[ERROR] DeepfakeBench non trovato:"
    echo "${DEEPFAKEBENCH_ROOT}"
    exit 1
fi

if [[ ! -d "${CONFIG_DIR}" ]]; then
    echo "[ERROR] Directory config non trovata:"
    echo "${CONFIG_DIR}"
    exit 1
fi

if [[ ! -d "${WEIGHTS_DIR}" ]]; then
    echo "[ERROR] Directory checkpoint non trovata:"
    echo "${WEIGHTS_DIR}"
    exit 1
fi

if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "[ERROR] File sbatch non trovato:"
    echo "${SBATCH_FILE}"
    exit 1
fi

if [[ ! -f "${JSONL_PATH}" ]]; then
    echo "[ERROR] JSONL non trovato:"
    echo "${JSONL_PATH}"
    exit 1
fi

submitted=0
skipped=0

for detector in "${DETECTORS[@]}"; do
    if ! config_path="$(find_config "${detector}")"; then
        echo "[SKIP] ${detector}: config non trovato"
        ((skipped+=1))
        continue
    fi

    if ! weights_path="$(find_weights "${detector}")"; then
        echo "[SKIP] ${detector}: checkpoint non trovato"
        ((skipped+=1))
        continue
    fi

    echo "============================================================"
    echo "[SUBMIT] ${detector}"
    echo "CONFIG  = ${config_path}"
    echo "WEIGHTS = ${weights_path}"
    echo "============================================================"

    sbatch \
        --job-name="dfb_${detector}_openfake" \
        --export=ALL,\
DETECTOR="${detector}",\
CONFIG_PATH="${config_path}",\
WEIGHTS_PATH="${weights_path}",\
DEEPFAKEBENCH_ROOT="${DEEPFAKEBENCH_ROOT}",\
JSONL_PATH="${JSONL_PATH}",\
OUTPUT_ROOT="${OUTPUT_ROOT}" \
        "${SBATCH_FILE}"

    ((submitted+=1))
done

echo "Job inviati: ${submitted}"
echo "Detector saltati: ${skipped}"