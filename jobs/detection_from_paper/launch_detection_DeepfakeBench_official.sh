#!/bin/bash

set -euo pipefail

ROOT="/work/cvcs2026/resnet_gang/external/deepfakebench"
WEIGHTS_DIR="${ROOT}/weights"
CONFIG_DIR="${ROOT}/training/config/detector"

SBATCH_FILE="/homes/mlusvarghi/cvcs2026/jobs/detection_from_paper/detection_DeepfakeBench_official.sbatch"

DETECTORS=(
capsule
cnnaug
core
effnb4
f3net
ffd
meso4Incep
meso4
recce
spsl
srm
ucf
xception
)

declare -A WEIGHTS=(
[capsule]="capsule_best.pth"
[cnnaug]="cnnaug_best.pth"
[core]="core_best.pth"
[effnb4]="effnb4_best.pth"
[f3net]="f3net_best.pth"
[ffd]="ffd_best.pth"
[meso4Incep]="meso4Incep_best.pth"
[meso4]="meso4_best.pth"
[recce]="recce_best.pth"
[spsl]="spsl_best.pth"
[srm]="srm_best.pth"
[ucf]="ucf_best.pth"
[xception]="xception_best.pth"
)

declare -A CONFIGS=(
[capsule]="capsule_net.yaml"
[cnnaug]="resnet34.yaml"
[core]="core.yaml"
[effnb4]="efficientnetb4.yaml"
[f3net]="f3net.yaml"
[ffd]="ffd.yaml"
[meso4Incep]="meso4Inception.yaml"
[meso4]="meso4.yaml"
[recce]="recce.yaml"
[spsl]="spsl.yaml"
[srm]="srm.yaml"
[ucf]="ucf.yaml"
[xception]="xception.yaml"
)

if [[ ! -d "${ROOT}" ]]; then
echo "Repository non trovata: ${ROOT}"
exit 1
fi

if [[ ! -f "${SBATCH_FILE}" ]]; then
echo "File sbatch non trovato: ${SBATCH_FILE}"
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
JOB_NAME="DFB_${DETECTOR}_OpenFake"


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