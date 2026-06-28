#!/bin/bash

set -euo pipefail

SBATCH_FILE="/homes/mlusvarghi/cvcs2026/jobs/detection_from_paper_social_like/detection_CoDE_social_like.sbatch"
MODEL_REPO="aimagelab/CoDE"

# Valori predefiniti:
# il 70% delle immagini subisce la pipeline social-like;
# il seed garantisce trasformazioni riproducibili.
SOCIAL_PROBABILITY="${1:-0.7}"
SOCIAL_SEED="${2:-42}"

if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "File SBATCH non trovato: ${SBATCH_FILE}"
    exit 1
fi

if ! [[ "${SOCIAL_SEED}" =~ ^[0-9]+$ ]]; then
    echo "SOCIAL_SEED deve essere un intero non negativo: ${SOCIAL_SEED}"
    exit 1
fi

module load python/3.10.16-gcc-11.4.0

source /work/cvcs2026/resnet_gang/env.sh

export HF_HOME=/work/cvcs2026/resnet_gang/.cache/huggingface

HF_REVISION="$(
python - <<'PY'
from huggingface_hub import model_info

print(model_info("aimagelab/CoDE").sha)
PY
)"

if [[ -z "${HF_REVISION}" ]]; then
    echo "Impossibile risolvere la revisione Hugging Face di ${MODEL_REPO}"
    exit 1
fi

echo "============================================================"
echo "CoDE Hugging Face revision: ${HF_REVISION}"
echo "Social probability:         ${SOCIAL_PROBABILITY}"
echo "Social seed:                ${SOCIAL_SEED}"
echo "============================================================"

for CLASSIFIER in linear knn svm; do
    JOB_NAME="CoDE_${CLASSIFIER}_openfake_social_like"

    echo "Invio job ${JOB_NAME}"

    sbatch \
        --job-name="${JOB_NAME}" \
        "${SBATCH_FILE}" \
        "${CLASSIFIER}" \
        "${HF_REVISION}" \
        "${SOCIAL_PROBABILITY}" \
        "${SOCIAL_SEED}"
done