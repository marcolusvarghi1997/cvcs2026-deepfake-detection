#!/bin/bash

set -euo pipefail

SBATCH_FILE="/homes/mlusvarghi/cvcs2026/jobs/detection_from_paper_social_like/detection_CoDE_social_like.sbatch"
MODEL_REPO="aimagelab/CoDE"

if [[ ! -f "${SBATCH_FILE}" ]]; then
    echo "File sbatch non trovato: ${SBATCH_FILE}"
    exit 1
fi

module load python/3.10.16-gcc-11.4.0
source /work/cvcs2026/resnet_gang/env.sh

export HF_HOME=/work/cvcs2026/resnet_gang/.cache/huggingface

HF_REVISION="$(
python - <<'PY'
from huggingface_hub import model_info

revision = model_info("aimagelab/CoDE").sha
if not revision:
    raise RuntimeError("Revisione Hugging Face non disponibile")
print(revision)
PY
)"

if [[ -z "${HF_REVISION}" ]]; then
    echo "Impossibile risolvere la revisione Hugging Face di ${MODEL_REPO}"
    exit 1
fi

echo "CoDE Hugging Face revision fissata: ${HF_REVISION}"

for CLASSIFIER in linear knn svm; do
    JOB_NAME="CoDE_${CLASSIFIER}_openfake_social_like"

    sbatch \
        --job-name="${JOB_NAME}" \
        "${SBATCH_FILE}" \
        "${CLASSIFIER}" \
        "${HF_REVISION}"
done