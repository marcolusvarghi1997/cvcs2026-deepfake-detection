#!/bin/bash

set -euo pipefail

SBATCH_FILE="/homes/mlusvarghi/cvcs2026/jobs/detection_from_paper/detection_CoDE_official.sbatch"
MODEL_REPO="aimagelab/CoDE"

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

echo "CoDE Hugging Face revision fissata: ${HF_REVISION}"

for CLASSIFIER in linear knn svm; do
    JOB_NAME="CoDE_${CLASSIFIER}_openfake_official"

    sbatch \
        --job-name="${JOB_NAME}" \
        "${SBATCH_FILE}" \
        "${CLASSIFIER}" \
        "${HF_REVISION}"
done