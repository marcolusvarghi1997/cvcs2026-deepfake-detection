#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --output=/work/cvcs2026/resnet_gang/logs/%j_%x.out
#SBATCH --error=/work/cvcs2026/resnet_gang/logs/%j_%x.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Uso: sbatch detection_CoDE_social_like.sbatch CLASSIFIER HF_REVISION"
    echo "CLASSIFIER: linear | knn | svm"
    exit 1
fi

CLASSIFIER="$1"
HF_REVISION="$2"

case "${CLASSIFIER}" in
    linear|knn|svm)
        ;;
    *)
        echo "CLASSIFIER non valido: ${CLASSIFIER}"
        echo "Valori ammessi: linear, knn, svm"
        exit 1
        ;;
esac

if [[ -z "${HF_REVISION}" ]]; then
    echo "HF_REVISION vuota"
    exit 1
fi

SCRIPT_DIR="/homes/mlusvarghi/cvcs2026/scripts/detection_from_paper_social_like"
SCRIPT_NAME="detection_CoDE_social_like.py"
DEGRADATION_MODULE="${SCRIPT_DIR}/degradation_social_like.py"
OUTPUT_ROOT="/work/cvcs2026/resnet_gang/outputs/CoDE/detection_from_paper_social_like"

if [[ ! -f "${SCRIPT_DIR}/${SCRIPT_NAME}" ]]; then
    echo "Script Python non trovato: ${SCRIPT_DIR}/${SCRIPT_NAME}"
    exit 1
fi

if [[ ! -f "${DEGRADATION_MODULE}" ]]; then
    echo "Modulo social-like non trovato: ${DEGRADATION_MODULE}"
    exit 1
fi

mkdir -p /work/cvcs2026/resnet_gang/logs
mkdir -p "${OUTPUT_ROOT}/${CLASSIFIER}"

cd /homes/mlusvarghi/cvcs2026

module load python/3.10.16-gcc-11.4.0
source /work/cvcs2026/resnet_gang/env.sh

export HF_HOME=/work/cvcs2026/resnet_gang/.cache/huggingface
export TRANSFORMERS_CACHE=/work/cvcs2026/resnet_gang/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Rende importabile degradation_social_like.py anche se cambia la cwd.
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "============================================================"
echo "START:        $(date --iso-8601=seconds)"
echo "JOB ID:       ${SLURM_JOB_ID}"
echo "JOB NAME:     ${SLURM_JOB_NAME}"
echo "HOST:         $(hostname)"
echo "CLASSIFIER:   ${CLASSIFIER}"
echo "HF REVISION:  ${HF_REVISION}"
echo "SCRIPT:       ${SCRIPT_DIR}/${SCRIPT_NAME}"
echo "DEGRADATION:  ${DEGRADATION_MODULE}"
echo "OUTPUT:       ${OUTPUT_ROOT}/${CLASSIFIER}"
echo "============================================================"

python -u "${SCRIPT_DIR}/${SCRIPT_NAME}" \
    --classifier "${CLASSIFIER}" \
    --revision "${HF_REVISION}"

echo "END: $(date --iso-8601=seconds)"