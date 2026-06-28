#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --job-name=UniversalFakeDetect_openfake_social_like
#SBATCH --account=cvcs2026
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

#SBATCH --output=/work/cvcs2026/resnet_gang/logs/%j_%x.out
#SBATCH --error=/work/cvcs2026/resnet_gang/logs/%j_%x.err

set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "Uso: sbatch detection_UniversalFakeDetect_social_like.sbatch"
    exit 1
fi

SCRIPT_NAME="detection_UniversalFakeDetect_social_like.py"
SCRIPT_DIR="/homes/mlusvarghi/cvcs2026/scripts/detection_from_paper_social_like"
OUTPUT_DIR="/work/cvcs2026/resnet_gang/outputs/CLIP/detection_from_paper_social_like"
UFD_REPO="/work/cvcs2026/resnet_gang/external/UniversalFakeDetect"
CHECKPOINT_PATH="${UFD_REPO}/pretrained_weights/fc_weights.pth"
JSONL_PATH="/work/cvcs2026/resnet_gang/datasets/json_standardized/json/openfake.jsonl"

if [[ ! -f "${SCRIPT_DIR}/${SCRIPT_NAME}" ]]; then
    echo "Script Python non trovato: ${SCRIPT_DIR}/${SCRIPT_NAME}"
    exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/degradation_social_like.py" ]]; then
    echo "Modulo social-like non trovato: ${SCRIPT_DIR}/degradation_social_like.py"
    exit 1
fi

if [[ ! -d "${UFD_REPO}" ]]; then
    echo "Repository UniversalFakeDetect non trovato: ${UFD_REPO}"
    exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint UniversalFakeDetect non trovato: ${CHECKPOINT_PATH}"
    exit 1
fi

if [[ ! -f "${JSONL_PATH}" ]]; then
    echo "JSONL OpenFake non trovato: ${JSONL_PATH}"
    exit 1
fi

mkdir -p /work/cvcs2026/resnet_gang/logs
mkdir -p "${OUTPUT_DIR}"

cd /homes/mlusvarghi/cvcs2026

module load python/3.10.16-gcc-11.4.0
source /work/cvcs2026/resnet_gang/env.sh

export HF_HOME=/work/cvcs2026/resnet_gang/.cache/huggingface
export TRANSFORMERS_CACHE=/work/cvcs2026/resnet_gang/.cache/huggingface
export TORCH_HOME=/work/cvcs2026/resnet_gang/.cache/torch
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

 echo "============================================================"
echo "START:               $(date --iso-8601=seconds)"
echo "JOB ID:              ${SLURM_JOB_ID}"
echo "JOB NAME:            ${SLURM_JOB_NAME}"
echo "HOST:                $(hostname)"
echo "DETECTOR:            UniversalFakeDetect"
echo "REPOSITORY:          ${UFD_REPO}"
echo "CHECKPOINT:          ${CHECKPOINT_PATH}"
echo "JSONL:               ${JSONL_PATH}"
echo "OUTPUT:              ${OUTPUT_DIR}"
echo "SCRIPT:              ${SCRIPT_DIR}/${SCRIPT_NAME}"
echo "SOCIAL MODULE:       ${SCRIPT_DIR}/degradation_social_like.py"
echo "============================================================"

python -u "${SCRIPT_DIR}/${SCRIPT_NAME}"

echo "END: $(date --iso-8601=seconds)"
