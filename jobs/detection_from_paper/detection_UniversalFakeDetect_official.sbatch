#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --qos=all_qos_dbg
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --output=/work/cvcs2026/resnet_gang/logs/%j_%x.out
#SBATCH --error=/work/cvcs2026/resnet_gang/logs/%j_%x.err

set -euo pipefail

SCRIPT_NAME="detection_UniversalFakeDetect_official.py"

UFD_REPO="/work/cvcs2026/resnet_gang/external/UniversalFakeDetect"
CHECKPOINT_PATH="${UFD_REPO}/pretrained_weights/fc_weights.pth"

cd /work/cvcs2026/resnet_gang

mkdir -p logs
mkdir -p outputs/CLIP/detection_from_paper

if [[ ! -d "${UFD_REPO}" ]]; then
    echo "Repository UniversalFakeDetect non trovato:"
    echo "${UFD_REPO}"
    exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint ufficiale non trovato:"
    echo "${CHECKPOINT_PATH}"
    exit 1
fi

cd /homes/mlusvarghi/cvcs2026

module load python/3.10.16-gcc-11.4.0

source /work/cvcs2026/resnet_gang/env.sh

export HF_HOME=/work/cvcs2026/resnet_gang/.cache/huggingface
export TORCH_HOME=/work/cvcs2026/resnet_gang/.cache/torch
export TOKENIZERS_PARALLELISM=false

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "============================================================"
echo "START:         $(date --iso-8601=seconds)"
echo "JOB ID:        ${SLURM_JOB_ID}"
echo "JOB NAME:      ${SLURM_JOB_NAME}"
echo "HOST:          $(hostname)"
echo "DETECTOR:      UniversalFakeDetect"
echo "ARCHITECTURE:  CLIP:ViT-L/14"
echo "CLASSIFIER:    linear"
echo "REPOSITORY:    ${UFD_REPO}"
echo "CHECKPOINT:    ${CHECKPOINT_PATH}"
echo "SCRIPT:        scripts/detection_from_paper/${SCRIPT_NAME}"
echo "============================================================"

python -u "scripts/detection_from_paper/${SCRIPT_NAME}"

echo "============================================================"
echo "END: $(date --iso-8601=seconds)"
echo "============================================================"