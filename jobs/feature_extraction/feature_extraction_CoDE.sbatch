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

if [[ $# -ne 2 ]]; then
    echo "Uso: sbatch feature_extraction_CoDE.sbatch CASE SPLIT_NAME"
    exit 1
fi

CASE="$1"
SPLIT_NAME="$2"

case "${CASE}" in
    case1|case2)
        ;;
    *)
        echo "CASE non valido: ${CASE}"
        exit 1
        ;;
esac

case "${SPLIT_NAME}" in
    train|val|test)
        ;;
    *)
        echo "SPLIT_NAME non valido: ${SPLIT_NAME}"
        exit 1
        ;;
esac

SCRIPT_NAME="feature_extraction_CoDE.py"

mkdir -p /work/cvcs2026/resnet_gang/logs

cd /homes/mlusvarghi/cvcs2026

module load python/3.10.16-gcc-11.4.0
source /work/cvcs2026/resnet_gang/env.sh

export HF_HOME=/work/cvcs2026/resnet_gang/.cache/huggingface
export TRANSFORMERS_CACHE=/work/cvcs2026/resnet_gang/.cache/huggingface

echo "============================================================"
echo "START:       $(date)"
echo "JOB ID:      ${SLURM_JOB_ID}"
echo "JOB NAME:    ${SLURM_JOB_NAME}"
echo "HOST:        $(hostname)"
echo "CASE:        ${CASE}"
echo "SPLIT:       ${SPLIT_NAME}"
echo "SCRIPT:      scripts/feature_extraction/${SCRIPT_NAME}"
echo "============================================================"

python -u "scripts/feature_extraction/${SCRIPT_NAME}" \
    --case "${CASE}" \
    --split "${SPLIT_NAME}"

echo "END: $(date)"