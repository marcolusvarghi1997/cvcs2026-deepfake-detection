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
    echo "Uso: sbatch detection_CoDE_official.sbatch CLASSIFIER HF_REVISION"
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

SCRIPT_NAME="detection_CoDE_official.py"

cd /work/cvcs2026/resnet_gang
mkdir -p logs
mkdir -p outputs/CoDE/detection_from_paper

cd /homes/mlusvarghi/cvcs2026

module load python/3.10.16-gcc-11.4.0

source /work/cvcs2026/resnet_gang/env.sh

export HF_HOME=/work/cvcs2026/resnet_gang/.cache/huggingface
export TRANSFORMERS_CACHE=/work/cvcs2026/resnet_gang/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "============================================================"
echo "START:        $(date --iso-8601=seconds)"
echo "JOB ID:       ${SLURM_JOB_ID}"
echo "JOB NAME:     ${SLURM_JOB_NAME}"
echo "HOST:         $(hostname)"
echo "CLASSIFIER:   ${CLASSIFIER}"
echo "HF REVISION:  ${HF_REVISION}"
echo "SCRIPT:       scripts/detection_from_paper/${SCRIPT_NAME}"
echo "============================================================"

python -u "scripts/detection_from_paper/${SCRIPT_NAME}" \
    --classifier "${CLASSIFIER}" \
    --revision "${HF_REVISION}"

echo "END: $(date --iso-8601=seconds)"