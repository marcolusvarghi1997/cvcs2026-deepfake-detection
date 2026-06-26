#!/bin/bash

#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/work/cvcs2026/resnet_gang/logs/%j_%x.out
#SBATCH --error=/work/cvcs2026/resnet_gang/logs/%j_%x.err

set -euo pipefail

if [[ $# -ne 3 ]]; then
echo "Uso: sbatch detection_DeepfakeBench_official.sbatch DETECTOR CONFIG WEIGHTS"
exit 1
fi

DETECTOR_ID="$1"
CONFIG_PATH="$2"
WEIGHTS_PATH="$3"

ROOT="/work/cvcs2026/resnet_gang/external/deepfakebench"

PYTHON_SCRIPT="/homes/mlusvarghi/cvcs2026/scripts/detection_from_paper/detection_DeepfakeBench_official.py"

JSONL_PATH="/work/cvcs2026/resnet_gang/datasets/json_standardized/json/openfake.jsonl"

OUTPUT_ROOT="/work/cvcs2026/resnet_gang/outputs/DeepfakeBench/openfake_v1_0_1"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
THRESHOLD="${THRESHOLD:-0.5}"
SEED="${SEED:-1024}"
USE_AMP="${USE_AMP:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -d "${ROOT}" ]]; then
echo "Repository non trovata: ${ROOT}"
exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
echo "Script Python non trovato: ${PYTHON_SCRIPT}"
exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
echo "Config non trovata: ${CONFIG_PATH}"
exit 1
fi

if [[ ! -s "${WEIGHTS_PATH}" ]]; then
echo "Checkpoint non trovato o vuoto: ${WEIGHTS_PATH}"
exit 1
fi

if [[ ! -f "${JSONL_PATH}" ]]; then
echo "Dataset JSONL non trovato: ${JSONL_PATH}"
exit 1
fi

mkdir -p /work/cvcs2026/resnet_gang/logs
mkdir -p "${OUTPUT_ROOT}"

module purge
module load python/3.10.16-gcc-11.4.0

source /work/cvcs2026/resnet_gang/env.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${ROOT}"

ARGS=(
--detector-id "${DETECTOR_ID}"
--config "${CONFIG_PATH}"
--weights "${WEIGHTS_PATH}"
--deepfakebench-root "${ROOT}"
--jsonl "${JSONL_PATH}"
--output-root "${OUTPUT_ROOT}"
--batch-size "${BATCH_SIZE}"
--workers "${NUM_WORKERS}"
--threshold "${THRESHOLD}"
--seed "${SEED}"
)

if [[ "${USE_AMP}" == "1" ]]; then
ARGS+=(--amp)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
ARGS+=(--overwrite)
fi

echo "START:      $(date --iso-8601=seconds)"
echo "JOB ID:     ${SLURM_JOB_ID}"
echo "DETECTOR:   ${DETECTOR_ID}"
echo "CONFIG:     ${CONFIG_PATH}"
echo "CHECKPOINT: ${WEIGHTS_PATH}"

srun python -u "${PYTHON_SCRIPT}" "${ARGS[@]}"

echo "END: $(date --iso-8601=seconds)"