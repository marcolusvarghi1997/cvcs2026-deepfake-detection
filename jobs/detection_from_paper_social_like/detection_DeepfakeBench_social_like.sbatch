#!/bin/bash

#SBATCH --partition=boost_usr_prod
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
    echo "Uso: sbatch detection_DeepfakeBench_social_like.sbatch DETECTOR CONFIG WEIGHTS"
    echo "DETECTOR: core | spsl | ucf"
    exit 1
fi

DETECTOR_ID="${1,,}"
CONFIG_PATH="$2"
WEIGHTS_PATH="$3"

case "${DETECTOR_ID}" in
    core|spsl|ucf)
        ;;
    *)
        echo "Detector non supportato: ${DETECTOR_ID}"
        echo "Valori ammessi: core, spsl, ucf"
        exit 1
        ;;
esac

ROOT="/work/cvcs2026/resnet_gang/external/deepfakebench"
CONDA_ENV="/work/cvcs2026/resnet_gang/env"

PYTHON_SCRIPT="/homes/mlusvarghi/cvcs2026/scripts/detection_from_paper_social_like/detection_DeepfakeBench_social_like.py"
DEGRADATION_SCRIPT="/homes/mlusvarghi/cvcs2026/scripts/detection_from_paper_social_like/degradation_social_like.py"

JSONL_PATH="/work/cvcs2026/resnet_gang/datasets/json_standardized/json/openfake.jsonl"

OUTPUT_ROOT="/work/cvcs2026/resnet_gang/outputs/DeepfakeBench/openfake_social_like_v1_0_1"

BACKBONE_WEIGHTS="${ROOT}/training/pretrained/xception-b5690688.pth"

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

if [[ ! -d "${CONDA_ENV}" ]]; then
    echo "Ambiente Conda non trovato: ${CONDA_ENV}"
    exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "Script Python non trovato: ${PYTHON_SCRIPT}"
    exit 1
fi

if [[ ! -f "${DEGRADATION_SCRIPT}" ]]; then
    echo "Modulo social-like non trovato: ${DEGRADATION_SCRIPT}"
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

if [[ ! -f "${BACKBONE_WEIGHTS}" ]]; then
    echo "Checkpoint Xception ImageNet non trovato: ${BACKBONE_WEIGHTS}"
    exit 1
fi

if [[ ! -f "${JSONL_PATH}" ]]; then
    echo "Dataset JSONL non trovato: ${JSONL_PATH}"
    exit 1
fi

mkdir -p /work/cvcs2026/resnet_gang/logs
mkdir -p "${OUTPUT_ROOT}"

module purge
module load anaconda3/2023.09-0
module load git/2.53.0
module load cuda/12.6.3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

PYTHON_BIN="$(command -v python || true)"

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Interprete Python non trovato nell'ambiente Conda"
    exit 1
fi

if [[ "${PYTHON_BIN}" != "${CONDA_ENV}/bin/python" ]]; then
    echo "Interprete Python inatteso: ${PYTHON_BIN}"
    echo "Interprete atteso: ${CONDA_ENV}/bin/python"
    exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

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

echo "============================================================"
echo "START:              $(date --iso-8601=seconds)"
echo "JOB ID:             ${SLURM_JOB_ID}"
echo "JOB NAME:           ${SLURM_JOB_NAME}"
echo "HOST:               $(hostname)"
echo "DETECTOR:           ${DETECTOR_ID}"
echo "CONFIG:             ${CONFIG_PATH}"
echo "CHECKPOINT:         ${WEIGHTS_PATH}"
echo "BACKBONE:           ${BACKBONE_WEIGHTS}"
echo "JSONL:              ${JSONL_PATH}"
echo "OUTPUT ROOT:        ${OUTPUT_ROOT}"
echo "BATCH SIZE:         ${BATCH_SIZE}"
echo "WORKERS:            ${NUM_WORKERS}"
echo "THRESHOLD:          ${THRESHOLD}"
echo "SEED:               ${SEED}"
echo "USE AMP:            ${USE_AMP}"
echo "OVERWRITE:          ${OVERWRITE}"
echo "CONDA ENV:          ${CONDA_ENV}"
echo "PYTHON:             ${PYTHON_BIN}"
echo "SOCIAL-LIKE MODULE: ${DEGRADATION_SCRIPT}"
echo "============================================================"

"${PYTHON_BIN}" --version

"${PYTHON_BIN}" - <<'PY'
import numpy
import torch

print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("CUDA disponibile:", torch.cuda.is_available())
print("CUDA build:", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA non disponibile nel job")

print("GPU:", torch.cuda.get_device_name(0))
PY

srun --export=ALL \
    "${PYTHON_BIN}" \
    -u "${PYTHON_SCRIPT}" \
    "${ARGS[@]}"

echo "END: $(date --iso-8601=seconds)"