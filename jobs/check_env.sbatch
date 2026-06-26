#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --qos=all_qos_dbg
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

#SBATCH --output=/homes/mlusvarghi/cvcs2026/logs/%x_%j.out
#SBATCH --error=/homes/mlusvarghi/cvcs2026/logs/%x_%j.err

set -euo pipefail

SCRIPT_NAME="check_env.py"

cd /homes/mlusvarghi/cvcs2026
mkdir -p logs

module load python/3.10.16-gcc-11.4.0

source /work/cvcs2026/resnet_gang/env.sh

echo "START $(date)"
echo "JOB ${SLURM_JOB_ID} on $(hostname)"
echo "SCRIPT scripts/${SCRIPT_NAME}"

python "scripts/${SCRIPT_NAME}"

echo "END $(date)"