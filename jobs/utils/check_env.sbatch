#!/bin/bash
#SBATCH --job-name=check_env2
#SBATCH --partition=boost_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --qos=all_qos_dbg
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --output=/homes/mlusvarghi/cvcs2026/logs/%x_%j.out
#SBATCH --error=/homes/mlusvarghi/cvcs2026/logs/%x_%j.err

set -euo pipefail

SCRIPT_NAME="check_env.py"

cd /homes/mlusvarghi/cvcs2026
mkdir -p logs

module load python/3.11.15

source /work/cvcs2026/resnet_gang/env2.sh

echo "START $(date)"
echo "JOB ${SLURM_JOB_ID} on $(hostname)"
echo "SCRIPT scripts/utils/${SCRIPT_NAME}"

python "scripts/utils/${SCRIPT_NAME}"

echo "END $(date)"