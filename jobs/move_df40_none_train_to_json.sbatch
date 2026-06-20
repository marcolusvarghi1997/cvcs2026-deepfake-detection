#!/bin/bash
#SBATCH --job-name=move_df40_none_train_to_json
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --qos=all_qos_dbg
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --output=/homes/mlusvarghi/cvcs2026/logs/%x_%j.out
#SBATCH --error=/homes/mlusvarghi/cvcs2026/logs/%x_%j.err

set -euo pipefail

SCRIPT_NAME="move_df40_none_train_to_json.py"

cd /homes/mlusvarghi/cvcs2026
mkdir -p logs

module load python/3.10.16-gcc-11.4.0

source /work/cvcs2026/resnet_gang/env.sh

echo "START $(date)"
echo "JOB ${SLURM_JOB_ID} on $(hostname)"
echo "SCRIPT scripts/${SCRIPT_NAME}"

python "scripts/${SCRIPT_NAME}"

echo "END $(date)"