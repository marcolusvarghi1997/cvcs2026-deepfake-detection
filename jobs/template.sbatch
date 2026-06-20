#!/bin/bash
#SBATCH --job-name=
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --qos=all_qos_dbg
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --output=/work/cvcs2026/resnet_gang/logs/%j_%x.out
#SBATCH --error=/work/cvcs2026/resnet_gang/logs/%j_%x.err

set -euo pipefail

SCRIPT_NAME="download_openfake_core_test.py"

cd /work/cvcs2026/resnet_gang
mkdir -p logs

cd /homes/mlusvarghi/cvcs2026

module load python/3.10.16-gcc-11.4.0

source /work/cvcs2026/resnet_gang/env.sh

echo "START $(date)"
echo "JOB ${SLURM_JOB_ID} on $(hostname)"
echo "SCRIPT scripts/${SCRIPT_NAME}"

python "scripts/${SCRIPT_NAME}"

echo "END $(date)"