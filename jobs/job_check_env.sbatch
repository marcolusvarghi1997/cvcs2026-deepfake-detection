#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --partition=all_usr_prod
#SBATCH --qos=all_qos_dbg
#SBATCH --account=cvcs2026
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00

mkdir -p /work/cvcs2026/resnet_gang/logs

LOGFILE="/work/cvcs2026/resnet_gang/logs/check_env_${SLURM_JOB_ID}.log"

{
    echo "START $(date)"
    echo "JOB_ID: ${SLURM_JOB_ID}"
    echo "NODE: $(hostname)"
    echo

    source /work/cvcs2026/resnet_gang/env.sh

    python /work/cvcs2026/resnet_gang/scripts/check_env.py

    echo
    echo "END $(date)"
} > "$LOGFILE" 2>&1