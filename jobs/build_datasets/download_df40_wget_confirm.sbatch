#!/bin/bash
#SBATCH --job-name=df40_wget
#SBATCH --partition=all_usr_prod
#SBATCH --qos=all_qos_dbg
#SBATCH --account=cvcs2026
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/work/cvcs2026/resnet_gang/logs/df40_wget_%j.out
#SBATCH --error=/work/cvcs2026/resnet_gang/logs/df40_wget_%j.err

set -e

source /work/cvcs2026/resnet_gang/env.sh

bash /work/cvcs2026/resnet_gang/scripts/download_df40_wget_confirm.sh
