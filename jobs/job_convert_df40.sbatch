#!/bin/bash
#SBATCH --job-name=conv_df40
#SBATCH --output=/work/cvcs2026/resnet_gang/logs/conv_df40_%j.out
#SBATCH --error=/work/cvcs2026/resnet_gang/logs/conv_df40_%j.err
#SBATCH --partition=all_usr_prod
#SBATCH --qos=all_qos_dbg
#SBATCH --account=cvcs2026
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /work/cvcs2026/resnet_gang

mkdir -p logs
mkdir -p datasets/standardized

source /work/cvcs2026/resnet_gang/env.sh

python scripts/convert_df40_to_parquet.py