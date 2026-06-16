#!/bin/bash
#SBATCH --job-name=df40_extract
#SBATCH --partition=all_usr_prod
#SBATCH --qos=all_qos_dbg
#SBATCH --account=cvcs2026
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/work/cvcs2026/resnet_gang/logs/df40_extract_%j.out
#SBATCH --error=/work/cvcs2026/resnet_gang/logs/df40_extract_%j.err

set -e

mkdir -p /work/cvcs2026/resnet_gang/logs

bash /work/cvcs2026/resnet_gang/scripts/extract_df40_zips.sh
