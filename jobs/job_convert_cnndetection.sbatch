#!/bin/bash
#SBATCH --job-name=conv_cnndet
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --qos=all_qos_dbg
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/cvcs2026/resnet_gang/logs/conv_cnndet_%j.out
#SBATCH --error=/work/cvcs2026/resnet_gang/logs/conv_cnndet_%j.err

mkdir -p /work/cvcs2026/resnet_gang/logs

source /work/cvcs2026/resnet_gang/env.sh

python /work/cvcs2026/resnet_gang/scripts/convert_cnndetection_to_parquet.py