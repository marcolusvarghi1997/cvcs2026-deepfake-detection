#!/bin/bash

module purge

module load python/3.10.16-gcc-11.4.0
module load python-venv/1.0-none-none
module load py-pip/25.1.1-none-none

module load git/2.53.0-gcc-11.4.0 

module load cuda/12.6.3-none-none
module load cuda/13.0.0-none-none
module load cudnn/8.9.7.29-13-none-none-cuda-13.0.0

module load py-numpy/1.26.4-gcc-11.4.0
module load py-torch/2.8.0-gcc-11.4.0-cuda-13.0.0
module load py-tqdm/4.67.1-none-none

source /work/cvcs2026/resnet_gang/CVenv/bin/activate