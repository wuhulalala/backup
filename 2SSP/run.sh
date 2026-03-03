#!/bin/bash
#SBATCH --job-name=ssp   
#SBATCH --gpus=1
#SBATCH --output=logs/ssp%j.out
#SBATCH --error=logs/ssp%j.err



source ~/.bashrc
conda activate ssp
source /home/bingxing2/home/scx9kvs/zsh/2SSP/env.sh
cd /home/bingxing2/home/scx9kvs/zsh/2SSP
#echo "LD_PRELOAD = $LD_PRELOAD"
bash scripts/runssp.sh
