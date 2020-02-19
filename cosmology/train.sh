#!/bin/bash
#SBATCH --partition gpu_yugroup
#SBATCH --gres gpu:1
module load python
python3 run_train.py
