#!/bin/bash
#SBATCH --job-name=ProWater1223
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --account=zhangguo

conda init
source ~/.bashrc

cd /project/cuihao/PromptWater
conda activate medSAM

# 运行脚本
python train_promptwaternet_linux.py