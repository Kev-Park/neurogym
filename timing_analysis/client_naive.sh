#!/bin/bash
#SBATCH --job-name=client_naive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --output=logs/client_naive_%j.out
#SBATCH --error=logs/client_naive_%j.err

cd /scratch/gpfs/PMITTAL/peiyang/px4668/IntroToRL/neurogym_dev

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /scratch/gpfs/PMITTAL/peiyang/px4668/conda_envs/rlintro

python client_latency_naive.py
