#!/bin/bash
#SBATCH --job-name=plant_disease
#SBATCH --output=/project/vcldron1/final_project/outputs/plant_disease_%j.out
#SBATCH --error=/project/vcldron1/final_project/outputs/plant_disease_%j.err
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Initialize conda
source /home/vcldron1/miniconda3/etc/profile.d/conda.sh
conda activate /project/vcldron1/envs/hw4env

# HuggingFace token — loaded here so it never appears in source code
export HF_TOKEN="hf_KrzPIeVT0OUxZXIYSGBXkmSpPtkVOCuQLB"

# Run pipeline
python /project/vcldron1/final_project/scripts/train_model.py