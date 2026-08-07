#!/bin/bash
#SBATCH --job-name=create_training_data
#SBATCH --partition=bigmem
#SBATCH --mem-per-cpu=32G
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

#SBATCH --output=/scratch/%u/slurm_logs/%x-%j-%N.out
#SBATCH --error=/scratch/%u/slurm_logs/%x-%j-%N.err

cd /scratch/pbhanda2/projects/mmv4
python -m training_data_generator.main
set -euo pipefail

