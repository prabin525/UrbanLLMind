#!/bin/bash
#SBATCH --job-name=20b
##SBATCH --qos=gpu
##SBATCH --partition=contrib-gpuq
##SBATCH --gres=gpu:A100.80gb:1
#SBATCH --qos=dpfoser
#SBATCH --partition=contrib-H100
#SBATCH --gres=gpu:H100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/%u/slurm_logs/%x-%j-%N.out
#SBATCH --error=/scratch/%u/slurm_logs/%x-%j-%N.err

set -euo pipefail


PROJECT_ROOT=${PROJECT_ROOT:-/scratch/pbhanda2/projects/mmv4}

HF_MODEL_ID='openai/gpt-oss-20b'

module load singularity
mkdir -p /scratch/$USER/slurm_logs

cd "${PROJECT_ROOT}/slurm"
source "envvars"

export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_IBEXT_DISABLE=${NCCL_IBEXT_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_NET=${NCCL_NET:-Socket}
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}



cd "${PROJECT_ROOT}"
automodel finetune llm -c train/gptoss_20b_peft.yaml
# python train/infer_hf_peft_compare.py
