#!/bin/bash
#SBATCH --job-name=run_vllm_H100


##HOPPER
#SBATCH --qos=gpu --partition=gpu --gres=gpu:H100.80gb:4

#SBATCH --output=/scratch/%u/slurm_logs/%x-%j-%N.out
#SBATCH --error=/scratch/%u/slurm_logs/%x-%j-%N.err

#SBATCH --mem=800G
#SBATCH --time=5-00:00:00

#SBATCH --nodes=1 --ntasks-per-node=1 

module load singularity
module load gnu9
module load gcc
module load cuda/13.1
cd /scratch/pbhanda2/projects/mmv4

source slurm/envvars

# Function to check if a TCP connection can be established
function wait_for_port() {
    local host=$1
    local port=$2
    while ! nc -z $host $port; do
        echo "Waiting for $host:$port to be available..."
        sleep 10
    done
    echo "$host:$port is available."
}

cleanup() {
  if [ -n "$PROM_PID" ]; then kill "$PROM_PID" >/dev/null 2>&1; fi
  if [ -n "$GRAFANA_PID" ]; then kill "$GRAFANA_PID" >/dev/null 2>&1; fi
  if [ -n "$VLLM_PID" ]; then kill "$VLLM_PID" >/dev/null 2>&1; fi
}
trap cleanup EXIT

export INFERENCE_MODEL=openai/gpt-oss-120b


export VLLM_IMAGE_LOC=/scratch/pbhanda2/singularity_container/vllm-latest.sif
# Get the actual hostname of the allocated node
VLLM_HOSTNAME=$(hostname -f)
echo "VLLM server will run on: $VLLM_HOSTNAME"

VLLM_ENDPOINT_DIR="/scratch/pbhanda2/projects/mmv4/vllm_endpoints"
mkdir -p "$VLLM_ENDPOINT_DIR"
VLLM_ENDPOINT_FILE="${VLLM_ENDPOINT_DIR}/vllm_endpoint_${SLURM_JOB_ID}.txt"


# export TORCH_CUDA_ARCH_LIST="8.0"

VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-128}
VLLM_MAX_BATCHED_TOKENS=${VLLM_MAX_BATCHED_TOKENS:-32768}
VLLM_GPU_MEMORY_UTIL=${VLLM_GPU_MEMORY_UTIL:-0.90}
VLLM_ENABLE_CHUNKED_PREFILL=${VLLM_ENABLE_CHUNKED_PREFILL:-1}
VLLM_DISABLE_LOG_REQUESTS=${VLLM_DISABLE_LOG_REQUESTS:-1}
VLLM_DISABLE_FRONTEND_METRICS=${VLLM_DISABLE_FRONTEND_METRICS:-0}

EXTRA_ARGS=""
if [ "$VLLM_ENABLE_CHUNKED_PREFILL" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --enable-chunked-prefill"
fi
if [ "$VLLM_DISABLE_FRONTEND_METRICS" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --disable-frontend-metrics"
fi

singularity run --nv \
    --bind ./slurm/tmp:/tmp \
    --bind $VOLUME:/root/.cache/huggingface \
    --env HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    $VLLM_IMAGE_LOC \
    --model $INFERENCE_MODEL \
    --port $VLLM_PORT \
    --max-model-len $VLLM_MAX_MODEL_LEN \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTIL \
    --max-num-batched-tokens $VLLM_MAX_BATCHED_TOKENS \
    --reasoning-parser openai_gptoss \
    --max-num-seqs $VLLM_MAX_NUM_SEQS \
    $EXTRA_ARGS &


VLLM_PID=$!


wait_for_port 127.0.0.1 $VLLM_PORT

echo "${VLLM_HOSTNAME}:${VLLM_PORT}" > "$VLLM_ENDPOINT_FILE"
echo "READY: ${VLLM_HOSTNAME}:${VLLM_PORT}"

wait $VLLM_PID

