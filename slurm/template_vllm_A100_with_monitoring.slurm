#!/bin/bash
#SBATCH --job-name=run_vllm_A100


##HOPPER
#SBATCH --qos=gpu --partition=gpu --gres=gpu:A100.80gb:4

#SBATCH --output=/scratch/%u/slurm_logs/%x-%j-%N.out
#SBATCH --error=/scratch/%u/slurm_logs/%x-%j-%N.err

#SBATCH --mem=400G
#SBATCH --time=5-00:00:00

#SBATCH --nodes=1 --ntasks-per-node=1 

module load singularity
module load gnu9
module load gcc
cd /scratch/pbhanda2/projects/mmv4

source slurm/envvars

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

# Get the actual hostname of the allocated node
VLLM_HOSTNAME=$(hostname -f)
echo "VLLM server will run on: $VLLM_HOSTNAME"

VLLM_ENDPOINT_DIR="/scratch/pbhanda2/projects/mmv4/vllm_endpoints"
mkdir -p "$VLLM_ENDPOINT_DIR"
VLLM_ENDPOINT_FILE="${VLLM_ENDPOINT_DIR}/vllm_endpoint_${SLURM_JOB_ID}.txt"


export TORCH_CUDA_ARCH_LIST="8.0"

VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1024}
VLLM_MAX_BATCHED_TOKENS=${VLLM_MAX_BATCHED_TOKENS:-524288}
VLLM_GPU_MEMORY_UTIL=${VLLM_GPU_MEMORY_UTIL:-0.95}
VLLM_ENABLE_CHUNKED_PREFILL=${VLLM_ENABLE_CHUNKED_PREFILL:-1}
VLLM_DISABLE_LOG_REQUESTS=${VLLM_DISABLE_LOG_REQUESTS:-1}
VLLM_DISABLE_FRONTEND_METRICS=${VLLM_DISABLE_FRONTEND_METRICS:-0}

EXTRA_ARGS=""
if [ "$VLLM_ENABLE_CHUNKED_PREFILL" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --enable-chunked-prefill"
fi
if [ "$VLLM_DISABLE_LOG_REQUESTS" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --disable-log-requests"
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
    --max-num-seqs $VLLM_MAX_NUM_SEQS \
    $EXTRA_ARGS &

VLLM_PID=$!


wait_for_port 127.0.0.1 $VLLM_PORT

echo "${VLLM_HOSTNAME}:${VLLM_PORT}" > "$VLLM_ENDPOINT_FILE"
echo "READY: ${VLLM_HOSTNAME}:${VLLM_PORT}"

# --- Prometheus + Grafana (same node)
DEFAULT_PROM_IMAGE="/scratch/pbhanda2/singularity_container/prometheus.sif"
DEFAULT_GRAFANA_IMAGE="/scratch/pbhanda2/singularity_container/grafana.sif"
PROM_IMAGE=${PROM_IMAGE:-$DEFAULT_PROM_IMAGE}
GRAFANA_IMAGE=${GRAFANA_IMAGE:-$DEFAULT_GRAFANA_IMAGE}

if [ ! -f "$PROM_IMAGE" ]; then
  echo "Prometheus image not found at $PROM_IMAGE"; exit 1;
fi
if [ ! -f "$GRAFANA_IMAGE" ]; then
  echo "Grafana image not found at $GRAFANA_IMAGE"; exit 1;
fi

PROM_CONFIG_SRC="vllm_concurrency_test/prometheus/prometheus.yml"
PROM_CONFIG_TMP="/tmp/prometheus_${SLURM_JOB_ID}.yml"
PROM_DATA_DIR=${PROM_DATA_DIR:-/tmp/prometheus_${SLURM_JOB_ID}}
mkdir -p "$PROM_DATA_DIR"

VLLM_TARGET="127.0.0.1:${VLLM_PORT}"
sed "s/__VLLM_TARGET__/${VLLM_TARGET}/" "$PROM_CONFIG_SRC" > "$PROM_CONFIG_TMP"

singularity run \
  --bind "$PROM_CONFIG_TMP":/etc/prometheus/prometheus.yml \
  --bind "$PROM_DATA_DIR":/prometheus \
  $PROM_IMAGE \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.listen-address=0.0.0.0:9090 &

PROM_PID=$!

GRAFANA_DATA_DIR=${GRAFANA_DATA_DIR:-/tmp/grafana_${SLURM_JOB_ID}}
mkdir -p "$GRAFANA_DATA_DIR"

singularity run \
  --bind "$GRAFANA_DATA_DIR":/var/lib/grafana \
  --bind "$PWD/vllm_concurrency_test/grafana/provisioning":/etc/grafana/provisioning \
  --bind "$PWD/vllm_concurrency_test/grafana/dashboards":/etc/grafana/provisioning/dashboards \
  --env GF_SECURITY_ADMIN_PASSWORD=${GF_ADMIN_PASSWORD:-admin} \
  --env GF_SERVER_HTTP_ADDR=0.0.0.0 \
  --env GF_SERVER_HTTP_PORT=3000 \
  $GRAFANA_IMAGE &

GRAFANA_PID=$!

echo "Prometheus on http://${VLLM_HOSTNAME}:9090"
echo "Grafana on http://${VLLM_HOSTNAME}:3000 (admin / ${GF_ADMIN_PASSWORD:-admin})"

wait $VLLM_PID

