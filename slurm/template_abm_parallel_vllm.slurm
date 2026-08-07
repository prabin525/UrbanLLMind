#!/bin/bash

#SBATCH --job-name=mini-world
#SBATCH --partition=normal
#SBATCH --output=/scratch/%u/slurm_logs/%x-%j-%N.out
#SBATCH --error=/scratch/%u/slurm_logs/%x-%j-%N.err
#SBATCH --mem-per-cpu=20G
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# --- Modules / env
module purge
module load gnu10
module load openmpi/4.1.2-4a

cd /scratch/pbhanda2/projects/mmv4
source myenv/bin/activate
# Ensure package import path is installed in this venv once:
# pip install -e /scratch/pbhanda2/projects/mmv4

# --- Runtime controls (Task 0)
# SHUTDOWN_VLLM_SERVER_ON_END:
#   false (default) -> keep launched vLLM jobs running after ABM exits
#   true            -> cancel vLLM jobs launched by this script during cleanup
SHUTDOWN_VLLM_SERVER_ON_END="${SHUTDOWN_VLLM_SERVER_ON_END:-true}"
SHUTDOWN_VLLM_SERVER_ON_END=$(printf '%s' "$SHUTDOWN_VLLM_SERVER_ON_END" | tr '[:upper:]' '[:lower:]')
case "$SHUTDOWN_VLLM_SERVER_ON_END" in
    true|false) ;;
    *)
        echo "Invalid SHUTDOWN_VLLM_SERVER_ON_END='$SHUTDOWN_VLLM_SERVER_ON_END' (expected true|false)."
        exit 1
        ;;
esac

# VLLM_ENDPOINTS_OVERRIDE:
#   optional comma-separated endpoint list, for example:
#   http://host1:8000/v1,http://host2:8000/v1
#   If set, skip vLLM launch/wait and use this list directly.
#   Default is preconfigured to reuse an already-running server on gpu007.
VLLM_ENDPOINTS_OVERRIDE_DEFAULT="http://gpu032.orc.gmu.edu:8080/v1"
VLLM_ENDPOINTS_OVERRIDE=$(printf '%s' "${VLLM_ENDPOINTS_OVERRIDE:-$VLLM_ENDPOINTS_OVERRIDE_DEFAULT}" | tr -d '[:space:]')

# --- Helper: wait for a single endpoint file and port
wait_for_endpoint() {
    local job_id="$1"
    local outfile="$2"
    local timeout="$3"
    local interval=10
    local elapsed=0

    echo "[$job_id] Waiting up to ${timeout}s for endpoint file $outfile ..."
    while [ ! -f "$outfile" ] && [ "$elapsed" -lt "$timeout" ]; do
        sleep "$interval"
        elapsed=$((elapsed + interval))
        echo "[$job_id] ... waited ${elapsed}s for endpoint file"
    done

    if [ ! -f "$outfile" ]; then
        echo "TIMEOUT $job_id file" >> "$VLLM_RESULTS_FILE"
        return
    fi

    local endpoint
    endpoint=$(cat "$outfile")
    rm -f "$outfile"

    IFS=':' read -r host port <<< "$endpoint"
    elapsed=0
    echo "[$job_id] File found (${endpoint}); waiting up to ${timeout}s for port ${host}:${port} ..."
    while ! nc -z "$host" "$port" && [ "$elapsed" -lt "$timeout" ]; do
        sleep "$interval"
        elapsed=$((elapsed + interval))
        echo "[$job_id] ... waited ${elapsed}s for port ${host}:${port}"
    done

    if ! nc -z "$host" "$port"; then
        echo "TIMEOUT $job_id port" >> "$VLLM_RESULTS_FILE"
        return
    fi

    echo "OK $job_id http://${endpoint}/v1" >> "$VLLM_RESULTS_FILE"
}

# --- vLLM orchestration defaults
NUM_VLLM_SERVERS=${NUM_VLLM_SERVERS:-4}
# VLLM_TEMPLATE=${VLLM_TEMPLATE:-/scratch/pbhanda2/projects/mmv4/slurm/template_vllm_A100_with_monitoring_HIGH_reasoning.slurm}
VLLM_TEMPLATE=${VLLM_TEMPLATE:-/scratch/pbhanda2/projects/mmv4/slurm/template_vllm_A100_with_monitoring.slurm}
# VLLM_TEMPLATE=${VLLM_TEMPLATE:-/scratch/pbhanda2/projects/mmv4/slurm/template_vllm_A100_lora_with_monitoring.slurm}
VLLM_ENDPOINT_DIR="/scratch/pbhanda2/projects/mmv4/vllm_endpoints"
VLLM_RESULTS_FILE="/tmp/vllm_results.${SLURM_JOB_ID}"
MAX_WAIT_TIME=${MAX_WAIT_TIME:-600}

declare -a VLLM_ENDPOINTS_ARR
declare -a VLLM_JOB_IDS
declare -a FAILED_JOBS

if [ -n "$VLLM_ENDPOINTS_OVERRIDE" ]; then
    if echo "$VLLM_ENDPOINTS_OVERRIDE" | grep -Eq '(^,|,,|,$)'; then
        echo "Invalid VLLM_ENDPOINTS_OVERRIDE='$VLLM_ENDPOINTS_OVERRIDE' (empty endpoint detected)."
        exit 1
    fi
    VLLM_ENDPOINT_LIST="$VLLM_ENDPOINTS_OVERRIDE"
    echo "Using VLLM_ENDPOINTS_OVERRIDE. Skipping vLLM launch and warm-up wait."
else
    mkdir -p "$VLLM_ENDPOINT_DIR"

    echo "Launching $NUM_VLLM_SERVERS vLLM server job(s)..."
    for i in $(seq 1 "$NUM_VLLM_SERVERS"); do
        JOB_ID=$(sbatch --parsable "$VLLM_TEMPLATE")
        echo "vLLM job $i submitted with ID: $JOB_ID"
        VLLM_JOB_IDS+=("$JOB_ID")
    done

    # --- Wait in parallel
    rm -f "$VLLM_RESULTS_FILE"
    for JOB_ID in "${VLLM_JOB_IDS[@]}"; do
        ENDPOINT_FILE="$VLLM_ENDPOINT_DIR/vllm_endpoint_${JOB_ID}.txt"
        wait_for_endpoint "$JOB_ID" "$ENDPOINT_FILE" "$MAX_WAIT_TIME" &
    done

    wait

    # --- Collect results
    if [ -f "$VLLM_RESULTS_FILE" ]; then
        while read -r status job url; do
            if [ "$status" = "OK" ]; then
                echo "vLLM job $job endpoint: $url"
                VLLM_ENDPOINTS_ARR+=("$url")
            else
                echo "vLLM job $job timed out waiting for endpoint."
                FAILED_JOBS+=("$job")
            fi
        done < "$VLLM_RESULTS_FILE"
        rm -f "$VLLM_RESULTS_FILE"
    fi

    # Cancel only failed jobs that this script launched.
    for CANCEL_ID in "${FAILED_JOBS[@]}"; do
        scancel "$CANCEL_ID" >/dev/null 2>&1
    done

    if [ ${#VLLM_ENDPOINTS_ARR[@]} -eq 0 ]; then
        echo "No vLLM endpoints became available. Exiting."
        exit 1
    fi

    VLLM_ENDPOINT_LIST=$(printf ",%s" "${VLLM_ENDPOINTS_ARR[@]}")
    VLLM_ENDPOINT_LIST=${VLLM_ENDPOINT_LIST:1}
fi

export VLLM_ENDPOINTS="$VLLM_ENDPOINT_LIST"
echo "Using VLLM endpoints: $VLLM_ENDPOINTS"

# --- Run ABM
sim_module="mini_world.main"
ncores=1
config_file="${1:-input_GABM_SERVER.yaml}"

RUN_CMD="srun --export=ALL,VLLM_ENDPOINTS=${VLLM_ENDPOINT_LIST} -n ${ncores} python -m ${sim_module} ${config_file}"
echo "Running: ${RUN_CMD}"
eval "${RUN_CMD}"
rc=$?

# --- Cleanup
if [[ $rc -ne 0 ]]; then
    echo "Command failed: ${RUN_CMD}"
fi

if [ ${#VLLM_JOB_IDS[@]} -gt 0 ]; then
    if [ "$SHUTDOWN_VLLM_SERVER_ON_END" = "true" ]; then
        echo "SHUTDOWN_VLLM_SERVER_ON_END=false: canceling launched vLLM jobs."
        for CANCEL_ID in "${VLLM_JOB_IDS[@]}"; do
            scancel "$CANCEL_ID" >/dev/null 2>&1
        done
    else
        echo "SHUTDOWN_VLLM_SERVER_ON_END=false: preserving launched vLLM jobs."
    fi
fi

exit $rc

