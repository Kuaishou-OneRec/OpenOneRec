#!/bin/bash
# Single Node Ray Initialization Script
# Usage: bash init_ray.sh <HEAD_NODE_IP> <PORT> <RANK>
#   HEAD_NODE_IP: IP address of the head node
#   PORT: Ray port (default: 6379)
#   RANK: Node rank (0 for head, >0 for workers)

set -e

# Parse arguments
HEAD_NODE_IP=${1:-"127.0.0.1"}
PORT=${2:-6379}
RANK=${3:-0}

# Configuration
NUM_CPUS=${NUM_CPUS:-""}
NUM_GPUS=${NUM_GPUS:-""}
OBJECT_STORE_MEMORY=${OBJECT_STORE_MEMORY:-""}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"benchmark"}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(hostname): $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(hostname): $1"
}

# Activate conda environment
if [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/root/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

if command -v conda &> /dev/null; then
    conda activate ${CONDA_ENV_NAME} 2>/dev/null || log_warn "Could not activate conda env: ${CONDA_ENV_NAME}"
fi

# Build ray start command options
RAY_OPTS=""
if [ -n "${NUM_CPUS}" ]; then
    RAY_OPTS="${RAY_OPTS} --num-cpus=${NUM_CPUS}"
fi
if [ -n "${NUM_GPUS}" ]; then
    RAY_OPTS="${RAY_OPTS} --num-gpus=${NUM_GPUS}"
fi
if [ -n "${OBJECT_STORE_MEMORY}" ]; then
    RAY_OPTS="${RAY_OPTS} --object-store-memory=${OBJECT_STORE_MEMORY}"
fi

# Stop existing Ray instance
ray stop --force 2>/dev/null || true
sleep 2

# Start Ray
if [ "${RANK}" -eq 0 ]; then
    log_info "Starting Ray HEAD node on port ${PORT}..."
    ray start --head --port=${PORT} ${RAY_OPTS}
else
    log_info "Starting Ray WORKER node, connecting to ${HEAD_NODE_IP}:${PORT}..."
    ray start --address=${HEAD_NODE_IP}:${PORT} ${RAY_OPTS}
fi

sleep 3

# Check status
log_info "Ray node started. Checking status..."
ray status
