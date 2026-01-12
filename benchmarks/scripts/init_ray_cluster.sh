#!/bin/bash
# Multi-node Ray Cluster Initialization Script
# Usage: bash init_ray_cluster.sh [--stop]
#   --stop: Stop Ray on all nodes instead of starting

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=${SCRIPT_DIR}

# Configuration
PORT=${RAY_PORT:-6379}
HOSTFILE=${HOSTFILE:-"/etc/mpi/hostfile"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"benchmark"}
LOG_DIR="${PROJECT_DIR}/logs/ray"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Generate conda initialization command that works with both anaconda and miniconda
get_conda_init_cmd() {
    cat << 'EOF'
for conda_sh in /root/miniconda3/etc/profile.d/conda.sh \
                /root/anaconda3/etc/profile.d/conda.sh \
                $HOME/miniconda3/etc/profile.d/conda.sh \
                $HOME/anaconda3/etc/profile.d/conda.sh \
                /opt/conda/etc/profile.d/conda.sh; do
    [ -f "$conda_sh" ] && source "$conda_sh" && break
done
EOF
}

# Function to stop Ray on all nodes
stop_cluster() {
    log_info "Stopping Ray on all nodes..."

    if [ ! -f "${HOSTFILE}" ]; then
        log_warn "Hostfile not found, stopping local Ray only"
        ray stop --force 2>/dev/null || true
        return
    fi

    ALL_NODES=$(awk '!a[$1]++ {print $1}' ${HOSTFILE})

    for node in ${ALL_NODES}; do
        log_info "Stopping Ray on ${node}..."
        ssh -n ${node} "$(get_conda_init_cmd) && conda activate ${CONDA_ENV_NAME} && ray stop --force" 2>/dev/null &
    done

    wait
    log_info "Ray stopped on all nodes"
}

# Function to start Ray cluster
start_cluster() {
    # Check hostfile
    if [ ! -f "${HOSTFILE}" ]; then
        log_error "Hostfile not found: ${HOSTFILE}"
        log_info "Please create a hostfile with one IP per line"
        log_info "Example:"
        echo "  192.168.1.100"
        echo "  192.168.1.101"
        echo "  192.168.1.102"
        exit 1
    fi

    # Get head node (first line)
    HEAD_NODE=$(awk 'NR==1 {print $1}' ${HOSTFILE})
    ALL_NODES=$(awk '!a[$1]++ {print $1}' ${HOSTFILE})

    log_info "Head node: ${HEAD_NODE}"
    log_info "Ray port: ${PORT}"
    log_info "Conda env: ${CONDA_ENV_NAME}"
    echo ""
    log_info "Nodes in cluster:"
    echo "${ALL_NODES}"
    echo ""

    # Create log directory
    mkdir -p "${LOG_DIR}"

    # Stop existing Ray instances first
    log_info "Stopping any existing Ray instances..."
    stop_cluster
    sleep 3

    # Start head node first (synchronously)
    log_info "Starting Ray HEAD on ${HEAD_NODE}..."
    ssh -n ${HEAD_NODE} "CONDA_ENV_NAME=${CONDA_ENV_NAME} bash ${SCRIPT_DIR}/init_ray.sh ${HEAD_NODE} ${PORT} 0" \
        > "${LOG_DIR}/ray_${HEAD_NODE}.log" 2>&1

    if [ $? -ne 0 ]; then
        log_error "Failed to start Ray HEAD. Check ${LOG_DIR}/ray_${HEAD_NODE}.log"
        exit 1
    fi
    log_info "Ray HEAD started successfully"

    # Wait for head to be ready
    sleep 5

    # Start worker nodes (asynchronously)
    rank=1
    for node in ${ALL_NODES}; do
        if [ "${node}" == "${HEAD_NODE}" ]; then
            continue
        fi

        log_info "Starting Ray WORKER on ${node} (rank ${rank})..."
        ssh -n ${node} "CONDA_ENV_NAME=${CONDA_ENV_NAME} bash ${SCRIPT_DIR}/init_ray.sh ${HEAD_NODE} ${PORT} ${rank}" \
            > "${LOG_DIR}/ray_${node}.log" 2>&1 &
        rank=$((rank + 1))
    done

    # Wait for all workers
    log_info "Waiting for all workers to join..."
    wait
    sleep 3

    # Check cluster status
    echo ""
    log_info "Ray cluster initialization complete!"
    log_info "Logs saved to: ${LOG_DIR}/"
    echo ""
    log_info "Cluster status:"
    ssh -n ${HEAD_NODE} "$(get_conda_init_cmd) && conda activate ${CONDA_ENV_NAME} && ray status"
}

# Main
case "${1}" in
    --stop)
        stop_cluster
        ;;
    *)
        start_cluster
        ;;
esac
