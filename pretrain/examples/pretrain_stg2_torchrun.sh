sed 's/=1/=8/g' /etc/mpi/hostfile > /etc/mpi/hostfile_seq

STAGE1_OUTPUT_DIR=/code/onerec_pretrain/model_output/stg1_opt_utils_big
MODEL_DIR=${STAGE1_OUTPUT_DIR}/step2000/global_step2000/converted
OUTPUT_DIR=/code/onerec_pretrain/model_output/stg2_opt_utils_big
mkdir -p $OUTPUT_DIR
mkdir -p /tmp/_wids_cache

HOSTFILE=/etc/mpi/hostfile_seq
NNODES=${NNODES:-$(wc -l < "${HOSTFILE}")}

set -x

SCRIPT_FILE=$(readlink -f $0)
echo `date '+%Y-%m-%d %H:%M:%S'` >> $OUTPUT_DIR/task_info.log
echo "script: ${SCRIPT_FILE}" >> $OUTPUT_DIR/task_info.log
echo "=========================" >> $OUTPUT_DIR/task_info.log

echo "Output: $OUTPUT_DIR"

export PYTHONPATH=$PWD:$PYTHONPATH

source set_env_torchrun.sh

TCP_NIC=$(ifconfig | grep -B1 " "$(hostname -i)" " | grep -o "^\w*")

MASTER_ADDR=${MASTER_ADDR:-$(awk 'NR==1{print $1}' "${HOSTFILE}")}
MASTER_PORT=${MASTER_PORT:-8499}
RDZV_ID=${RDZV_ID:-onerec}

if [ -z "${NODE_RANK}" ]; then
    HOST_SHORT=$(hostname)
    HOST_IP=$(hostname -I | awk '{print $1}')
    if ! NODE_RANK=$(awk -v host="${HOST_SHORT}" -v ip="${HOST_IP}" \
        '($1==host || $1==ip){print NR-1; found=1; exit} END{if (!found) exit 1}' \
        "${HOSTFILE}"); then
        echo "NODE_RANK not set and current host not found in ${HOSTFILE}." >&2
        echo "Set NODE_RANK manually for torchrun multi-node." >&2
        exit 1
    fi
fi

if [ -z "${NPROC_PER_NODE}" ]; then
    NPROC_PER_NODE=$(awk -F'slots=' 'NR==1{if (NF>1) print $2}' "${HOSTFILE}" | awk '{print $1}')
    NPROC_PER_NODE=${NPROC_PER_NODE:-8}
fi

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=${TCP_NIC}
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NET_OVERHEAD=1000
export NCCL_IB_TIMEOUT=20
export LD_PRELOAD=${LD_PRELOAD}
export NO_COLOR=1
export TERM=dumb
export COLORTERM=0
export PYTHONIOENCODING=utf-8
export LD_LIBRARY_PATH=${LIBRARY_PATH}
export MASTER_ADDR
export MASTER_PORT
export TOKENIZERS_PARALLELISM=false

torchrun \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id "${RDZV_ID}" \
    with_nccl_local_env \
    bash scripts/numa_runner.sh python3 recipes/train_qwen3.py \
        --model_dir $MODEL_DIR \
        --output_dir $OUTPUT_DIR \
        --dataset_config examples/dataset_config/pretrain.json \
        --use_tie_weights \
        --start_optimize_embedding_index 151669 \
        --model_class Qwen3ForCausalLM \
        --monitor_datasource_loss \
        --monitor_datasource_cnt \
        --max_length 32768 \
        --learning_rate 2e-4 \
        --min_lr 1e-4 \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 200 \
        --num_training_steps 5000 \
        --save_checkpoint_per_step 50 \
        --minibatch_size 16384 \
        --logging_per_step 5 \
        --use_fp32_weight \
        --seed 19260817 \
        --enable_profiler \
        --enable_gradient_checkpointing \
        --use_chunked_loss_computer \
    > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &

    # --resume_from $STAGE1_OUTPUT_DIR/step2000 \
    # --resume_from_tag global_step2000 \
