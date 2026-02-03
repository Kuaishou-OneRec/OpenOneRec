#!/bin/bash
# RecIF Data Processing Script
# Generate all pretrain and SFT data

set -e

# ============== Task Selection ==============
# Comment out tasks you don't want to run

# Pretrain tasks
RUN_PRETRAIN_VIDEO_REC=1
RUN_PRETRAIN_USER_PROFILE=1
RUN_PRETRAIN_ITEM_UNDERSTAND=1

# SFT tasks
RUN_SFT_VIDEO_REC=1
RUN_SFT_INTERACTIVE_REC=1
RUN_SFT_LABEL_COND_REC=1
RUN_SFT_LABEL_PRED=1
RUN_SFT_AD_REC=1
RUN_SFT_PRODUCT_REC=1
RUN_SFT_ITEM_UNDERSTAND=1
RUN_SFT_REC_REASON=1

# ============== Configuration ==============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_METADATA="../../raw_data/onerec_data/onerec_bench_release.parquet"
PID2SID_MAPPING="../../raw_data/onerec_data/video_ad_pid2sid.parquet"
PRODUCT_PID2SID_MAPPING="../../raw_data/onerec_data/product_pid2sid.parquet"
CAPTION_INPUT="../../raw_data/onerec_data/pid2caption.parquet"
OUTPUT_BASE_DIR="../../output"

SEED=42

# ============== Helper Function ==============
run_task() {
    local task_type=$1
    local task_name=$2
    local script_path=$3
    shift 3
    local extra_args="$@"

    local output_file="${OUTPUT_BASE_DIR}/${task_type}_${task_name}.parquet"
    local temp_dir=$(mktemp -d)

    echo "  Output: ${output_file}"
    python3 "${script_path}" --output_dir "${temp_dir}" ${extra_args}

    if [ -f "${temp_dir}/train.parquet" ]; then
        mv "${temp_dir}/train.parquet" "${output_file}"
    fi
    rm -rf "${temp_dir}"
}

# ============== Main ==============
echo "========================================"
echo "RecIF Data Processing"
echo "========================================"
echo "Metadata: ${INPUT_METADATA}"
echo "PID2SID: ${PID2SID_MAPPING}"
echo "Caption: ${CAPTION_INPUT}"
echo "Output: ${OUTPUT_BASE_DIR}"
echo ""

mkdir -p "${OUTPUT_BASE_DIR}"

# ============== Pretrain Tasks ==============
echo "========================================"
echo "Pretrain Tasks"
echo "========================================"

if [ "${RUN_PRETRAIN_VIDEO_REC}" = "1" ]; then
    echo "[pretrain] video_rec..."
    run_task "pretrain" "video_rec" "${SCRIPT_DIR}/pretrain/video_rec.py" \
        --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}"
fi

if [ "${RUN_PRETRAIN_USER_PROFILE}" = "1" ]; then
    echo "[pretrain] user_profile..."
    run_task "pretrain" "user_profile" "${SCRIPT_DIR}/pretrain/user_profile.py" \
        --input "${INPUT_METADATA}"
fi

if [ "${RUN_PRETRAIN_ITEM_UNDERSTAND}" = "1" ]; then
    echo "[pretrain] item_understand..."
    run_task "pretrain" "item_understand" "${SCRIPT_DIR}/pretrain/item_understand.py" \
        --input "${CAPTION_INPUT}" --pid2sid "${PID2SID_MAPPING}" --seed ${SEED}
fi

# ============== SFT Tasks ==============
echo ""
echo "========================================"
echo "SFT Tasks"
echo "========================================"

if [ "${RUN_SFT_VIDEO_REC}" = "1" ]; then
    echo "[sft] video_rec..."
    run_task "sft" "video_rec" "${SCRIPT_DIR}/sft/video_rec.py" \
        --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}" --seed ${SEED}
fi

if [ "${RUN_SFT_INTERACTIVE_REC}" = "1" ]; then
    echo "[sft] interactive_rec..."
    run_task "sft" "interactive_rec" "${SCRIPT_DIR}/sft/interactive_rec.py" \
        --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}" --seed ${SEED}
fi

if [ "${RUN_SFT_LABEL_COND_REC}" = "1" ]; then
    echo "[sft] label_cond_rec..."
    run_task "sft" "label_cond_rec" "${SCRIPT_DIR}/sft/label_cond_rec.py" \
        --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}" --seed ${SEED}
fi

if [ "${RUN_SFT_LABEL_PRED}" = "1" ]; then
    echo "[sft] label_pred..."
    run_task "sft" "label_pred" "${SCRIPT_DIR}/sft/label_pred.py" \
        --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}" --seed ${SEED}
fi

if [ "${RUN_SFT_AD_REC}" = "1" ]; then
    echo "[sft] ad_rec..."
    run_task "sft" "ad_rec" "${SCRIPT_DIR}/sft/ad_rec.py" \
        --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}" --seed ${SEED}
fi

if [ "${RUN_SFT_PRODUCT_REC}" = "1" ]; then
    echo "[sft] product_rec..."
    run_task "sft" "product_rec" "${SCRIPT_DIR}/sft/product_rec.py" \
        --input "${INPUT_METADATA}" --pid2sid "${PID2SID_MAPPING}" \
        --product_pid2sid "${PRODUCT_PID2SID_MAPPING}" --seed ${SEED}
fi

if [ "${RUN_SFT_ITEM_UNDERSTAND}" = "1" ]; then
    echo "[sft] item_understand..."
    run_task "sft" "item_understand" "${SCRIPT_DIR}/sft/item_understand.py" \
        --input "${CAPTION_INPUT}" --pid2sid "${PID2SID_MAPPING}" --seed ${SEED}
fi

if [ "${RUN_SFT_REC_REASON}" = "1" ]; then
    echo "[sft] rec_reason..."
    run_task "sft" "rec_reason" "${SCRIPT_DIR}/sft/rec_reason.py" \
        --input "${INPUT_METADATA}"
fi

# ============== Summary ==============
echo ""
echo "========================================"
echo "Summary"
echo "========================================"
ls -lh "${OUTPUT_BASE_DIR}"/*.parquet 2>/dev/null || echo "No parquet files found"
echo ""
echo "Done!"
