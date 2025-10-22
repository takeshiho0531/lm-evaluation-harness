#!/bin/bash

LOGDIR="logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
MODE=$([ -n "${LIMIT:-}" ] && echo "TRIAL" || echo "FULL")
LOGFILE="$LOGDIR/run_smaller_models_${MODE}_$TIMESTAMP.log"
ERRFILE="$LOGDIR/run_smaller_models_${MODE}_${TIMESTAMP}_stderr.log"
exec > >(tee -a "$LOGFILE")

export TORCHDYNAMO_DISABLE=1

models=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "google/gemma-3-270m"
    "google/gemma-3-1b-pt"
    "google/gemma-3-4b-pt"
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-3B"
    "tiiuae/Falcon3-1B-Base"
    "tiiuae/Falcon3-3B-Base"
)

TASKS="nq_open"
NUM_FEWSHOT=5
BATCH_SIZE=2
DEVICE="cuda:0"

for model in "${models[@]}"; do
    echo "============================"
    echo " Running model: $model "
    echo "============================"

    MODEL_ARGS="pretrained=$model"
    if [[ "$model" == "google/gemma-3-4b-pt" ]]; then
        MODEL_ARGS="$MODEL_ARGS,max_length=8192"
    fi

    echo "Model args: $MODEL_ARGS"
    lm-eval \
        --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks $TASKS \
        --num_fewshot $NUM_FEWSHOT \
        --batch_size $BATCH_SIZE \
        --device $DEVICE ${LIMIT:+--limit $LIMIT}

    if [ $? -ne 0 ]; then
        echo "⚠️ Error with $model, skipping..."
    fi

done

echo "✅ All models attempted."
