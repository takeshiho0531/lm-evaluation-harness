#!/bin/bash

LOGDIR="logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

if [[ -z "${LIMIT:-}" ]]; then
    MODE="FULL"
elif [[ "$LIMIT" == "5000" ]]; then
    MODE="SUBSET"
else
    MODE="TRIAL"
fi

LOGFILE="$LOGDIR/run_models_${MODE}_$TIMESTAMP.log"
ERRFILE="$LOGDIR/run_models_${MODE}_${TIMESTAMP}_stderr.log"
exec > >(tee -a "$LOGFILE")
exec 2> >(tee -a "$ERRFILE" >&2)

export TORCHDYNAMO_DISABLE=1

models=(
    "Qwen/Qwen3-30B-A3B"
    "google/gemma-3-27b-pt"
)

TASKS="mmlu"
NUM_FEWSHOT=5
GEN_KWARGS='{"temperature":0, "do_sample":false,"top_p":1,"max_new_tokens":20}'
BATCH_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1,2


for model in "${models[@]}"; do
    echo "============================"
    echo " Running model: $model "
    echo "============================"

    MODEL_ARGS="pretrained=$model,parallelize=True"

    if [[ "$model" == google/gemma-3-*b-pt ]]; then
        MODEL_ARGS="$MODEL_ARGS,max_length=8192"
    fi


    echo "Model args: $MODEL_ARGS"
    lm-eval \
        --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks $TASKS \
        --num_fewshot $NUM_FEWSHOT \
        --batch_size $BATCH_SIZE \
        --gen_kwargs "$GEN_KWARGS" ${LIMIT:+--limit $LIMIT}
    if [ $? -ne 0 ]; then
        echo "⚠️ Error with $model, skipping..."
    fi
done

echo "✅ All models attempted"
