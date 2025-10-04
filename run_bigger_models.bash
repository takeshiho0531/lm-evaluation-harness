#!/bin/bash

LOGDIR="logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
MODE=$([ -n "${LIMIT:-}" ] && echo "TRIAL" || echo "FULL")
LOGFILE="$LOGDIR/run_bigger_models_${MODE}_$TIMESTAMP.log"
ERRFILE="$LOGDIR/run_bigger_models_${MODE}_${TIMESTAMP}_stderr.log"
exec > >(tee -a "$LOGFILE")

export RETRIEVED_JSONS_DIR="lm_eval/retrieved_docs/mmlu/compactds/_IVFPQ.65536.64.256.5" 
export TORCHDYNAMO_DISABLE=1

models=(
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "google/gemma-3-12b-pt"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-14B"
    "tiiuae/Falcon3-7B-Base"
    "tiiuae/Falcon3-10B-Base"
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
)

TASKS="mmlu"
NUM_FEWSHOT=5
GEN_KWARGS='{"temperature":0, "do_sample":false,"top_p":1,"max_new_tokens":20}'
BATCH_SIZE=2
DEVICE="cuda:0"

for model in "${models[@]}"; do
    echo "============================"
    echo " Running model: $model "
    echo "============================"

    MODEL_ARGS="pretrained=$model"
    if [[ "$model" == "google/gemma-3-12b-pt" ]]; then
        MODEL_ARGS="$MODEL_ARGS,max_length=8192"
    fi

    echo "Model args: $MODEL_ARGS"
    lm-eval \
        --model hf \
        --model_args "$MODEL_ARGS" \
        --tasks $TASKS \
        --num_fewshot $NUM_FEWSHOT \
        --gen_kwargs "$GEN_KWARGS" \
        --batch_size $BATCH_SIZE \
        --device $DEVICE ${LIMIT:+--limit $LIMIT}

    if [ $? -ne 0 ]; then
        echo "⚠️ Error with $model, skipping..."
    fi

done

echo "✅ All models attempted."
