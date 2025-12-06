#!/bin/bash

LOGDIR="logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
MODE=$([ -n "${LIMIT:-}" ] && echo "TRIAL" || echo "FULL")
LOGFILE="$LOGDIR/run_models_${MODE}_$TIMESTAMP.log"
ERRFILE="$LOGDIR/run_models_${MODE}_${TIMESTAMP}_stderr.log"
exec > >(tee -a "$LOGFILE")

export TORCHDYNAMO_DISABLE=1

retrieved_json_paths=(
    "lm_eval/retrieved_docs/triviaqa/compactds/triviaqa::olmes_q_retrieved_results_::_IVFPQ.65536.64.256.5.json"
)

models=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "google/gemma-3-270m"
    "google/gemma-3-1b-pt"
    "google/gemma-3-4b-pt"
    "google/gemma-3-12b-pt"
)

TASKS="nq_open"
NUM_FEWSHOT=5
GEN_KWARGS='{"temperature":0, "do_sample":false,"top_p":1,"max_new_tokens":20}'
BATCH_SIZE=2
DEVICE="cuda:0"

for retrieved_json_path in "${retrieved_json_paths[@]}"; do
    export RETRIEVED_JSON_PATH="$retrieved_json_path"
    echo "--------------------------------------"
    echo " Using retrieved file: $RETRIEVED_JSON_PATH"
    echo "--------------------------------------"

    for model in "${models[@]}"; do
        echo "============================"
        echo " Running model: $model "
        echo "============================"

        MODEL_ARGS="pretrained=$model"
        if [[ "$model" == "google/gemma-3-4b-pt" || \
            "$model" == "google/gemma-3-1b-pt" || \
            "$model" == "google/gemma-3-12b-pt" ]]; then
            MODEL_ARGS="$MODEL_ARGS,max_length=8192"
        fi

        echo "Model args: $MODEL_ARGS"
        echo "Retrieved: $RETRIEVED_JSON_PATH"

        lm-eval \
            --model hf \
            --model_args "$MODEL_ARGS" \
            --tasks $TASKS \
            --num_fewshot $NUM_FEWSHOT \
            --batch_size $BATCH_SIZE \
            --device $DEVICE \
            --gen_kwargs "$GEN_KWARGS" ${LIMIT:+--limit $LIMIT}

        if [ $? -ne 0 ]; then
            echo "⚠️ Error with $model (retrieved=$RETRIEVED_JSON_PATH), skipping..."
        fi
    done
done

echo "✅ All retrieved sets and models attempted."

