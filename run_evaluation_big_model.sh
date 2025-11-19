#!/bin/bash

LOGDIR="logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
MODE=$([ -n "${LIMIT:-}" ] && echo "TRIAL" || echo "FULL")
LOGFILE="$LOGDIR/run_bigger_models_${MODE}_$TIMESTAMP.log"
ERRFILE="$LOGDIR/run_bigger_models_${MODE}_${TIMESTAMP}_stderr.log"

exec > >(tee -a "$LOGFILE")
exec 2> >(tee -a "$ERRFILE" >&2)

export TORCHDYNAMO_DISABLE=1

retrieved_json_paths=(
    "lm_eval/retrieved_docs/triviaqa/_IVFPQ.65536.64.256/reranked/triviaqa::olmes_q_retrieved_results::_IVFPQ.65536.64.256.1::k50_reranked.json"
    "lm_eval/retrieved_docs/triviaqa/_IVFPQ.65536.64.256/triviaqa::olmes_q_retrieved_results_::_IVFPQ.65536.64.256.1.json"
)

models=(
    "Qwen/Qwen3-235B-A22B-FP8"
)

TASKS="triviaqa"
NUM_FEWSHOT=5
BATCH_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
VLLM_COMMON_ARGS="quantization=fp8,kv_cache_dtype=fp8_e4m3,gpu_memory_utilization=0.90,max_model_len=4096,tensor_parallel_size=4,pipeline_parallel_size=2,add_bos_token=True,enforce_eager=True"

for retrieved_json_path in "${retrieved_json_paths[@]}"; do
    export RETRIEVED_JSON_PATH="$retrieved_json_path"
    echo "--------------------------------------"
    echo " Using retrieved file: $RETRIEVED_JSON_PATH"
    echo "--------------------------------------"

    for model in "${models[@]}"; do
        echo "============================"
        echo " Running model: $model "
        echo "============================"

        MODEL_ARGS="pretrained=$model,$VLLM_COMMON_ARGS"
        if [[ "$model" == "google/gemma-3-4b-pt" || \
            "$model" == "google/gemma-3-1b-pt" || \
            "$model" == "google/gemma-3-12b-pt" ]]; then
            MODEL_ARGS="$MODEL_ARGS,max_length=8192"
        fi

        echo "Model args: $MODEL_ARGS"
        echo "Retrieved: $RETRIEVED_JSON_PATH"

        lm-eval \
            --model vllm \
            --model_args "$MODEL_ARGS" \
            --tasks $TASKS \
            --num_fewshot $NUM_FEWSHOT \
            --batch_size $BATCH_SIZE \
            ${LIMIT:+--limit $LIMIT}

        if [ $? -ne 0 ]; then
            echo "⚠️ Error with $model (retrieved=$RETRIEVED_JSON_PATH), skipping..."
        fi
    done
done

echo "✅ All models attempted"
