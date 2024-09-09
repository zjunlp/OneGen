# ATTENTION: you need to modify the config.json in model_path. 
# Specically, change the `architectures` fileds to specific class name.
# e.g. LlamaModelForCausalLM

CUDA_ID="$1"
mode="$2"
model_path="$3"
model_tag="$4"
dataset_folder="$5"
ndocs="$6"
execute_env_setup="${7}"
execute_generate_score="${8}"
tokenizer_path="${9:-$model_path}"
embed_batch_size="${10:-64}"

echo "CUDA_ID=${CUDA_ID}"
echo "model_tag=${model_tag}"
echo "mode=${mode}"
echo "model_path=${model_path}"
echo "dataset_folder=${dataset_folder}"
echo "ndocs=${ndocs}"
echo "tokenizer_path=${tokenizer_path}"
echo "embed_batch_size=${embed_batch_size}"

# install environment
if [ "$execute_env_setup" == "true" ]; then
    bash self_rag/vllm_env.sh
fi

# generate score file
CUDA_VISIBLE_DEVICES=$CUDA_ID python generate_score.py \
    --model_path $model_path \
    --output_folder "${dataset_folder}" \
    --batch_size $embed_batch_size \
    --tokenizer_path $tokenizer_path
# generate score file
if [ "$execute_generate_score" == "true" ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_ID python generate_score.py \
        --model_path $model_path \
        --output_folder "${dataset_folder}" \
        --batch_size $embed_batch_size \
        --tokenizer_path $tokenizer_path
fi

# evaluation
echo "PopQA Evaluation Start."
CUDA_VISIBLE_DEVICES=$CUDA_ID python run_short_form.py \
    --model_name $model_path \
    --input_file "${dataset_folder}/popqa_longtail_w_gs.jsonl" \
    --mode $mode --max_new_tokens 100 \
    --threshold 0.2 \
    --output_file "temp-output/popqa-${model_tag}-${mode}.json" \
    --metric match --ndocs $ndocs --use_groundness --use_utility --use_seqscore
echo "PopQA Evaluation Done."

echo "Public Health Evaluation Start."
CUDA_VISIBLE_DEVICES=$CUDA_ID python run_short_form.py \
  --model_name $model_path \
  --input_file "${dataset_folder}/health_claims_processed.jsonl" \
  --max_new_tokens 100 \
  --threshold 0.2 --output_file "temp-output/public-health-${model_tag}-${mode}.json" \
  --metric match --ndocs $ndocs \
  --use_groundness --use_utility --use_seqscore \
  --task fever \
  --mode $mode
echo "Public Health Evaluation Done."

echo "ARC QA Evaluation Start."
CUDA_VISIBLE_DEVICES=$CUDA_ID python run_short_form.py \
  --model_name $model_path \
  --input_file "${dataset_folder}/arc_challenge_processed.jsonl" \
  --max_new_tokens 50 --threshold 0.2 \
  --output_file "temp-output/arcqa-${model_tag}-${mode}.json" \
  --metric match --ndocs $ndocs --use_groundness --use_utility --use_seqscore \
  --task arc_c \
  --mode $mode
echo "ARC QA Evaluation Done."

echo "TriviaQA Evaluation Start."
CUDA_VISIBLE_DEVICES=$CUDA_ID python run_short_form.py \
    --model_name $model_path \
    --input_file "${dataset_folder}/triviaqa_test_w_gs.jsonl" \
    --mode $mode --max_new_tokens 100 \
    --threshold 0.2 \
    --output_file "temp-output/triviaqa-${model_tag}-${mode}.json" \
    --metric match --ndocs $ndocs --use_groundness --use_utility --use_seqscore 
echo "TriviaQA Evaluation Done."

