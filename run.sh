CUDA_VISIBLE_DEVICES=0 \
python -m lm_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=bfloat16,tensor_parallel_size=1 \
  --log_samples \
  --device cuda \
  --batch_size=1 \
  --verbosity ERROR \
  --output_path "./test" \
  --include_path industrial_tasks/ \
  --trust_remote_code \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks agro_bench,aqua_bench,med_bench \
  --limit 5