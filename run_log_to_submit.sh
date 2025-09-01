python scripts/log_to_submission.py \
  --outputs_dir ./test/Qwen__Qwen2.5-0.5B-Instruct \
  --dst_dir submission_zip \
  --model_args "pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=bfloat16,tensor_parallel_size=1"