# MERA_Industrial
Отраслевая ветка MERA

## How to run
use script `run.sh`
or directly:

```bash
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
```


## Build submission ZIP from logs
1) 
```bash
mkdir -p code_tasks
cat > code_tasks/config.yaml << 'YAML'
BENCHMARK_STORAGE: null
SUBMISSIONS_DIR: null
YAML
```

2) use script `run_log_to_submit.sh`
or directly

```bash
python scripts/log_to_submission.py \
  --outputs_dir ./test/Qwen__Qwen2.5-0.5B-Instruct \
  --dst_dir submission_zip \
  --model_args "pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=bfloat16,tensor_parallel_size=1"
```