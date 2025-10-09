# MERA Industrial

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/mera-industrial-logo.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/mera-industrial-logo-black.svg">
    <img alt="MERA Industrial" src="docs/mera-industrial-logo.svg" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
    <a href="https://opensource.org/licenses/MIT">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <a href="https://github.com/MERA-Evaluation/MERA_Industrial/tree/main">
    <img alt="Release" src="https://img.shields.io/badge/release-v1.0.0-blue">
    </a>

</p>

<h2 align="center">
    <p> MERA Induscrial: A Unified Framework for Evaluating Industrial tasks.
</p>
</h2>

## 🚀 About

**MERA Industrial** brings together a domain-specific collection of evaluation tasks under one roof. Built on top of the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) (v0.4.9), it enables researchers and practitioners to:

- **Compare models** on identical tasks and metrics
- **Reproduce results** with fixed prompts and few-shot settings
- **Submit** standardized ZIP archives for leaderboard integration


## 🔍 Datasets Overview

| Set         | Task Name                    | Metrics                | Size  | Prompts | Skills                                                        |
| ----------- | ---------------------------- | ---------------------- | ----- | ------- | ------------------------------------------------------------- |
| **Private** | **ruTXTMedQFundamental**     | ExactMatch, F1         | 4590  | 10      | Anatomy, Biochemistry, Bioorganic Chemistry, Biophysics, Clinical Laboratory Diagnostics, Faculty Surgery, General Chemistry, General Surgery, Histology, Hygiene, Microbiology, Normal Physiology, Parasitology, Pathological Anatomy, Pathological physiology, Pharmacology, Propaedeutics in Internal Medicine |
| **Private** | **ruTXTAgroBench**           | ExactMatch, F1         | 2642   | 10      | Botany, Forage Production and Grassland Management, Land Reclamation, General Genetics, General Agriculture, Fundamentals of Plant Breeding, Plant Production, Seed Production and Seed Science, Agricultural Systems in Various Agricultural Landscapes, Crop Cultivation Technologies |
| **Private** | **ruTXTAquaBench**           | ExactMatch, F1         | 992   | 10      | Industrial aquaculture; Ichthyopathology: veterinary medicine, prevention and optimization of fish farming technologies; Feeding fish and other aquatic organisms; Mariculture, Breeding crayfish and shrimp, Artificial pearl cultivation. |


## 🛠 Getting Started <a name="evaluation"></a>

### Clone the repository with submodule

First, you need to clone the MERA_CODE repository and load the submodule:

```bash
### Go to the folder where the repository will be cloned ###
mkdir mera_industrial
cd mera_industrial

### Clone & install core libs ###
git clone --recurse-submodules https://github.com/MERA-Evaluation/MERA_Industrial.git
cd MERA_Industrial
```

### Installing dependencies

**Remote Scoring**: quick setup for cloud-based scoring — install only core dependencies, run the evaluation, and submit the resulting ZIP archive to our website to get the score. 

Install lm-eval library and optional packages for evaluations:

```bash
### Install lm-eval ###
cd lm-evaluation-harness
pip install -e .

### Install additional libs for models evaluation [Optional] ###
# vLLM engine
pip install -e ".[vllm]"
# API scoring
pip install -e ".[api]"

### Go to MERA_Industrial folder ###
cd ../
```

### Running evaluations

We have prepared the script that launches evaluations via `lm-eval` library and packs the evaluation logs into zip archive:

```bash
### Run evaluation and pack logs ###
bash scripts/run_evaluation.sh \
 --model vllm \
 --model_args "pretrained=Qwen/Qwen2.5-0.5B-Instruct,tensor_parallel_size=1" \
 --output_path "./results/Qwen2.5-0.5B-Instruct"
```

More details on `run_evaluation.sh` usage may be obtained by:
```bash
bash scripts/run_evaluation.sh --help
```

<details>
<summary>
How it works inside...
</summary>

```bash
### run lm-eval
lm_eval \
    --model vllm \  # use vLLM engile
    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=bfloat16 \  # model init details
    --log_samples \  # save eval logs (model generations)
    --device cuda \  # inference on cuda
    --batch_size=1 \  # use batch_size=1
    --verbosity ERROR \  # only essential prints
    --output_path="./results/Qwen2.5-0.5B-Instruct" \  # where to save the logs
    --include_path industrial_tasks/ \  # include out custom tasks
    --trust_remote_code \  # may be needed for some models
    --apply_chat_template \  # use chat template of the model
    --fewshot_as_multiturn \  # along with apply_chat_template, matters only for num_fewshot > 0
    --tasks agro_bench,aqua_bench,med_bench  # eval tasks
  
### pack logs into zip archive
python scripts/log_to_submission.py \
    --outputs_dir ./results/Qwen2.5-0.5B-Instruct \
    --dst_dir ./results/Qwen2.5-0.5B-Instruct \
    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=bfloat16

### there would appear results/Qwen2.5-0.5B-Instruct_submission.zip archive ready for submission
```

</details>

## 📁 Repository Structure

```text
MERA_CODE/
├── industrial_tasks/            # Code for each task
├── datasets/                    # Task descriptions, metadata, readme
├── docs/                        # Additional documentation and design notes
 ├── dataset_formatting.md       # Dataset formatting requirements
 ├── model_scoring.md            # How to use lm-eval to evaluate the LMs
 ├── task_codebase.md            # How to add a new task to the codebase
├── lm-evaluation-harness/       # Submodule (codebase)
└── scripts/                     # Helpers: add tasks, run evaluations, and scoring
```


## 💪 How to Join the Leaderboard

Follow these steps to see your model on the Leaderboard:

1. **Run Remote Scoring**  
 Evaluate the benchmark in the **Remote Scoring** regime (see [🛠 Getting Started](#evaluation) above). Pay attention that for **private** tasks we do not provide golden answers, so no local scoring is provided.
 > You’ll end up with a logs folder **and** a ready-to-submit zip archive like `Qwen2.5-0.5B-Instruct_submission.zip`.

2. **Submit on the website**  
 Head over to [Create Submission](https://mera.a-ai.ru/ru/industrial/submits/create), upload the archive, and move on to the form.

3. **Fill in Model Details**  
 Provide accurate information about the model and evaluation. These details are crucial for reproducibility—if something is missing, administrators may ping you (or your Submission might be rejected).

4. **Wait for Scoring** ⏳  
 Scoring usually wraps up in **~10-15 minutes**.

1. **Publish your result**  
 Once scoring finishes, click **"Submit for moderation"**. After approval, your model goes **Public** and appears on the [Leaderboard](https://mera.a-ai.ru/ru/industrial/leaderboard).  

Good luck, and happy benchmarking! 🎉
   
## 📝 License

Distributed under the MIT License. See LICENSE for details.
