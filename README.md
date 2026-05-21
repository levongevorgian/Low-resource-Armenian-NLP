# Low-Resource Armenian NLP: Tokenization Surgery and Lightweight Adaptation

This repository contains the final code, notebooks, model artifacts, figures, results, and report for an NLP course project on improving Armenian tokenization efficiency through tokenizer analysis, Armenian-specific tokenizer training, vocabulary grafting, and lightweight LoRA adaptation.

## Project Overview

Many general-purpose language models tokenize Armenian inefficiently because their vocabularies contain few Armenian-specific subword units. This project studies that issue empirically, trains Armenian-focused SentencePiece tokenizers, grafts a custom tokenizer into a Qwen2.5-0.5B workflow, and evaluates recovery with parameter-efficient fine-tuning.

The repository is organized around the five project goals. Large intermediate corpora, full base-model checkpoints, temporary training checkpoints, caches, screenshots, duplicate working folders, and LaTeX build artifacts are intentionally excluded.

## Motivation

Tokenizer fertility affects sequence length, compute cost, and the amount of text a model can process within a fixed context window. For Armenian, weak tokenizer coverage can cause severe word fragmentation. The project tests whether tokenizer surgery can reduce that fragmentation and whether lightweight adaptation can recover useful language-model likelihood after adding Armenian-oriented tokens.

## Research Goals

1. Measure Armenian tokenizer fertility and fragmentation across existing tokenizers.
2. Train and evaluate Armenian-specific BPE and Unigram tokenizers.
3. Graft the best Armenian tokenizer into a base-model workflow and compare embedding initialization strategies.
4. Run LoRA recovery fine-tuning and evaluate perplexity/loss across initialization strategies.
5. Consolidate final analysis, generation diagnostics, preservation checks, and figures.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── report.pdf
├── data/sample/                       # Small Armenian evaluation sample
├── docs/                              # Handoff notes, setup notes, and goal mapping
├── figures/                           # Final figures used for reporting and analysis
├── goal_1_tokenizer_fertility/        # Fertility analysis script and Goal 1 notes/results
├── goal_2_tokenizer_surgery/          # Tokenizer training and evaluation notebooks
├── goal_3_lightweight_adaptation/     # Tokenizer grafting / initialization notebooks
├── goal_4_evaluation/                 # LoRA fine-tuning and evaluation notebooks
├── goal_5_generation_or_analysis/     # Final synthesis/evaluation notebook
├── models/grafted_tokenizers/         # Tokenizer/config artifacts from grafting experiments
├── models/lora_adapters/              # Compact LoRA adapter artifacts
├── models/tokenizers/                 # Trained Armenian tokenizer model/vocab files
├── results/                           # JSON outputs grouped by goal
└── scripts/                           # Supporting utility scripts
```

## Goal 1: Tokenizer Fertility

Goal 1 measures Armenian tokenization efficiency for existing tokenizers. The main script computes fertility, bytes per token, single-token word retention, severe fragmentation, and Armenian vocabulary coverage.

Main files:

- `goal_1_tokenizer_fertility/fertility_analysis.py`
- `goal_1_tokenizer_fertility/fertility_results.json`
- `results/goal1/fertility_results.json`

## Goal 2: Tokenizer Surgery

Goal 2 trains Armenian-specific SentencePiece tokenizers at 8k, 16k, and 32k vocabulary sizes and compares BPE and Unigram variants against baseline tokenizers.

Main files:

- `goal_2_tokenizer_surgery/goal2_train_tokenizers.ipynb`
- `goal_2_tokenizer_surgery/goal2_eval_only.ipynb`
- `models/tokenizers/`
- `results/goal2/goal2_eval_results.json`

## Goal 3: Tokenizer Grafting and Initialization

Goal 3 explores vocabulary grafting and embedding initialization strategies: mean initialization, heuristic/FOCUS-style initialization, and nearest-token initialization. Full base-model weight files are not included because they are too large for a normal GitHub repository.

Main files:

- `goal_3_lightweight_adaptation/goal3_grafting.ipynb`
- `goal_3_lightweight_adaptation/goal3_grafting_with_results.ipynb`
- `models/grafted_tokenizers/`
- `results/goal3/goal3_results.json`

## Goal 4: Evaluation and LoRA Fine-Tuning

Goal 4 runs parameter-efficient adaptation experiments and compares post-fine-tuning perplexity and loss across initialization strategies. Compact LoRA adapter artifacts are included; repeated trainer checkpoints are excluded.

Main files:

- `goal_4_evaluation/goal4_lora_finetuning_h100.ipynb`
- `goal_4_evaluation/goal4_lora_finetuning_heuristic_colab.ipynb`
- `goal_4_evaluation/goal4_lora_finetuning_nearest_init_rtx_pro_6000.ipynb`
- `models/lora_adapters/`
- `results/goal4/goal4_results.json`
- `results/goal4/goal4_global_benchmarks.json`

## Goal 5: Final Analysis

Goal 5 consolidates the earlier stages into final tables, figures, and diagnostics. It summarizes the fertility, tokenizer, grafting, adaptation, and qualitative generation results.

Main files:

- `goal_5_generation_or_analysis/goal5_evaluation.ipynb`
- `results/goal5/goal5_final_results.json`
- `figures/`

## Setup

Use Python 3.10 or newer. A GPU is recommended for model grafting and LoRA fine-tuning notebooks, but the fertility analysis script and result inspection can run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some notebooks load Hugging Face models that may require authentication or license access. Run `huggingface-cli login` if a gated model cannot be downloaded.

## Reproducing the Main Steps

Run the project in goal order. Goal 1 can be run directly on the included sample:

```bash
python goal_1_tokenizer_fertility/fertility_analysis.py \
  --input data/sample/hy_sample_raw.txt \
  --output results/goal1/fertility_results.json
```

Then open and run the notebooks as needed:

```text
goal_2_tokenizer_surgery/goal2_train_tokenizers.ipynb
goal_2_tokenizer_surgery/goal2_eval_only.ipynb
goal_3_lightweight_adaptation/goal3_grafting_with_results.ipynb
goal_4_evaluation/goal4_lora_finetuning_h100.ipynb
goal_5_generation_or_analysis/goal5_evaluation.ipynb
```

Exact end-to-end reproduction requires the full cleaned Armenian corpus `hy_clean.txt`, which is not included because of size. Several notebooks were originally run across local, Colab, H100, and RTX environments, so hardware-specific path cells may need small adjustments before rerunning.

## Main Outputs

- Goal 1 baseline fertility metrics: `results/goal1/fertility_results.json`
- Goal 2 custom tokenizer evaluation: `results/goal2/goal2_eval_results.json`
- Goal 3 grafting metrics: `results/goal3/goal3_results.json`
- Goal 4 LoRA benchmark metrics: `results/goal4/`
- Goal 5 consolidated results: `results/goal5/goal5_final_results.json`
- Final figures: `figures/`
- Final report: `report.pdf`

## Figures

The `figures/` directory contains the final visualization assets used for analysis and reporting, including baseline fertility/compression plots, custom tokenizer comparisons, grafting efficiency, perplexity recovery, adaptation curves, and final Goal 5 summaries.

## Report

The final report is stored at the repository root:

```text
report.pdf
```

## Team and Collaboration Note

This was a collaborative NLP course project. The repository preserves the final code, notebooks, results, figures, compact model artifacts, and report needed for review and public submission while omitting duplicated scratch work and large intermediate files.

## Citation and Acknowledgments

This project uses open-source NLP tooling including Hugging Face Transformers, PyTorch, SentencePiece, PEFT, Datasets, NumPy, Matplotlib, tqdm, and tabulate. If you reuse this work, please cite the relevant upstream libraries and model providers used in your experiments.
