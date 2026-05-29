# Low-Resource Armenian NLP: Tokenization Surgery and Lightweight Adaptation

![Project Preview](preview.png)

A research-oriented NLP project focused on improving Armenian tokenization efficiency through tokenizer analysis, Armenian-specific tokenizer training, vocabulary grafting, and lightweight LoRA adaptation.

---

## Project Overview

Many general-purpose language models tokenize Armenian inefficiently because their vocabularies contain few Armenian-specific subword units. This project studies that issue empirically, trains Armenian-focused SentencePiece tokenizers, grafts a custom tokenizer into a Qwen2.5-0.5B workflow, and evaluates recovery through parameter-efficient fine-tuning.

The repository is organized around five major research goals:

* Tokenizer fertility analysis
* Armenian tokenizer training
* Vocabulary grafting
* LoRA recovery fine-tuning
* Final evaluation and generation diagnostics

---

# Quick Links

| Resource     | Link                                   |
| ------------ | -------------------------------------- |
| Final Report | [`report.pdf`](report.pdf)             |
| Figures      | [`figures/`](figures/)                 |
| Results      | [`results/`](results/)                 |
| Sample Data  | [`data/sample/`](data/sample/)         |
| Requirements | [`requirements.txt`](requirements.txt) |

---

# Research Goals

## Goal 1 — Tokenizer Fertility

Measure Armenian tokenizer fertility and fragmentation across existing tokenizers.

### Main Files

* [`goal_1_tokenizer_fertility/fertility_analysis.py`](goal_1_tokenizer_fertility/fertility_analysis.py)
* [`results/goal1/fertility_results.json`](results/goal1/fertility_results.json)

---

## Goal 2 — Tokenizer Surgery

Train Armenian-specific BPE and Unigram SentencePiece tokenizers at multiple vocabulary sizes.

### Main Files

* [`goal_2_tokenizer_surgery/goal2_train_tokenizers.ipynb`](goal_2_tokenizer_surgery/goal2_train_tokenizers.ipynb)
* [`goal_2_tokenizer_surgery/goal2_eval_only.ipynb`](goal_2_tokenizer_surgery/goal2_eval_only.ipynb)
* [`results/goal2/goal2_eval_results.json`](results/goal2/goal2_eval_results.json)

---

## Goal 3 — Lightweight Adaptation

Explore vocabulary grafting and embedding initialization strategies for integrating Armenian tokens into Qwen2.5-0.5B.

### Main Files

* [`goal_3_lightweight_adaptation/goal3_grafting.ipynb`](goal_3_lightweight_adaptation/goal3_grafting.ipynb)
* [`goal_3_lightweight_adaptation/goal3_grafting_with_results.ipynb`](goal_3_lightweight_adaptation/goal3_grafting_with_results.ipynb)
* [`results/goal3/goal3_results.json`](results/goal3/goal3_results.json)

---

## Goal 4 — LoRA Recovery Fine-Tuning

Run parameter-efficient adaptation experiments and compare post-fine-tuning perplexity across initialization strategies.

### Main Files

* [`goal_4_evaluation/goal4_lora_finetuning_h100.ipynb`](goal_4_evaluation/goal4_lora_finetuning_h100.ipynb)
* [`goal_4_evaluation/goal4_lora_finetuning_heuristic_colab.ipynb`](goal_4_evaluation/goal4_lora_finetuning_heuristic_colab.ipynb)
* [`results/goal4/`](results/goal4/)

---

## Goal 5 — Final Analysis

Consolidate final tables, figures, diagnostics, and qualitative generation results.

### Main Files

* [`goal_5_generation_or_analysis/goal5_evaluation.ipynb`](goal_5_generation_or_analysis/goal5_evaluation.ipynb)
* [`results/goal5/goal5_final_results.json`](results/goal5/goal5_final_results.json)
* [`figures/`](figures/)

---

# Repository Structure

| Folder                           | Description                    |
| -------------------------------- | ------------------------------ |
| `data/sample/`                   | Armenian evaluation samples    |
| `docs/`                          | Notes and documentation        |
| `figures/`                       | Final visualizations and plots |
| `goal_1_tokenizer_fertility/`    | Fertility analysis             |
| `goal_2_tokenizer_surgery/`      | Tokenizer training             |
| `goal_3_lightweight_adaptation/` | Vocabulary grafting            |
| `goal_4_evaluation/`             | LoRA adaptation                |
| `goal_5_generation_or_analysis/` | Final evaluation               |
| `models/`                        | Tokenizers and LoRA adapters   |
| `results/`                       | JSON outputs and metrics       |
| `scripts/`                       | Utility scripts                |

---

# Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some notebooks require Hugging Face authentication:

```bash
huggingface-cli login
```

---

# Running the Project

Example Goal 1 reproduction:

```bash
python goal_1_tokenizer_fertility/fertility_analysis.py \
  --input data/sample/hy_sample_raw.txt \
  --output results/goal1/fertility_results.json
```

Then run the notebooks in goal order.

---

# Main Outputs

| Output                     | Location         |
| -------------------------- | ---------------- |
| Baseline fertility metrics | `results/goal1/` |
| Tokenizer evaluation       | `results/goal2/` |
| Grafting metrics           | `results/goal3/` |
| LoRA benchmarks            | `results/goal4/` |
| Final evaluation           | `results/goal5/` |
| Final report               | `report.pdf`     |

---

# Figures

The `figures/` directory contains:

* Fertility comparison plots
* Tokenizer evaluation figures
* Grafting efficiency plots
* Perplexity recovery curves
* Final summary visualizations

---

# Team

Collaborative NLP course project focused on low-resource Armenian language modeling and tokenizer adaptation.

---

# Acknowledgments

This project uses:

* Hugging Face Transformers
* PyTorch
* SentencePiece
* PEFT
* NumPy
* Matplotlib
* tqdm
* tabulate
