# NLP Course Goal Mapping

This note maps the repository layout to the course project goals and explains what was preserved for GitHub submission.

## Goal 1: Baseline Tokenizer Fertility

Folder: `goal_1_tokenizer_fertility/`

The fertility analysis script evaluates Armenian tokenization quality across existing model tokenizers. The preserved JSON output records corpus statistics and tokenizer-level metrics such as fertility, bytes per token, single-token word rate, and severe fragmentation.

## Goal 2: Armenian Tokenizer Training and Evaluation

Folder: `goal_2_tokenizer_surgery/`

The notebooks train and evaluate Armenian-specific BPE and Unigram tokenizers at 8k, 16k, and 32k vocabulary sizes. The trained tokenizer files are stored in `models/tokenizers/`, and evaluation summaries are stored in `results/goal2/`.

## Goal 3: Tokenizer Grafting and Initialization

Folder: `goal_3_lightweight_adaptation/`

The notebooks explore grafting Armenian tokenizer artifacts into a base-model workflow and comparing initialization strategies. Grafted tokenizer/config outputs are preserved under `models/grafted_tokenizers/`. Full model weight files were skipped because each was roughly 994 MB.

## Goal 4: LoRA Adaptation and Evaluation

Folder: `goal_4_evaluation/`

The fine-tuning notebooks compare mean, heuristic, and nearest-neighbor initialization strategies after LoRA adaptation. Compact adapter folders are preserved in `models/lora_adapters/`; repeated trainer checkpoints are excluded.

## Goal 5: Final Synthesis and Analysis

Folder: `goal_5_generation_or_analysis/`

The final notebook and JSON output consolidate the results into project-level metrics and figures. The generated plots are stored in the repository-level `figures/` directory.

## Files Excluded from the Public Repository

The repository intentionally excludes duplicate working folders, notebook checkpoints, OS metadata, zip archives, screenshots not needed for the report, LaTeX build byproducts, large raw training samples, temporary checkpoint folders, and full base-model `.safetensors` outputs.
