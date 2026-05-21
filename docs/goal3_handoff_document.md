# Handoff Document: Armenian Tokenizer Surgery Project
## Complete Context for Goal 3 (Grafting)

---

## Project Description

Investigation of how standard multilingual tokenizers handle Armenian text, then performing "tokenizer surgery": training a better Armenian BPE/Unigram tokenizer, grafting it onto a small pretrained multilingual or English-centric language model by reinitializing its embedding table (mean-init, heuristic re-init, nearest-token init), and recovering performance through short full-model or PEFT fine-tuning on an Armenian corpus.

**Team size:** 4 people, 3 full days. We are on Day 2.

---

## Goal 1 — Tokenizer Fertility Analysis (COMPLETE)

Measured tokenizer fertility across 9 popular tokenizers on 25,621 lines of Armenian text (516,860 words from CC-100).

### Results (sorted by fertility, lower = better):

| Tokenizer | Type | Vocab | Armenian Tokens | Fertility | STRR | HY/EN Ratio |
|-----------|------|------:|----------------:|----------:|-----:|------------:|
| XLM-R-base | Unigram (SP) | 250,002 | 3,528 | 2.18 | 0.450 | 1.62x |
| mT5-small | Unigram (SP) | 250,100 | 2,274 | 2.85 | 0.117 | 1.91x |
| Gemma-2-2B | BPE (SP) | 256,000 | 246 | 4.49 | 0.100 | 3.55x |
| Qwen2.5-0.5B | BPE | 151,665 | 0 | 7.83 | 0.059 | 6.07x |
| LLaMA-2-7B | BPE (SP) | 32,000 | 29 | 8.53 | 0.004 | 5.80x |
| Mistral-v0.3 | BPE (SP) | 32,768 | 28 | 8.57 | 0.004 | 5.96x |
| mGPT-Armenian | BPE (SP) | 100,000 | 0 | 12.43 | 0.003 | 9.84x |
| LLaMA-3-8B | BPE (tiktoken) | 128,256 | 0 | 12.46 | 0.005 | 9.92x |
| GPT-2 | BPE | 50,257 | 0 | 14.26 | 0.004 | 11.20x |

### Key findings:
- 6.5x spread between best (XLM-R, 2.18) and worst (GPT-2, 14.26)
- Both top tokenizers use Unigram (SentencePiece), not BPE
- mGPT-Armenian has 0 Armenian vocab tokens despite being "Armenian fine-tuned" — validates the project thesis
- LLaMA-3 is worse than LLaMA-2 for Armenian (12.46 vs 8.53) despite 4x larger vocab
- English-centric tokenizers impose a 6-11x cost penalty on Armenian text

### Metric definitions:
- **Fertility** = tokens per word (lower = better)
- **STRR** = single token retention rate (% of words kept as 1 token, higher = better)
- **HY/EN Ratio** = Armenian fertility / English fertility
- **Severe Fragmentation** = % of words needing 5+ tokens

---

## Goal 2 — Custom Armenian Tokenizer Training & Evaluation (COMPLETE)

Trained 6 custom tokenizers using SentencePiece on a 5M-sentence sample reservoir-sampled from the cleaned CC-100 Armenian corpus (4.48GB, 12.1M clean lines). All evaluated on a held-out 25,611-line / 536,051-word Armenian sample.

### Evaluation Results — All 6 Custom Tokenizers Beat XLM-R:

| Model | Fertility | Bytes/Tok | STRR | Severe% | UNK% |
|-------|----------:|----------:|-----:|--------:|-----:|
| **bpe_32k** | **1.67** | 8.72 | **0.683** | 2.5% | 0% |
| **unigram_32k** | **1.68** | 8.67 | 0.682 | 2.7% | 0% |
| **bpe_16k** | **1.83** | 7.97 | 0.611 | 3.5% | 0% |
| **unigram_16k** | **1.83** | 7.95 | 0.620 | 3.9% | 0% |
| **bpe_8k** | **2.06** | 7.08 | 0.518 | 5.2% | 0% |
| **unigram_8k** | **2.06** | 7.06 | 0.538 | 6.0% | 0% |
| XLM-R (prev. best) | 2.18 | 6.65 | 0.450 | 7.3% | 0% |
| mT5 | 2.85 | 5.09 | 0.117 | 8.4% | 0% |
| Qwen2.5 (graft target) | 7.83 | 1.85 | 0.059 | 69.2% | 0% |
| GPT-2 (worst) | 14.26 | 1.02 | 0.004 | 84.7% | 0% |

**Key result: Even our smallest 8k tokenizer (fertility 2.06) beats XLM-R (2.18), which has a 250k vocabulary.** Our 32k models achieve fertility 1.67 — a 23% improvement over the previous best. BPE and Unigram perform nearly identically at each vocab size.

### Vocabulary Composition:

| Model | Vocab | Armenian Tokens | Armenian % | Byte Fallback | Special | Other |
|-------|------:|----------------:|-----------:|--------------:|--------:|------:|
| unigram_8k | 8,000 | 7,230 | 90.4% | 0 | 261 | 509 |
| unigram_16k | 16,000 | 14,784 | 92.4% | 0 | 261 | 955 |
| unigram_32k | 32,000 | 29,562 | 92.4% | 0 | 261 | 2,177 |
| bpe_8k | 8,000 | 7,178 | 89.7% | 0 | 261 | 561 |
| bpe_16k | 16,000 | 14,685 | 91.8% | 0 | 261 | 1,054 |
| bpe_32k | 32,000 | 29,641 | 92.6% | 0 | 261 | 2,098 |

All models: 5 special tokens (pad=0, unk=1, bos=2, eos=3, mask=4), zero UNK rate via byte_fallback=True.

### Model Ranking for Grafting:

The automated ranking (weighted: 40% fertility, 30% STRR, 20% compression, 10% vocab efficiency) produced:

| Rank | Model | Score | Fertility | STRR | Vocab |
|-----:|-------|------:|----------:|-----:|------:|
| 1 | bpe_32k | 0.6442 | 1.67 | 0.683 | 32,000 |
| 2 | unigram_32k | 0.6415 | 1.68 | 0.682 | 32,000 |
| 3 | unigram_16k | 0.6133 | 1.83 | 0.620 | 16,000 |
| 4 | bpe_16k | 0.6119 | 1.83 | 0.611 | 16,000 |
| 5 | unigram_8k | 0.5964 | 2.06 | 0.538 | 8,000 |
| 6 | bpe_8k | 0.5916 | 2.06 | 0.518 | 8,000 |

**Note on ranking vs practical choice:** The ranking favors bpe_32k on pure metrics, but for grafting there's a tradeoff: 32k tokens means 32k new embeddings to learn during recovery fine-tuning (more GPU time, more data needed). 16k models add half as many new embeddings with only a small fertility penalty (1.83 vs 1.67). Consider experimenting with both 16k and 32k if time permits.

### Training Parameters:
- character_coverage: 0.9999
- byte_fallback: True
- normalization: NFKC
- split_digits: True
- max_sentencepiece_length: 24 bytes (up to 12 Armenian chars per token)
- Training data: 5M sentences reservoir-sampled (SEED=42) from 12.1M clean lines
- num_threads: 8, shuffle_input_sentence=True

---

## Goal 3 — What Needs to Be Done (YOUR TASK)

### Objective
Graft the trained Armenian tokenizer onto Qwen2.5-0.5B by:
1. Extending its vocabulary with new Armenian tokens
2. Reinitializing the embedding table using 3 strategies
3. Evaluating initial perplexity before fine-tuning

### Base model: Qwen2.5-0.5B
Chosen because:
- Smallest (0.5B params) = fastest surgery and fine-tuning
- Zero Armenian tokens = cleanest surgery (no conflicting fragments)
- Ungated (no HuggingFace auth needed on cloud VM)
- Good English performance (fertility 1.29)
- 152k vocab has room for new tokens
- Modern architecture (RoPE, GQA, SwiGLU)

### Armenian tokenizer for grafting
Primary: Use the top-ranked model from Goal 2 evaluation. The automated ranking selected **bpe_32k** but **unigram_16k** is also a strong practical candidate (fewer new embeddings to learn, only small fertility penalty). Consider running the grafting experiment with both and comparing.

### Strategy: Vocabulary Extension (not Replacement)
ADD new Armenian tokens to the existing Qwen2.5 vocab rather than replacing it entirely. This preserves all English capability and requires initializing only the new token embeddings.

### Three embedding initialization strategies to compare:

**A. Mean-init (baseline):** new_embedding = mean(all old embeddings) + small random noise. Simplest approach. All new tokens start near-identical.

**B. Heuristic-init (FOCUS method, Dobler & de Melo 2023):** For each new Armenian token, encode its surface form using the OLD tokenizer (which byte-fallbacks it), then average those old subtoken embeddings. Semantically informed starting point.

**C. Nearest-init:** Compute heuristic embedding as target, find nearest existing token by cosine similarity, copy its embedding. Avoids averaging artifacts.

### Compute setup
Use Google Cloud with A100 40GB GPU. The setup guide is in the project files.
- Machine type: a2-highgpu-1g (1x A100 40GB, 12 vCPUs, 85GB RAM)
- Boot disk: Deep Learning VM with CUDA 12.4, 200GB SSD
- Cost: ~$3.67/hour on-demand, ~$1.10/hour spot
- Request GPU quota FIRST (takes approval time)
- Use tmux so training survives SSH disconnection
- STOP THE VM when done

### Expected outputs from Goal 3:
- 3 grafted model directories (mean_init/, heuristic_init/, nearest_init/)
- Perplexity comparison table (before fine-tuning)
- Fertility comparison: original Qwen2.5 vs extended tokenizer
- Token count reduction demo on sample Armenian text
- goal3_results.json

---

## Goal 4 — Recovery Fine-tuning (AFTER GOAL 3)

LoRA fine-tuning on Armenian corpus using the 3 grafted models as starting points. Compare which init strategy recovers fastest. Corpus: hy_clean.txt (4.48GB, 12.1M lines).

---

## Files in This Repository

| File | Purpose |
|------|---------|
| `goal_3_lightweight_adaptation/goal3_grafting.ipynb` | Complete grafting notebook |
| `docs/gcloud_a100_setup_guide.md` | Step-by-step Google Cloud setup |
| `models/tokenizers/hy_unigram_8k.model` + `.vocab` | Trained 8k Unigram tokenizer |
| `models/tokenizers/hy_unigram_16k.model` + `.vocab` | Trained 16k Unigram tokenizer |
| `models/tokenizers/hy_unigram_32k.model` + `.vocab` | Trained 32k Unigram tokenizer |
| `models/tokenizers/hy_bpe_8k.model` + `.vocab` | Trained 8k BPE tokenizer |
| `models/tokenizers/hy_bpe_16k.model` + `.vocab` | Trained 16k BPE tokenizer |
| `models/tokenizers/hy_bpe_32k.model` + `.vocab` | Trained 32k BPE tokenizer (top-ranked) |

### External files needed for full reruns:
- `data/sample/hy_sample_raw.txt` is included as the held-out Armenian evaluation sample.
- `hy_clean.txt` (4.48 GB) is not committed; use the cleaned corpus archive for Goal 4 and upload via GCS bucket if running on a VM.

---

## Technical Notes

- Armenian Unicode range: U+0530-U+058F (main block), U+FB13-U+FB17 (ligatures)
- Armenian characters are 2 bytes each in UTF-8
- SentencePiece space marker: Unicode U+2581
- Qwen2.5-0.5B: hidden_size=896, vocab_size=151,665, tied word embeddings
- When resizing embeddings, both input embedding table and LM head must grow
- Since Qwen2.5 has tied embeddings, updating input embeddings auto-updates the LM head

## Corpus Files Reference

| File | Size | Lines | Description |
|------|------|------:|-------------|
| hy.txt | 5.91 GB | 25.3M | Raw CC-100 Armenian |
| hy_clean.txt | 4.48 GB | 12.1M | Cleaned (deduped, filtered) |
| hy_clean.jsonl | ~4.5 GB | 12.1M | Same, JSONL format |
| `data/sample/hy_sample_raw.txt` | 7.2 MB | 30,000 | Raw sample for evaluation |
| hy_train_sample.txt | ~2 GB | 5M | Reservoir-sampled training data |
