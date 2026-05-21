# Goal 1 — Tokenizer Fertility Analysis Report (Final)
## Armenian Tokenizer Surgery Project

**Date:** May 16, 2026
**Corpus:** CC-100 Armenian (`hy.txt`), 30k line raw sample → 25,621 lines after Armenian content filtering
**Corpus Stats:** 516,860 words · 4,104,249 characters · 7,497,625 bytes
**Tokenizers Tested:** 9/9 (all loaded successfully)

---

## Executive Summary

We measured tokenizer fertility across 9 widely-used tokenizers spanning English-centric, large-vocab general, explicitly multilingual, and Armenian-specific models. The results reveal a **6.5x spread** in Armenian tokenization efficiency between the best (XLM-R at 2.18 tokens/word) and worst (GPT-2 at 14.26 tokens/word). English-centric tokenizers impose a **6–11x cost penalty** on Armenian text compared to English, directly motivating the tokenizer surgery planned in Goals 2–4. Two major surprises emerged: LLaMA-3's 128k vocabulary performs *worse* than LLaMA-2's 32k on Armenian, and mGPT-Armenian — a model explicitly fine-tuned for Armenian — has zero Armenian vocabulary tokens.

---

## Complete Results (9 Tokenizers, Sorted by Fertility)

| Rank | Tokenizer | Type | Vocab | HY Tokens | Fertility ↓ | Bytes/Tok | STRR ↑ | Severe% | EN Fert. | HY/EN |
|------|-----------|------|------:|----------:|------------:|----------:|-------:|--------:|---------:|------:|
| 1 | **XLM-R-base** | Unigram (SP) | 250,002 | 3,528 (1.41%) | **2.18** | 6.65 | **0.450** | 7.3% | 1.35 | **1.62x** |
| 2 | **mT5-small** | Unigram (SP) | 250,100 | 2,274 (0.91%) | **2.85** | 5.09 | 0.117 | 8.4% | 1.50 | **1.91x** |
| 3 | Gemma-2-2B | BPE (SP) | 256,000 | 246 (0.10%) | 4.49 | 3.23 | 0.100 | 43.3% | 1.26 | 3.55x |
| 4 | Qwen2.5-0.5B | BPE | 151,665 | 0 (0%) | 7.83 | 1.85 | 0.059 | 69.2% | 1.29 | 6.07x |
| 5 | LLaMA-2-7B | BPE (SP) | 32,000 | 29 (0.09%) | 8.53 | 1.70 | 0.004 | 79.9% | 1.47 | 5.80x |
| 6 | Mistral-v0.3 | BPE (SP) | 32,768 | 28 (0.09%) | 8.57 | 1.69 | 0.004 | 79.9% | 1.44 | 5.96x |
| 7 | mGPT-Armenian | BPE (SP) | 100,000 | 0 (0%) | 12.43 | 1.17 | 0.003 | 82.3% | 1.26 | 9.84x |
| 8 | LLaMA-3-8B | BPE (tiktoken) | 128,256 | 0 (0%) | 12.46 | 1.17 | 0.005 | 82.3% | 1.26 | 9.92x |
| 9 | GPT-2 | BPE | 50,257 | 0 (0%) | 14.26 | 1.02 | 0.004 | 84.7% | 1.27 | 11.20x |

---

## Metric Definitions

- **Fertility** (tokens/word): Average number of tokens per whitespace-delimited word. Lower is better. English-optimized tokenizers typically achieve 1.2–1.5 on English text.
- **Bytes/Token**: How many UTF-8 bytes each token represents on average. Higher means more efficient compression. Armenian characters are 2 bytes each in UTF-8.
- **STRR** (Single Token Retention Rate): Proportion of words that survive as a single token. Higher indicates more vocabulary dedicated to the language.
- **Severe Fragmentation %**: Proportion of words requiring 5 or more tokens. These words are the hardest for the model to reconstruct semantically.
- **HY/EN Ratio**: Armenian fertility divided by English fertility for the same tokenizer. Directly translates to relative inference cost and context window consumption.

---

## Detailed Findings by Tier

### Tier 1 — Explicitly Multilingual (fertility < 3)

**XLM-R-base** is the clear winner with fertility 2.18 and an extraordinary STRR of 0.450, meaning 45% of Armenian words survive as single tokens. It dedicates 3,528 tokens (1.41% of its 250k vocab) to Armenian. The sample tokenization (see STEP 4 terminal output) demonstrates effective morphological chunking: it captures common suffixes and morphological boundaries, tokenizes a full surname as a single token, and compresses 7 Armenian words + punctuation into just 18 tokens. Only 7.3% of words suffer severe fragmentation.

**mT5-small** is the runner-up at 2.85 fertility. Despite a similar 250k vocab, it has fewer Armenian tokens (2,274 vs 3,528) and a much lower STRR (0.117 vs 0.450). This gap likely reflects different training data balance: XLM-R was trained on CC-100 with more balanced multilingual sampling, while mT5 used C4 which skews more heavily toward English.

Both use **Unigram (SentencePiece)** tokenization rather than BPE. This is a significant finding: Unigram's probabilistic segmentation approach appears to handle Armenian's rich agglutinative morphology more effectively than greedy BPE merges. This insight should inform our tokenizer training choices in Goal 2.

### Tier 2 — Large-Vocab with Incidental Armenian Coverage (fertility 4–8)

**Gemma-2-2B** (fertility 4.49) is the best of the non-explicitly-multilingual models. Its enormous 256k vocabulary includes 246 Armenian tokens, enough for some common subwords. STRR of 0.100 means 10% of words are single tokens, and 43.3% severe fragmentation is a major step down from mT5/XLM-R but far better than the models below.

**Qwen2.5-0.5B** (fertility 7.83) has zero Armenian-specific tokens despite its 152k vocabulary. It handles Armenian entirely through individual Unicode character tokens (one character = one token). The 6% STRR comes from very short common words. 69.2% of words are severely fragmented.

### Tier 3 — Character-Level Fallback (fertility 8–9)

**LLaMA-2-7B** (fertility 8.53) and **Mistral-v0.3** (fertility 8.57) are nearly identical. Both have 32k-class vocabularies with ~28–29 Armenian tokens, which is not even enough to cover the full 38-letter Armenian alphabet. The sample tokenization shows they have individual Armenian character tokens for most (but not all) letters, with some characters falling back to multi-byte sequences (the `??` outputs visible in the demo). 80% of words are severely fragmented.

### Tier 4 — Byte-Level Catastrophe (fertility > 12)

**mGPT-Armenian** (fertility 12.43) is the most significant finding of this analysis. Despite being explicitly marketed as an Armenian fine-tuned model (`ai-forever/mGPT-1.3B-armenian`), it has **zero Armenian vocabulary tokens** and produces fertility nearly as bad as GPT-2. The sample tokenization shows pure byte-fallback output (`??`, `??`). This confirms that the ai-forever team fine-tuned model weights on Armenian data but **never modified the tokenizer**. The base mGPT tokenizer was trained on English/Russian text and encodes Armenian as raw UTF-8 bytes.

**LLaMA-3-8B** (fertility 12.46) is a surprise. Despite expanding vocabulary from 32k (LLaMA-2) to 128k, Armenian performance is **dramatically worse** — 12.46 vs 8.53. This counterintuitive result occurs because LLaMA-3 switched from SentencePiece to tiktoken-style byte-level BPE. While LLaMA-2's SentencePiece preserved individual Armenian characters as tokens (fertility ~8.5, character-level), LLaMA-3's tiktoken merges raw bytes into multi-byte tokens optimized for English and code. Armenian UTF-8 byte sequences get merged into nonsensical cross-character groups rather than aligning with character boundaries, producing worse-than-character-level fragmentation.

**GPT-2** (fertility 14.26) is the worst, as expected. Every Armenian character becomes 2 raw byte tokens (Armenian characters = 2 bytes in UTF-8, and GPT-2's byte-level fallback has no merged byte sequences for Armenian). The bytes/token ratio of 1.02 confirms virtually zero compression — almost exactly 1 byte per token.

---

## The Tokenization Tax — Quantified

Armenian text processed by GPT-2 consumes **11.2x more tokens** than equivalent English text. Concretely:
- A 4,096-token context window holds ~3,000 English words but only ~287 Armenian words with GPT-2.
- With XLM-R's tokenizer, the same window holds ~1,879 Armenian words — a 6.5x improvement.
- Inference cost scales linearly with token count, making GPT-2 11.2x more expensive for Armenian than English.
- Even the "best" non-multilingual option (Gemma-2) costs 3.55x more for Armenian, meaning two-thirds of the context window is wasted on tokenization overhead.

---

## Key Surprise: LLaMA-3 Worse Than LLaMA-2

This finding deserves emphasis because it contradicts the assumption that "bigger vocabulary = better multilingual support." LLaMA-3's 128k vocab makes it worse for Armenian (12.46) than LLaMA-2's 32k vocab (8.53). The reason is architectural:

- LLaMA-2 (SentencePiece BPE): has individual character tokens for most Armenian letters. Each Armenian character ≈ 1 token. Character-level but at least character-aligned.
- LLaMA-3 (tiktoken BPE): merges raw bytes into multi-byte tokens. The merges were learned from English/code data, so Armenian byte sequences get split at arbitrary positions that cross character boundaries. Each Armenian character ≈ 2+ tokens.

This demonstrates that vocabulary size alone does not determine multilingual quality — the training data distribution and tokenization algorithm matter more.

---

## The mGPT-Armenian Anomaly

This finding directly validates the project's thesis. mGPT-Armenian demonstrates that:
1. Fine-tuning model weights without fixing the tokenizer is insufficient.
2. A model can "learn Armenian" at the weight level while still being crippled by byte-level tokenization.
3. Tokenizer surgery (Goal 3) combined with recovery fine-tuning (Goal 4) is the correct approach — not just more fine-tuning on a broken tokenizer.

This is mGPT-Armenian's fundamental limitation: every Armenian word costs 11–12 tokens of context and compute, regardless of how well-tuned the model weights are. Grafting a proper Armenian tokenizer onto this model could unlock performance that's already latent in its weights.

---

## Sample Tokenization Comparison

The sample sentence used for all tokenizers (from `fertility_results.json`):
> (See `sample_sentence` field in `fertility_results.json` for the original Armenian text, or STEP 4 of the terminal output.)

*Note: Armenian text may not render correctly in this report. Refer to STEP 4 of the terminal output (`fertility_analysis.py`) for the exact tokenizations with correct Armenian script. The token counts below are accurate.*

| Tokenizer | Token Count | Fragmentation Behavior |
|-----------|-------------|----------------------|
| XLM-R | 18 | Morphological subwords: captures suffixes, common roots, and full proper nouns as single tokens. Most tokens are 2–4 characters. |
| mT5 | 19 | Similar to XLM-R but slightly more fragmented. Captures common suffixes and roots as multi-character subwords. |
| Gemma-2 | 32 | Mixed: some 2–3 character Armenian subwords, but many single-character tokens. Better than character-level but far from morphological. |
| Qwen2.5 | 53 | Character-level: each Armenian character is a separate token. No subword merges for Armenian. |
| LLaMA-2 | 58 | Mostly character-level with some byte fallback. A few uppercase Armenian letters fall back to multi-byte tokens. |
| Mistral | 58 | Nearly identical to LLaMA-2. Same character-level behavior with the same byte fallback gaps. |
| mGPT-Arm. | 82 | Byte-level: raw UTF-8 byte sequences, no character alignment. Each Armenian character = ~2 byte tokens. |
| LLaMA-3 | 83 | Byte-level: despite 128k vocab, Armenian bytes are merged into cross-character groups. Worse than character-level. |
| GPT-2 | 96 | Byte-level: every Armenian character = exactly 2 byte tokens. Zero compression. |

---

## Implications for Goals 2–5

### Goal 2 (Custom Tokenizer Training)
Target: train a BPE or Unigram tokenizer on the cleaned 12M-line Armenian corpus achieving fertility below 2.0. Based on these results, **Unigram (SentencePiece) is recommended over BPE** since both top performers use Unigram. A vocabulary of 8k–16k Armenian-specific tokens should be sufficient. We should experiment with both algorithms and compare.

### Goal 3 (Grafting)
Best grafting candidates ranked by strategic value:
1. **Qwen2.5-0.5B** — Small, ungated, 0 Armenian tokens (clean surgery, no conflicting fragments), good English performance (fertility 1.29).
2. **Gemma-2-2B** — Slightly better base Armenian (246 tokens) but gated and larger.
3. **mGPT-Armenian** — Strategically interesting because it already has Armenian-tuned weights but a catastrophically broken tokenizer. Grafting could unlock latent Armenian capability.

### Goal 4 (Recovery Fine-tuning)
The cleaned 12M-line CC-100 corpus provides ample data. Recovery training should be relatively fast given the small model sizes (0.5B–2B parameters).

### Goal 5 (Evaluation)
Perplexity comparison will be the primary metric. We expect dramatic improvement when grafting a proper Armenian tokenizer onto byte-fallback models. The HY/EN fertility ratio provides a built-in sanity check: post-surgery fertility should drop from 7–14 range to the 2–3 range.

---

## Project Log — What Was Done

### Data Acquisition
- Downloaded CC-100 Armenian corpus (`hy.txt`, 5.91 GB, 25.3M lines)
- Extracted 30k-line raw sample (`hy_sample_raw.txt`) via `head -30000 hy.txt`

### Corpus Cleaning (parallel task)
- Ran `clean_armenian_corpus.py` on the full 5.91 GB corpus
- Pipeline: NFC normalization → HTML removal → URL removal → MD5 deduplication → Armenian character ratio filter (40%) → minimum length filter (25 chars)
- Result: **12,129,443 clean lines** from 25,355,483 total (47.8% retention)
- 4,832,970 duplicate lines removed
- Output: `hy_clean.txt` + `hy_clean.jsonl`

### Sample Preprocessing (Goal 1)
- Light Armenian-safe cleaning on 30k-line sample: NFC normalization, control character removal, HTML/URL stripping
- Preserved all Armenian punctuation and particles
- 25,621 of 30,000 lines passed the Armenian content filter (85.4%)

### Fertility Analysis (Goal 1)
- Run 1: Loaded 6/9 tokenizers (3 gated models skipped)
- Accepted licenses for LLaMA-2, LLaMA-3, Gemma-2 on HuggingFace
- Created HF Read access token and authenticated via `huggingface-cli login`
- Run 2: Loaded **9/9 tokenizers** successfully
- Computed fertility, bytes/token, STRR, severe fragmentation, English baseline, Armenian vocab count
- Results saved to `fertility_results.json`

### Goal 1 Status: COMPLETE
All 9 tokenizers analyzed. Key findings documented. Data and results saved for downstream use.

### Next Steps
- **Goal 2**: Train custom Armenian BPE and Unigram tokenizers on `hy_clean.txt`
- **Goal 3**: Graft the best custom tokenizer onto Qwen2.5-0.5B (or mGPT-Armenian)
- **Goal 4**: Recovery fine-tuning with LoRA on the cleaned Armenian corpus
- **Goal 5**: Evaluate perplexity, classification, and generation quality
