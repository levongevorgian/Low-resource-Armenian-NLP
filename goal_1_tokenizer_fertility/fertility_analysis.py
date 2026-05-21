#!/usr/bin/env python3
"""
Goal 1: Tokenizer Fertility Analysis for Armenian Text
Usage: python3 fertility_analysis.py --input hy_sample_raw.txt
Requirements: pip install transformers sentencepiece protobuf tqdm tabulate
"""

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

from tqdm import tqdm

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "sample" / "hy_sample_raw.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "goal1" / "fertility_results.json"

# ============================================================
# 1. ARMENIAN-SAFE PREPROCESSING
# ============================================================

NOISE_CODEPOINTS = set()
NOISE_CODEPOINTS.update(range(0x0000, 0x0009))
NOISE_CODEPOINTS.update(range(0x000B, 0x000D))
NOISE_CODEPOINTS.update(range(0x000E, 0x0020))
NOISE_CODEPOINTS.add(0x007F)
NOISE_CODEPOINTS.update(range(0x0080, 0x00A0))
NOISE_CODEPOINTS.add(0x00AD)
NOISE_CODEPOINTS.update([0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF])


def is_armenian_char(c):
    cp = ord(c)
    return (0x0530 <= cp <= 0x058F) or (0xFB13 <= cp <= 0xFB17)


def clean_line_safe(line):
    line = unicodedata.normalize("NFC", line)
    line = "".join(c for c in line if ord(c) not in NOISE_CODEPOINTS)
    line = re.sub(r"<[^>]+>", " ", line)
    line = re.sub(r"https?://\S+", " ", line)
    line = re.sub(r"[^\S\n]+", " ", line)
    return line.strip()


def is_armenian_line(line, min_ratio=0.3, min_arm_chars=10):
    if not line or len(line) < 15:
        return False
    arm_count = sum(1 for c in line if is_armenian_char(c))
    if arm_count < min_arm_chars:
        return False
    alpha_chars = sum(1 for c in line if c.isalpha())
    if alpha_chars == 0:
        return False
    return (arm_count / alpha_chars) >= min_ratio


def preprocess_corpus(input_path, max_lines=None):
    lines = []
    skipped = 0
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for i, raw_line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            cleaned = clean_line_safe(raw_line)
            if is_armenian_line(cleaned):
                lines.append(cleaned)
            else:
                skipped += 1
    print("Preprocessing: kept {:,} lines, skipped {:,}".format(len(lines), skipped))
    return lines


# ============================================================
# 2. TOKENIZER REGISTRY
# ============================================================


def build_tokenizer_registry():
    registry = {}

    registry["GPT-2"] = dict(
        hf_name="openai-community/gpt2",
        tok_type="BPE",
        vocab_size="50k",
        notes="English-only BPE, byte-level fallback",
        gated=False,
    )
    registry["LLaMA-2-7B"] = dict(
        hf_name="meta-llama/Llama-2-7b-hf",
        tok_type="BPE (SP)",
        vocab_size="32k",
        notes="English-centric SentencePiece",
        gated=True,
    )
    registry["LLaMA-3-8B"] = dict(
        hf_name="meta-llama/Meta-Llama-3-8B",
        tok_type="BPE (tiktoken)",
        vocab_size="128k",
        notes="Expanded vocab, better multilingual",
        gated=True,
    )
    registry["Mistral-v0.3"] = dict(
        hf_name="mistralai/Mistral-7B-v0.3",
        tok_type="BPE (SP)",
        vocab_size="32.8k",
        notes="May miss Armenian alphabet chars",
        gated=True,
    )
    registry["Qwen2.5-0.5B"] = dict(
        hf_name="Qwen/Qwen2.5-0.5B",
        tok_type="BPE",
        vocab_size="152k",
        notes="Large vocab, CJK-heavy but broad",
        gated=False,
    )
    registry["Gemma-2-2B"] = dict(
        hf_name="google/gemma-2-2b",
        tok_type="BPE (SP)",
        vocab_size="256k",
        notes="Largest open vocab, best chance for Armenian",
        gated=True,
    )
    registry["mT5-small"] = dict(
        hf_name="google/mt5-small",
        tok_type="Unigram (SP)",
        vocab_size="250k",
        notes="Trained on CC-100 (includes Armenian)",
        gated=False,
    )
    registry["XLM-R-base"] = dict(
        hf_name="FacebookAI/xlm-roberta-base",
        tok_type="Unigram (SP)",
        vocab_size="250k",
        notes="Trained on CC-100 (includes Armenian)",
        gated=False,
    )
    registry["mGPT-Armenian"] = dict(
        hf_name="ai-forever/mGPT-1.3B-armenian",
        tok_type="BPE (SP)",
        vocab_size="100k",
        notes="Fine-tuned on Armenian, mGPT base vocab",
        gated=False,
    )
    return registry


def load_tokenizer(name, info):
    try:
        tok = AutoTokenizer.from_pretrained(
            info["hf_name"],
            trust_remote_code=True,
            use_fast=True,
        )
        return tok
    except OSError as e:
        msg = str(e).lower()
        if "gated" in msg or "401" in str(e) or "403" in str(e):
            print("  [SKIP] {}: gated model (run huggingface-cli login)".format(name))
        else:
            print("  [SKIP] {}: load failed - {}".format(name, e))
        return None
    except Exception as e:
        print("  [SKIP] {}: load failed - {}".format(name, e))
        return None


# ============================================================
# 3. FERTILITY METRICS
# ============================================================


def armenian_word_tokenize(text):
    raw_words = text.split()
    words = [w for w in raw_words if any(c.isalpha() for c in w)]
    return words


def compute_metrics(tokenizer, lines, tokenizer_name):
    total_tokens = 0
    total_words = 0
    total_bytes = 0
    single_token_words = 0
    word_fertilities = []

    for line in lines:
        words = armenian_word_tokenize(line)
        if not words:
            continue

        line_tokens = tokenizer.encode(line, add_special_tokens=False)
        total_tokens += len(line_tokens)
        total_bytes += len(line.encode("utf-8"))
        total_words += len(words)

        for word in words:
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            n_tok = len(word_tokens)
            word_fertilities.append(n_tok)
            if n_tok == 1:
                single_token_words += 1

    if total_words == 0 or total_tokens == 0:
        return None

    fertility = total_tokens / total_words
    bytes_per_token = total_bytes / total_tokens
    strr = single_token_words / total_words

    median_fert = statistics.median(word_fertilities) if word_fertilities else 0
    max_fert = max(word_fertilities) if word_fertilities else 0
    severe_frag = sum(1 for f in word_fertilities if f >= 5)
    severe_frag_pct = severe_frag / total_words

    return dict(
        tokenizer=tokenizer_name,
        total_tokens=total_tokens,
        total_words=total_words,
        fertility=round(fertility, 3),
        bytes_per_token=round(bytes_per_token, 3),
        strr=round(strr, 4),
        median_fertility=median_fert,
        max_word_fertility=max_fert,
        severe_fragmentation_pct=round(severe_frag_pct, 4),
    )


def count_armenian_vocab_tokens(tokenizer):
    vocab = tokenizer.get_vocab()
    armenian_tokens = 0
    pure_armenian_tokens = 0

    for token_str in vocab:
        clean = token_str.replace("\u2581", "").replace("\u0120", "")
        if not clean:
            continue
        has_armenian = any(is_armenian_char(c) for c in clean)
        if has_armenian:
            armenian_tokens += 1
            alpha_chars = [c for c in clean if c.isalpha()]
            if alpha_chars and all(is_armenian_char(c) for c in alpha_chars):
                pure_armenian_tokens += 1

    return dict(
        total_vocab=len(vocab),
        armenian_tokens=armenian_tokens,
        pure_armenian_tokens=pure_armenian_tokens,
        armenian_pct=round(armenian_tokens / max(len(vocab), 1) * 100, 3),
    )


def show_sample_tokenization(tokenizer, name, sample_sentence):
    tokens = tokenizer.encode(sample_sentence, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    print("\n  {}:".format(name))
    print("    Input:  {}".format(sample_sentence))
    preview = decoded[:30]
    suffix = "..." if len(tokens) > 30 else ""
    print("    Tokens ({}): {}{}".format(len(tokens), preview, suffix))


# ============================================================
# 4. ENGLISH BASELINE
# ============================================================

ENGLISH_SAMPLE = [
    (
        "The Republic of Armenia is a landlocked country in the Armenian "
        "Highlands of Western Asia."
    ),
    (
        "It is bordered by Turkey to the west, Georgia to the north, "
        "Azerbaijan to the east and south."
    ),
    (
        "Armenia is a unitary multi-party democratic nation-state with an "
        "ancient cultural heritage."
    ),
    "The Armenian alphabet was created in 405 AD by Mesrop Mashtots.",
    "The country has a population of approximately three million people.",
    "Yerevan is the capital and largest city of Armenia.",
    "Mount Ararat is a prominent national symbol of Armenia.",
    (
        "Armenia was the first country to adopt Christianity as a state "
        "religion in 301 AD."
    ),
    "The economy of Armenia is primarily based on industry agriculture and services.",
    (
        "The Armenian language is an independent branch of the Indo-European "
        "language family."
    ),
]


# ============================================================
# 5. MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Armenian Tokenizer Fertility Analysis"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to raw Armenian text file",
    )
    parser.add_argument(
        "--max-lines", type=int, default=None, help="Max lines to process (None = all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    TOKENIZER_REGISTRY = build_tokenizer_registry()

    # --- Preprocess ---
    print("=" * 70)
    print("STEP 1: Preprocessing Armenian text")
    print("=" * 70)
    lines = preprocess_corpus(args.input, max_lines=args.max_lines)

    if len(lines) < 100:
        print("ERROR: Only {} usable lines. Need at least 100.".format(len(lines)))
        sys.exit(1)

    total_chars = sum(len(l) for l in lines)
    total_words = sum(len(armenian_word_tokenize(l)) for l in lines)
    total_bytes = sum(len(l.encode("utf-8")) for l in lines)
    print(
        "Corpus stats: {:,} lines, {:,} words, {:,} chars, {:,} bytes".format(
            len(lines), total_words, total_chars, total_bytes
        )
    )

    sample_sentences = [l for l in lines if 40 < len(l) < 120]
    sample_sent = sample_sentences[0] if sample_sentences else lines[0]

    # --- Load tokenizers ---
    print("")
    print("=" * 70)
    print("STEP 2: Loading tokenizers")
    print("=" * 70)
    loaded = {}
    for name in TOKENIZER_REGISTRY:
        info = TOKENIZER_REGISTRY[name]
        print("  Loading {} ({})...".format(name, info["hf_name"]))
        tok = load_tokenizer(name, info)
        if tok is not None:
            loaded[name] = tok
            print("    OK - vocab size: {:,}".format(len(tok.get_vocab())))

    if not loaded:
        print("ERROR: No tokenizers loaded. Check internet / HF login.")
        sys.exit(1)

    print("\nLoaded {}/{} tokenizers.".format(len(loaded), len(TOKENIZER_REGISTRY)))

    # --- Compute metrics ---
    print("")
    print("=" * 70)
    print("STEP 3: Computing fertility metrics (Armenian)")
    print("=" * 70)

    all_results = []
    for name in loaded:
        tok = loaded[name]
        print("\n  Analyzing {}...".format(name))
        t0 = time.time()

        metrics = compute_metrics(tok, lines, name)
        if metrics is None:
            print("    FAILED - no valid data")
            continue

        vocab_info = count_armenian_vocab_tokens(tok)
        metrics["vocab_total"] = vocab_info["total_vocab"]
        metrics["vocab_armenian"] = vocab_info["armenian_tokens"]
        metrics["vocab_pure_armenian"] = vocab_info["pure_armenian_tokens"]
        metrics["vocab_armenian_pct"] = vocab_info["armenian_pct"]

        eng_metrics = compute_metrics(tok, ENGLISH_SAMPLE, name)
        if eng_metrics is not None:
            metrics["english_fertility"] = eng_metrics["fertility"]
            metrics["english_bytes_per_token"] = eng_metrics["bytes_per_token"]
            metrics["fertility_ratio_hy_en"] = round(
                metrics["fertility"] / eng_metrics["fertility"], 2
            )

        metrics["tok_type"] = TOKENIZER_REGISTRY[name]["tok_type"]
        metrics["notes"] = TOKENIZER_REGISTRY[name]["notes"]

        elapsed = time.time() - t0
        print(
            "    OK - fertility={:.2f}, bytes/tok={:.2f}, STRR={:.3f}, "
            "Armenian vocab={:,} ({:.1f}s)".format(
                metrics["fertility"],
                metrics["bytes_per_token"],
                metrics["strr"],
                vocab_info["armenian_tokens"],
                elapsed,
            )
        )

        all_results.append(metrics)

    # --- Sample tokenization ---
    print("")
    print("=" * 70)
    print("STEP 4: Sample tokenization demo")
    print("=" * 70)
    print("Sentence: {}".format(sample_sent))
    for name in loaded:
        show_sample_tokenization(loaded[name], name, sample_sent)

    # --- Summary table ---
    print("")
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    all_results.sort(key=lambda x: x["fertility"])

    headers = [
        "Tokenizer",
        "Type",
        "Vocab",
        "HY Tokens",
        "Fertility",
        "Bytes/Tok",
        "STRR",
        "Severe%",
        "EN Fert.",
        "HY/EN",
    ]
    rows = []
    for r in all_results:
        en_fert = r.get("english_fertility", "N/A")
        ratio = r.get("fertility_ratio_hy_en", "N/A")
        if ratio != "N/A":
            ratio = str(ratio) + "x"
        rows.append(
            [
                r["tokenizer"],
                r.get("tok_type", ""),
                "{:,}".format(r["vocab_total"]),
                "{:,}".format(r["vocab_armenian"]),
                "{:.2f}".format(r["fertility"]),
                "{:.2f}".format(r["bytes_per_token"]),
                "{:.3f}".format(r["strr"]),
                "{:.1f}%".format(r["severe_fragmentation_pct"] * 100),
                str(en_fert),
                str(ratio),
            ]
        )

    if tabulate is not None:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        header_line = "  ".join(h.ljust(14) for h in headers)
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print("  ".join(str(v).ljust(14) for v in row))

    # --- Interpretation ---
    print("")
    print("=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    best = all_results[0]
    worst = all_results[-1]
    print(
        "  Best fertility:  {} ({:.2f} tokens/word)".format(
            best["tokenizer"], best["fertility"]
        )
    )
    print(
        "  Worst fertility: {} ({:.2f} tokens/word)".format(
            worst["tokenizer"], worst["fertility"]
        )
    )
    spread = worst["fertility"] / best["fertility"]
    print("  Fertility spread: {:.1f}x between best and worst".format(spread))
    print("")
    print("  Key insight: A fertility of N means Armenian text costs ~Nx more")
    print("  context window and compute than equivalent English text would")
    print("  with an English-optimized tokenizer (EN fertility is about 1.2-1.5).")
    print("")
    print("  STRR > 0.10 means the tokenizer has meaningful Armenian vocabulary.")
    print("  STRR < 0.05 means almost every word gets fragmented.")

    # --- Save results ---
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                corpus_stats=dict(
                    file=args.input,
                    lines=len(lines),
                    words=total_words,
                    total_bytes=total_bytes,
                ),
                results=all_results,
                sample_sentence=sample_sent,
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\n  Results saved to: {}".format(output_path))


if __name__ == "__main__":
    main()
