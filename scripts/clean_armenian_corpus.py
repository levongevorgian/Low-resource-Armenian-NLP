import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import regex as re  # better Unicode support than std re
from tqdm import tqdm

# Optional: for language ID
# import fasttext
# model = fasttext.load_model('lid.176.ftz')  # download if needed


class ArmenianCorpusCleaner:
    def __init__(
        self,
        input_path: str,
        output_path: str = None,
        min_len=30,
        dedup_sentences=True,
        output_jsonl=False,
    ):
        self.input_path = Path(input_path)
        self.output_path = (
            Path(output_path)
            if output_path
            else self.input_path.with_name("hy_clean.txt")
        )
        self.jsonl_path = (
            self.output_path.with_suffix(".jsonl") if output_jsonl else None
        )
        self.min_len = min_len
        self.dedup_sentences = dedup_sentences
        self.seen_hashes = set()  # For exact sentence dedup

    def normalize_armenian(self, text: str) -> str:
        """Armenian-specific normalization"""
        # Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)

        # Replace common variants / fix spacing
        text = re.sub(r"\s+", " ", text.strip())

        # Remove control characters, excessive symbols
        text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", text)
        return text

    def is_good_line(self, line: str) -> bool:
        """Quality filters"""
        if len(line) < self.min_len:
            return False
        if len(line.split()) < 5:  # too few words
            return False

        # Armenian character ratio (rough filter)
        arm_chars = sum(
            1 for c in line if "\u0530" <= c <= "\u058F" or "\uFB00" <= c <= "\uFB17"
        )
        if arm_chars / max(len(line), 1) < 0.4:  # at least 40% Armenian script
            return False

        return True

    def clean_line(self, line: str) -> str:
        """Full cleaning pipeline"""
        line = line.strip()
        if not line:
            return None

        # Remove HTML remnants
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"https?://\S+", "", line)

        line = self.normalize_armenian(line)

        if not self.is_good_line(line):
            return None

        return line

    def sentence_hash(self, sentence: str) -> str:
        """Fast hash for deduplication"""
        return hashlib.md5(sentence.lower().encode("utf-8")).hexdigest()

    def process(self):
        """Main processing loop"""
        seen_count = 0
        kept_count = 0
        total_lines = 0  # Approximate with tqdm

        with open(self.input_path, "r", encoding="utf-8", errors="ignore") as fin, open(
            self.output_path, "w", encoding="utf-8"
        ) as fout:

            if self.jsonl_path:
                jsonl_file = open(self.jsonl_path, "w", encoding="utf-8")

            for line in tqdm(fin, desc="Processing corpus"):
                total_lines += 1
                cleaned = self.clean_line(line)
                if not cleaned:
                    continue

                # Sentence-level deduplication
                if self.dedup_sentences:
                    h = self.sentence_hash(cleaned)
                    if h in self.seen_hashes:
                        seen_count += 1
                        continue
                    self.seen_hashes.add(h)

                fout.write(cleaned + "\n")
                kept_count += 1

                if self.jsonl_path:
                    json.dump(
                        {"text": cleaned, "id": kept_count},
                        jsonl_file,
                        ensure_ascii=False,
                    )
                    jsonl_file.write("\n")

            if self.jsonl_path:
                jsonl_file.close()

        print("\nDone.")
        print(f"Total lines processed: {total_lines:,}")
        print(f"Kept lines: {kept_count:,}")
        if self.dedup_sentences:
            print(f"Duplicates removed: {seen_count:,}")
        print(f"Output: {self.output_path}")
        if self.jsonl_path:
            print(f"JSONL: {self.jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean and deduplicate Armenian corpus text."
    )
    parser.add_argument("input_path", help="Raw Armenian text file to clean")
    parser.add_argument(
        "--output",
        default=None,
        help="Output text path; defaults to hy_clean.txt next to input",
    )
    parser.add_argument(
        "--min-len", type=int, default=25, help="Minimum cleaned line length"
    )
    parser.add_argument(
        "--no-dedup", action="store_true", help="Disable exact sentence deduplication"
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Also write a JSONL version next to the text output",
    )
    args = parser.parse_args()

    cleaner = ArmenianCorpusCleaner(
        input_path=args.input_path,
        output_path=args.output,
        min_len=args.min_len,
        dedup_sentences=not args.no_dedup,
        output_jsonl=args.jsonl,
    )
    cleaner.process()
