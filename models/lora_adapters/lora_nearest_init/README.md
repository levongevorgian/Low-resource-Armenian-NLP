---
base_model: models/grafted_tokenizers/nearest_init
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- transformers
- armenian
---

# LoRA Adapter: Nearest-Token Initialization

This directory contains the compact PEFT/LoRA adapter produced for the nearest-token-initialized grafted tokenizer experiment.

Expected base artifact within this repository:

```text
models/grafted_tokenizers/nearest_init
```

The full base-model weights are not included in this repository.
