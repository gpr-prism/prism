# Data

This directory is the expected location for pre-tokenized causal language
modeling data.

`train.py` checks for the following structure:

```text
data/
├── train/
│   ├── data-00000-of-NNNNN.arrow
│   └── ...
├── validation/
│   └── data-00000-of-00001.arrow
└── test/
    └── data-00000-of-00001.arrow
```

Each Arrow file should contain either an `input_ids` column with token ID lists
or a `text` column that can be tokenized by the configured tokenizer.

If this structure is absent, `train.py` falls back to WikiText-103 raw data. The
fallback is useful for smoke tests, but it is not the large-scale SlimPajama
setting used by the reference LM numbers.

## Tokenizer

The default training script uses GPT-2 tokenization:

```python
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
```

If `ft_local/tokenizer_gpt2/` exists, that local tokenizer is used instead.
Keep the tokenizer fixed across preprocessing, training, and evaluation.

## Example SlimPajama Preparation

The following script streams text from SlimPajama, tokenizes it with GPT-2, and
writes Arrow files compatible with `train.py`.

```python
# save as prepare_data.py and run from llm/: python prepare_data.py

from datasets import load_dataset
from transformers import GPT2Tokenizer
import pyarrow as pa
import os

TOKENIZER_NAME = "gpt2"
SEQ_LEN = 1024
TARGET_TRAIN_TOKENS = 2_000_000_000
FLUSH_TOKENS = 100_000_000

tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)

def write_split(name, token_budget=None):
    os.makedirs(f"data/{name}", exist_ok=True)
    ids, total, file_idx = [], 0, 0
    src = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)

    for item in src:
        text = item.get("text", "").strip()
        if not text:
            continue

        chunk = tokenizer.encode(text, add_special_tokens=False)
        ids.extend(chunk)
        total += len(chunk)

        while len(ids) >= FLUSH_TOKENS:
            flush = ids[:FLUSH_TOKENS]
            ids = ids[FLUSH_TOKENS:]
            table = pa.table({"input_ids": [flush]})
            path = f"data/{name}/data-{file_idx:05d}.arrow"
            with pa.ipc.new_file(path, table.schema) as writer:
                writer.write_table(table)
            print(f"[{name}] wrote {path} ({len(flush):,} tokens)")
            file_idx += 1

        if token_budget and total >= token_budget:
            break

    if ids:
        table = pa.table({"input_ids": [ids]})
        path = f"data/{name}/data-{file_idx:05d}.arrow"
        with pa.ipc.new_file(path, table.schema) as writer:
            writer.write_table(table)
        print(f"[{name}] wrote {path} ({len(ids):,} tokens)")

    print(f"[{name}] total tokens: {total:,}")

write_split("train", token_budget=TARGET_TRAIN_TOKENS)
```

Create validation and test splits with the same tokenizer and Arrow schema.
For faster repeated runs, download the raw dataset files locally instead of
streaming every time.

