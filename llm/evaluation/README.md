# Evaluation

This directory documents the evaluation protocol for the LM implementation.

## Built-In Perplexity Evaluation

`train.py` reports validation and test perplexity for the configured dataset:

```bash
python train.py \
  --models prism \
  --config large_130m \
  --eval_only \
  --ckpt_dir checkpoints/prism
```

Expected checkpoint names are:

```text
checkpoints/prism/prism_large_130m_best.pt
```

The saved files are raw PyTorch state dictionaries.

## Downstream Benchmarks

The original LM reference table reports LAMBADA and zero-shot accuracy on PIQA,
HellaSwag, Winogrande, ARC, BoolQ, OpenBookQA, and SciQ. To reproduce those with
`lm-evaluation-harness`, first export or wrap the model as a HuggingFace
compatible causal LM. The raw `*.pt` checkpoints written by `train.py` are not
directly loadable via:

```bash
lm_eval --model hf --model_args pretrained=checkpoints/prism
```

After exporting a HuggingFace-compatible directory, use:

```bash
lm_eval --model hf \
  --model_args pretrained=/path/to/exported/prism \
  --tasks lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,openbookqa,sciq \
  --device cuda \
  --batch_size 16
```

## Reference Setup

| Setting | Value |
|---------|-------|
| Config | `large_130m` |
| Tokenizer | GPT-2 by default, or `ft_local/tokenizer_gpt2` if present |
| Sequence length | 1024 in `large_130m` |
| Optimizer | AdamW |
| Learning rate | 6e-5 |
| Warmup steps | 1500 |
| Gradient clip | 0.5 |

The larger reference run used pre-tokenized large-scale text data under
`data/`. Without that directory, the public WikiText-103 fallback is used.

